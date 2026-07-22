import 'dart:async';
import 'dart:io' show Platform;
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:wakelock_plus/wakelock_plus.dart';

import 'heuristics/heuristics_engine.dart';
import 'heuristics/fencenet_channel.dart';
import 'pose/pose_service.dart';
import 'pose/pose_painter.dart';
import 'pose/yolo_pose_service.dart';
import 'pose/activity_gatekeeper.dart';
import 'package:uuid/uuid.dart';
import 'database/entities.dart';
import 'database/app_database.dart';
import 'screens/history_screen.dart';
import 'screens/postgame_screen.dart';
import 'ai/gemini_agent.dart';

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

List<CameraDescription> _cameras = [];

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  _cameras = await availableCameras();
  runApp(const FencingCoachApp());
}

// ---------------------------------------------------------------------------
// App root
// ---------------------------------------------------------------------------

class FencingCoachApp extends StatelessWidget {
  const FencingCoachApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'AI Fencing Coach',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark().copyWith(
        scaffoldBackgroundColor: const Color(0xFF0A0A0F),
        cardColor: const Color(0xFF141420),
        colorScheme: const ColorScheme.dark(
          primary: Color(0xFFFF6600),
          secondary: Color(0xFF00D4FF),
          surface: Color(0xFF141420),
          error: Color(0xFFFF3D3D),
        ),
        appBarTheme: const AppBarTheme(
          backgroundColor: Color(0xFF0A0A0F),
          elevation: 0,
          titleTextStyle: TextStyle(
            color: Colors.white,
            fontSize: 18,
            fontWeight: FontWeight.w700,
            letterSpacing: 0.5,
          ),
        ),
        tabBarTheme: const TabBarThemeData(
          indicatorColor: Color(0xFFFF6600),
          labelColor: Color(0xFFFF6600),
          unselectedLabelColor: Colors.white54,
        ),
      ),
      home: const MainScreen(),
    );
  }
}

// ---------------------------------------------------------------------------
// Error display labels (Chinese/English)
// ---------------------------------------------------------------------------

const Map<String, String> kErrorLabels = {
  'lunge_overextension': '長刺過度前傾 (Lunge Overextension)',
  'guard_dropped': '持劍手掉落 (Guard Dropped)',
  'bounce_excessive': '步伐上下浮動 (Excessive Bounce)',
  'foot_before_hand': '手腳順序錯誤 (Foot Before Hand)',
  'over_parrying': '防守動作太大 (Over-Parrying)',
  'stance_too_high': '預備姿勢沒蹲好 (Stance Too High)',
  'incomplete_arm_extension': '手沒有伸直 (Incomplete Extension)',
  'wide_step': '步伐太大 (Wide Step)',
  'narrow_step': '步伐太小 (Narrow Step)',
  'center_of_mass_in_front': '重心向前 (CoM Forward)',
  'center_of_mass_leaning_backward': '重心向後 (CoM Backward)',
};

const Map<String, String> kErrorVoice = {
  'lunge_overextension': '長刺過度前傾，注意不要彎太低',
  'guard_dropped': '持劍手掉落，請抬高手腕',
  'bounce_excessive': '步伐上下浮動，保持重心穩定',
  'foot_before_hand': '腳比手先動，先出手再移步',
  'over_parrying': '防守動作太大，縮小防守範圍',
  'stance_too_high': '預備姿勢太高，請蹲低一點',
  'incomplete_arm_extension': '手沒有伸直，刺出時請完全伸展手臂',
  'wide_step': '步伐太大，縮小步距',
  'narrow_step': '步伐太小，加大步距',
  'center_of_mass_in_front': '重心偏前，保持重心平衡',
  'center_of_mass_leaning_backward': '重心偏後，向前調整重心',
};

const Map<String, List<String>> kErrorSupportedModes = {
  'foot_before_hand': ['Target Practice'],
  'lunge_overextension': ['Target Practice'],
  'incomplete_arm_extension': ['Target Practice'],
  'guard_dropped': ['Footwork', 'Target Practice', 'Free Bouting'],
  'stance_too_high': ['Footwork', 'Target Practice', 'Free Bouting'],
  'bounce_excessive': ['Footwork', 'Target Practice', 'Free Bouting'],
  'center_of_mass_in_front': ['Footwork', 'Target Practice', 'Free Bouting'],
  'center_of_mass_leaning_backward': ['Footwork', 'Target Practice', 'Free Bouting'],
  'over_parrying': ['Footwork', 'Target Practice', 'Free Bouting'],
  'wide_step': ['Footwork', 'Target Practice', 'Free Bouting'],
  'narrow_step': ['Footwork', 'Target Practice', 'Free Bouting'],
};

// ---------------------------------------------------------------------------
// Main Screen
// ---------------------------------------------------------------------------

class MainScreen extends StatefulWidget {
  const MainScreen({super.key});
  @override
  State<MainScreen> createState() => _MainScreenState();
}

class _MainScreenState extends State<MainScreen>
    with TickerProviderStateMixin {
  // Tab controller
  late TabController _tabController;

  // Session tracking
  int? _sessionStartTimeMs;
  final Map<String, int> _sessionActionCounts = {};
  final List<CueHistoryItem> _sessionCues = [];

  // Camera
  CameraController? _cameraController;
  bool _isCameraInitialized = false;
  bool _isProcessingFrame = false;

  // Settings
  String _targetSide = 'left';
  String _trainingMode = 'Footwork';
  bool _voiceEnabled = true;
  List<String> _focusErrors = [];
  List<String> _muteErrors = [];
  bool _onlySelected = false;
  CameraLensDirection _cameraLensDirection = CameraLensDirection.back;

  // Pose & inference
  final YoloPoseService _yoloPoseService = YoloPoseService();
  bool _yoloPoseLoaded = false;
  final FenceNetChannel _fenceNet = FenceNetChannel();
  bool _fenceNetLoaded = false;

  // Heuristics
  late HeuristicsEngine _heuristics;

  // Skeleton window buffer (28 frames)
  final List<Skeleton> _skeletonBuffer = [];
  FencingSkeleton? _currentSkeleton;

  // Reference for normalization (nose at t=0 of action)
  Offset? _refNose;
  double? _refScale;

  // FenceNet window buffer (28 skeletons for classification)
  final List<Skeleton> _classifierBuffer = [];

  // Effective pose fps estimate (frames actually processed per second) —
  // used so the heuristics engine's duration-based thresholds stay
  // consistent across devices with different pose throughput.
  final List<int> _poseTimestampsMs = [];

  // Live state
  String _currentAction = 'Idle';
  double _actionConfidence = 0.0;
  List<String> _activeErrors = [];
  late ActivityGatekeeper _gatekeeper;
  String _coachState = 'IDLE'; // 'ACTIVE' | 'IDLE'
  int _frameCount = 0;

  // TTS
  late FlutterTts _tts;
  bool _ttsReady = false;
  bool _isSpeaking = false;
  DateTime _lastSpokenTime = DateTime.fromMillisecondsSinceEpoch(0);

  // Flash animation
  late AnimationController _flashController;
  late Animation<double> _flashAnimation;

  // Debug metrics
  final Map<String, bool> _heuristicTriggered = {};

  @override
  void initState() {
    super.initState();
    // Use length 5 for Live, Postgame, Settings, History, Debug
    _tabController = TabController(length: 5, vsync: this);
    _heuristics = HeuristicsEngine(
      targetSide: _targetSide,
      trainingMode: _trainingMode,
    );
    _gatekeeper = ActivityGatekeeper(fps: 30);

    _flashController = AnimationController(
      duration: const Duration(milliseconds: 600),
      vsync: this,
    );
    _flashAnimation = Tween<double>(begin: 0, end: 1).animate(
      CurvedAnimation(parent: _flashController, curve: Curves.easeInOut),
    );

    _initTts();
    _initCamera();
    _initFenceNet();
    _initYoloPose();
    WakelockPlus.enable();
  }

  @override
  void dispose() {
    _tabController.dispose();
    _flashController.dispose();
    _cameraController?.dispose();
    _fenceNet; // no dispose needed for channel
    _tts.stop();
    WakelockPlus.disable();
    super.dispose();
  }

  // ── TTS ────────────────────────────────────────────────────────────────────

  Future<void> _initTts() async {
    _tts = FlutterTts();

    try {
      if (Platform.isIOS) {
        // Keep spoken cues audible even when the iPhone silent switch is on.
        await _tts.setIosAudioCategory(
          IosTextToSpeechAudioCategory.playback,
          [
            IosTextToSpeechAudioCategoryOptions.defaultToSpeaker,
          ],
          IosTextToSpeechAudioMode.defaultMode,
        );
      }

      bool hasZhTw = false;
      try {
        hasZhTw = await _tts.isLanguageAvailable('zh-TW');
      } catch (_) {}

      if (hasZhTw) {
        await _tts.setLanguage('zh-TW');
      } else {
        bool hasZhCn = false;
        try {
          hasZhCn = await _tts.isLanguageAvailable('zh-CN');
        } catch (_) {}
        if (hasZhCn) {
          await _tts.setLanguage('zh-CN');
        } else {
          await _tts.setLanguage('en-US');
        }
      }

      await _tts.setSpeechRate(0.5);
      await _tts.setVolume(1.0);
      await _tts.awaitSpeakCompletion(false);
      _tts.setStartHandler(() => _isSpeaking = true);
      _tts.setCompletionHandler(() => _isSpeaking = false);
      _tts.setCancelHandler(() => _isSpeaking = false);
      _tts.setErrorHandler((_) => _isSpeaking = false);
      _ttsReady = true;
    } catch (e) {
      debugPrint('TTS init error: $e');
      _ttsReady = false;
    }
  }

  Future<void> _speak(String text) async {
    final cue = text.trim();
    if (!_voiceEnabled || !_ttsReady || _isSpeaking || cue.isEmpty) return;
    final now = DateTime.now();
    if (now.difference(_lastSpokenTime).inSeconds < 4) return;
    _lastSpokenTime = now;
    _isSpeaking = true;
    try {
      await _tts.speak(cue);
    } catch (e) {
      debugPrint('TTS speak error: $e');
      _isSpeaking = false;
    }
  }

  // ── Camera ─────────────────────────────────────────────────────────────────

  Future<void> _initCamera() async {
    if (_cameras.isEmpty) return;
    final cameraPermission = await Permission.camera.request();
    if (!cameraPermission.isGranted) {
      debugPrint('Camera permission denied');
      return;
    }

    // Prefer selected lens direction
    CameraDescription cam = _cameras.first;
    for (final c in _cameras) {
      if (c.lensDirection == _cameraLensDirection) {
        cam = c;
        break;
      }
    }

    _cameraController = CameraController(
      cam,
      ResolutionPreset.medium, // 640×480 good for ML Kit speed
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.bgra8888, // iOS native
    );

    try {
      await _cameraController!.initialize();
      if (!mounted) return;
      _resetLiveBuffers();
      setState(() => _isCameraInitialized = true);
      _cameraController!.startImageStream(_onCameraFrame);
    } catch (e) {
      debugPrint('Camera init error: $e');
    }
  }

  Future<void> _switchCamera(CameraLensDirection direction) async {
    if (_cameras.isEmpty) return;

    CameraDescription? targetCam;
    for (final c in _cameras) {
      if (c.lensDirection == direction) {
        targetCam = c;
        break;
      }
    }

    targetCam ??= _cameras.first;

    if (_cameraController != null) {
      try {
        await _cameraController!.stopImageStream();
      } catch (e) {
        debugPrint('Error stopping image stream: $e');
      }
      await _cameraController!.dispose();
    }

    setState(() {
      _isCameraInitialized = false;
      _cameraLensDirection = direction;
    });

    _cameraController = CameraController(
      targetCam,
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.bgra8888,
    );

    try {
      await _cameraController!.initialize();
      if (!mounted) return;
      _resetLiveBuffers();
      setState(() => _isCameraInitialized = true);
      _cameraController!.startImageStream(_onCameraFrame);
    } catch (e) {
      debugPrint('Camera switch error: $e');
    }
  }

  // ── FenceNet ────────────────────────────────────────────────────────────────

  Future<void> _initFenceNet() async {
    final loaded = await _fenceNet.load();
    if (mounted) {
      setState(() => _fenceNetLoaded = loaded);
    }
  }

  // ── YOLOv8-Pose ─────────────────────────────────────────────────────────────

  Future<void> _initYoloPose() async {
    final loaded = await _yoloPoseService.load();
    if (mounted) {
      setState(() => _yoloPoseLoaded = loaded);
    }
  }

  // ── Camera Frame Processing ─────────────────────────────────────────────────

  Future<void> _onCameraFrame(CameraImage image) async {
    if (_isProcessingFrame) return;
    _isProcessingFrame = true;

    try {
      if (!_yoloPoseLoaded) return;

      final plane = image.planes[0];
      final skeleton = await _yoloPoseService.processImageBytes(
        bytes: plane.bytes,
        width: image.width,
        height: image.height,
        bytesPerRow: plane.bytesPerRow,
        targetSide: _targetSide,
        isFrontCamera: _cameraController?.description.lensDirection == CameraLensDirection.front,
      );

      if (!mounted) return;

      if (skeleton != null) {
        _frameCount++;
        _currentSkeleton = skeleton;

        _poseTimestampsMs.add(DateTime.now().millisecondsSinceEpoch);
        if (_poseTimestampsMs.length > 30) {
          _poseTimestampsMs.removeAt(0);
        }

        // Add to heuristic buffer
        _skeletonBuffer.add(skeleton.joints);
        if (_skeletonBuffer.length > 60) {
          _skeletonBuffer.removeAt(0);
        }

        // Add to classifier buffer
        _classifierBuffer.add(skeleton.joints);
        if (_classifierBuffer.length > 28) {
          _classifierBuffer.removeAt(0);
        }

        // Set reference for normalization on first frame
        if (_refNose == null && skeleton.nose != null) {
          _refNose = skeleton.nose;
          _refScale = skeleton.scale;
        }

        // Classify action every 10 frames
        if (_frameCount % 10 == 0 && _classifierBuffer.length == 28) {
          await _classifyAndEvaluate();
        }

        // Update coachState
        final isActive = _gatekeeper.update(_currentSkeleton, _targetSide);
        final newState = isActive ? 'ACTIVE' : 'IDLE';
        if (newState != _coachState) {
          setState(() => _coachState = newState);
        }
      } else {
        final isActive = _gatekeeper.update(null, _targetSide);
        final newState = isActive ? 'ACTIVE' : 'IDLE';
        if (newState != _coachState) {
          if (newState == 'ACTIVE') {
            _sessionStartTimeMs = DateTime.now().millisecondsSinceEpoch;
            _sessionActionCounts.clear();
            _sessionCues.clear();
          } else if (newState == 'IDLE') {
            if (_sessionStartTimeMs != null) {
              final endTimeMs = DateTime.now().millisecondsSinceEpoch;
              final elapsedSeconds = (endTimeMs - _sessionStartTimeMs!) ~/ 1000;
              if (elapsedSeconds > 10) {
                _saveSessionWithGemini(
                  startTimeMs: _sessionStartTimeMs!,
                  endTimeMs: endTimeMs,
                  elapsedSeconds: elapsedSeconds,
                  actionCounts: Map.from(_sessionActionCounts),
                  cues: List.from(_sessionCues),
                  trainingMode: _trainingMode,
                  targetSide: _targetSide,
                );
              }
              _sessionStartTimeMs = null;
            }
          }
          setState(() {
            _coachState = newState;
            if (newState == 'IDLE') _currentSkeleton = null;
          });
        }
      }
    } finally {
      _isProcessingFrame = false;
    }
  }

  Future<void> _saveSessionWithGemini({
    required int startTimeMs,
    required int endTimeMs,
    required int elapsedSeconds,
    required Map<String, int> actionCounts,
    required List<CueHistoryItem> cues,
    required String trainingMode,
    required String targetSide,
  }) async {
    final actionsList = actionCounts.entries
        .map((e) => ActionCountItem(action: e.key, count: e.value))
        .toList();

    String summary = "";
    final agent = GeminiAgent();
    if (agent.isEnabled) {
      summary = await agent.generateSummary(
        trainingMode: trainingMode,
        targetSide: targetSide,
        actionCounts: actionsList,
        cuesFired: cues,
        userSettingsName: "Fencer",
      );
    }

    final report = PracticeReport(
      id: const Uuid().v4(),
      startTimeMs: startTimeMs,
      endTimeMs: endTimeMs,
      elapsedSeconds: elapsedSeconds,
      actionCounts: actionsList,
      cueTimeline: cues,
      llmSummary: summary,
    );

    await AppDatabase.instance.savePracticeReport(report);
  }

  double get _effectivePoseFps {
    if (_poseTimestampsMs.length < 5) return 30.0;
    final spanMs = _poseTimestampsMs.last - _poseTimestampsMs.first;
    if (spanMs <= 0) return 30.0;
    return (_poseTimestampsMs.length - 1) * 1000.0 / spanMs;
  }

  Future<void> _classifyAndEvaluate() async {
    // ── FenceNet action classification ───────────────────────────────────────
    String action = 'SF'; // default footwork
    if (_fenceNetLoaded) {
      final inputFlat = _buildFenceNetInput(_classifierBuffer);
      final result = await _fenceNet.classify(inputFlat);
      action = result.action;
      if (mounted) {
        setState(() {
          _currentAction = action == 'Idle' ? 'Idle' : action;
          _actionConfidence = result.confidence;
        });
      }
    }

    // ── Heuristic evaluation ─────────────────────────────────────────────────
    final errors = _heuristics.evaluateWindow(
      action: action,
      skeletons: List.from(_skeletonBuffer),
      fps: _effectivePoseFps,
    );

    // Filter by mode
    final filtered = errors.where((e) {
      final supported = kErrorSupportedModes[e] ?? [];
      if (!supported.contains(_trainingMode)) return false;
      if (_muteErrors.contains(e)) return false;
      if (_onlySelected && _focusErrors.isNotEmpty && !_focusErrors.contains(e)) {
        return false;
      }
      return true;
    }).toList();

    // Sort: focus errors first
    filtered.sort((a, b) {
      final aFocus = _focusErrors.contains(a) ? 0 : 1;
      final bFocus = _focusErrors.contains(b) ? 0 : 1;
      return aFocus.compareTo(bFocus);
    });

    // Update metrics map
    for (final key in kErrorLabels.keys) {
      _heuristicTriggered[key] = filtered.contains(key);
    }

    // Session Tracking
    if (action != 'Idle') {
      _sessionActionCounts[action] = (_sessionActionCounts[action] ?? 0) + 1;
    }
    for (final error in filtered) {
      _sessionCues.add(CueHistoryItem(
        timestampMs: DateTime.now().millisecondsSinceEpoch,
        label: error,
      ));
    }

    if (filtered.isNotEmpty) {
      debugPrint('Triggered heuristics: $filtered');
    }

    if (mounted) {
      setState(() => _activeErrors = filtered);
    }

    // Flash & voice
    if (filtered.isNotEmpty) {
      _flashController.forward(from: 0);
      final voiceText = kErrorVoice[filtered.first] ?? filtered.first;
      _speak(voiceText);
    }
  }

  // ── Build FenceNet input tensor ─────────────────────────────────────────────

  List<double> _buildFenceNetInput(List<Skeleton> window) {
    const joints = kModelJoints;
    const T = 28;
    final input = List<double>.filled(18 * T, 0.0);

    for (int t = 0; t < T; t++) {
      final skel = window[t];
      for (int j = 0; j < 9; j++) {
        final jointName = joints[j];
        final pt = skel[jointName];
        double x = 0, y = 0;
        if (pt != null && _refNose != null && _refScale != null && _refScale! > 1e-6) {
          x = (pt.dx - _refNose!.dx) / _refScale!;
          y = (pt.dy - _refNose!.dy) / _refScale!;
        }
        input[(j * 2) * T + t] = x;
        input[(j * 2 + 1) * T + t] = y;
      }
    }
    return input;
  }

  // ── Settings helpers ────────────────────────────────────────────────────────

  void _resetLiveBuffers() {
    _yoloPoseService.resetTracking();
    _refNose = null;
    _refScale = null;
    _classifierBuffer.clear();
    _skeletonBuffer.clear();
    _poseTimestampsMs.clear();
    _activeErrors = [];
    _currentSkeleton = null;
    _currentAction = 'Idle';
    _actionConfidence = 0.0;
    _coachState = 'IDLE';
    _gatekeeper.reset();
  }

  void _rebuildHeuristics() {
    _heuristics = HeuristicsEngine(
      targetSide: _targetSide,
      trainingMode: _trainingMode,
    );
    _resetLiveBuffers();
  }

  // ──────────────────────────────────────────────────────────────────────────
  // BUILD
  // ──────────────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0A0A0F),
      appBar: AppBar(
        titleSpacing: 16,
        title: Row(
          children: [
            const Icon(Icons.sports_martial_arts, color: Color(0xFFFF6600), size: 22),
            const SizedBox(width: 8),
            const Text('AI Fencing Coach'),
            const Spacer(),
            _buildStatusPill(),
          ],
        ),
        bottom: TabBar(
          controller: _tabController,
          indicatorWeight: 3,
          isScrollable: true,
          tabs: const [
            Tab(icon: Icon(Icons.videocam, size: 20), text: 'Live'),
            Tab(icon: Icon(Icons.video_library, size: 20), text: 'Postgame'),
            Tab(icon: Icon(Icons.tune, size: 20), text: 'Settings'),
            Tab(icon: Icon(Icons.history, size: 20), text: 'History'),
            Tab(icon: Icon(Icons.analytics, size: 20), text: 'Debug'),
          ],
        ),
      ),
      body: TabBarView(
        controller: _tabController,
        physics: const NeverScrollableScrollPhysics(),
        children: [
          _buildLiveTab(),
          const PostgameScreen(),
          _buildSettingsTab(),
          const HistoryScreen(),
          _buildDebugTab(),
        ],
      ),
    );
  }

  // ── Status pill ─────────────────────────────────────────────────────────────

  Widget _buildStatusPill() {
    final isActive = _coachState == 'ACTIVE';
    return AnimatedContainer(
      duration: const Duration(milliseconds: 300),
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(
        color: isActive
            ? Colors.green.withAlpha(40)
            : Colors.orange.withAlpha(40),
        border: Border.all(
          color: isActive ? Colors.greenAccent : Colors.orange,
          width: 1,
        ),
        borderRadius: BorderRadius.circular(20),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 6,
            height: 6,
            decoration: BoxDecoration(
              color: isActive ? Colors.greenAccent : Colors.orange,
              shape: BoxShape.circle,
            ),
          ),
          const SizedBox(width: 5),
          Text(
            isActive ? 'ACTIVE' : 'IDLE',
            style: TextStyle(
              fontSize: 11,
              fontWeight: FontWeight.bold,
              color: isActive ? Colors.greenAccent : Colors.orange,
            ),
          ),
        ],
      ),
    );
  }

  // ──────────────────────────────────────────────────────────────────────────
  // TAB 1: LIVE FEED
  // ──────────────────────────────────────────────────────────────────────────

  Widget _buildLiveTab() {
    if (!_isCameraInitialized || _cameraController == null) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.camera_alt_outlined, size: 72, color: Color(0xFF333355)),
            const SizedBox(height: 20),
            const Text(
              '初始化相機...\nInitializing Camera',
              textAlign: TextAlign.center,
              style: TextStyle(color: Colors.white38, fontSize: 16),
            ),
            const SizedBox(height: 20),
            const CircularProgressIndicator(color: Color(0xFFFF6600)),
          ],
        ),
      );
    }

    return Column(
      children: [
        // Camera + Skeleton overlay
        Expanded(
          flex: 6,
          child: _buildCameraOverlay(),
        ),

        // Error cards
        Expanded(
          flex: 4,
          child: _buildFeedbackPanel(),
        ),
      ],
    );
  }

  Widget _buildCameraOverlay() {
    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      return Container();
    }

    // Always compute a portrait ratio regardless of sensor orientation.
    // On iOS 26, CameraPreview already handles rotation internally, so
    // we must NOT double-flip with 1.0/aspectRatio.
    final previewSize = _cameraController!.value.previewSize;
    final double aspectRatio;
    if (previewSize != null) {
      final w = previewSize.width;
      final h = previewSize.height;
      // Force portrait (smaller dimension / larger dimension)
      aspectRatio = w < h ? w / h : h / w;
    } else {
      aspectRatio = 3.0 / 4.0;
    }

    return Center(
      child: AspectRatio(
        aspectRatio: aspectRatio,
        child: Stack(
          fit: StackFit.expand,
          children: [
            // Camera preview
            CameraPreview(_cameraController!),

            // Skeleton overlay — keypoints from YOLOv8 are pre-normalized [0,1]
            if (_currentSkeleton != null)
              CustomPaint(
                painter: PosePainter(
                  skeleton: _currentSkeleton!,
                  imageSize: const Size(1.0, 1.0),
                  triggeredError: _activeErrors.isNotEmpty ? _activeErrors.first : null,
                  currentAction: _currentAction,
                  isFrontCamera: _cameraController?.description.lensDirection == CameraLensDirection.front,
                ),
              ),

            // Warning flash overlay
            AnimatedBuilder(
              animation: _flashAnimation,
              builder: (context, _) => CustomPaint(
                painter: WarningFlashPainter(opacity: _flashAnimation.value),
              ),
            ),

            // Action label pill (top center)
            Positioned(
              top: 12,
              left: 0,
              right: 0,
              child: Center(child: _buildActionPill()),
            ),

            // FenceNet status (top left)
            Positioned(
              top: 12,
              left: 12,
              child: _buildFenceNetBadge(),
            ),

            // Camera toggle button (top right)
            Positioned(
              top: 12,
              right: 12,
              child: _buildCameraToggleButton(),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildActionPill() {
    final Color c;
    final String label;
    if (_currentAction == 'Idle') {
      c = Colors.white30;
      label = '⏸  Idle';
    } else if ({'R', 'IS', 'WW', 'JS'}.contains(_currentAction)) {
      c = const Color(0xFFFF4D00);
      label = '⚔️  $_currentAction  (${(_actionConfidence * 100).toStringAsFixed(0)}%)';
    } else {
      c = const Color(0xFF00D4FF);
      label = '🏃  $_currentAction  (${(_actionConfidence * 100).toStringAsFixed(0)}%)';
    }
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 6),
      decoration: BoxDecoration(
        color: Colors.black.withAlpha(170),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: c.withAlpha(180), width: 1.5),
      ),
      child: Text(
        label,
        style: TextStyle(
          color: c,
          fontWeight: FontWeight.bold,
          fontSize: 13,
          letterSpacing: 0.5,
        ),
      ),
    );
  }

  Widget _buildFenceNetBadge() {
    final bool isAllLoaded = _fenceNetLoaded && _yoloPoseLoaded;
    final String label = isAllLoaded
        ? '🧠 CoreML'
        : (!_yoloPoseLoaded ? '⚠️ No Pose Model' : '⚠️ No Action Model');
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
        color: Colors.black.withAlpha(140),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(
          color: isAllLoaded ? Colors.greenAccent.withAlpha(120) : Colors.orange.withAlpha(120),
        ),
      ),
      child: Text(
        label,
        style: TextStyle(
          color: isAllLoaded ? Colors.greenAccent : Colors.orange,
          fontSize: 10,
          fontWeight: FontWeight.w600,
        ),
      ),
    );
  }

  Widget _buildCameraToggleButton() {
    return Container(
      decoration: BoxDecoration(
        color: Colors.black.withAlpha(140),
        borderRadius: BorderRadius.circular(20),
      ),
      child: IconButton(
        icon: const Icon(
          Icons.flip_camera_ios,
          color: Colors.white,
          size: 20,
        ),
        onPressed: () {
          final nextDir = _cameraLensDirection == CameraLensDirection.back
              ? CameraLensDirection.front
              : CameraLensDirection.back;
          _switchCamera(nextDir);
        },
      ),
    );
  }

  Widget _buildFeedbackPanel() {
    Widget content;
    if (_activeErrors.isEmpty) {
      content = Center(
        child: SingleChildScrollView(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Container(
                width: 56,
                height: 56,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: Colors.green.withAlpha(30),
                  border: Border.all(color: Colors.greenAccent.withAlpha(100)),
                ),
                child: const Icon(Icons.check, color: Colors.greenAccent, size: 30),
              ),
              const SizedBox(height: 12),
              const Text(
                '動作正確 ✓\nGood Technique',
                textAlign: TextAlign.center,
                style: TextStyle(
                  color: Colors.greenAccent,
                  fontSize: 16,
                  fontWeight: FontWeight.w600,
                ),
              ),
              const SizedBox(height: 4),
              Text(
                _coachState == 'IDLE' ? '等待偵測到擊劍手...' : '繼續保持！',
                style: const TextStyle(color: Colors.white70, fontSize: 12),
                textAlign: TextAlign.center,
              ),
            ],
          ),
        ),
      );
    } else {
      content = ListView.builder(
        padding: const EdgeInsets.all(12),
        itemCount: _activeErrors.length,
        itemBuilder: (ctx, idx) {
          final key = _activeErrors[idx];
          final label = kErrorLabels[key] ?? key;
          final isPrimary = idx == 0;
          return AnimatedContainer(
            duration: const Duration(milliseconds: 300),
            margin: const EdgeInsets.only(bottom: 8),
            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
            decoration: BoxDecoration(
              gradient: LinearGradient(
                colors: isPrimary
                    ? [
                        const Color(0xFF5A0000),
                        const Color(0xFF2A0000),
                      ]
                    : [
                        const Color(0xFF3A2000),
                        const Color(0xFF1A1000),
                      ],
              ),
              borderRadius: BorderRadius.circular(10),
              border: Border.all(
                color: isPrimary
                    ? Colors.redAccent.withAlpha(200)
                    : Colors.orangeAccent.withAlpha(100),
              ),
            ),
            child: Row(
              children: [
                Icon(
                  isPrimary ? Icons.error_rounded : Icons.warning_rounded,
                  color: isPrimary ? Colors.redAccent : Colors.orangeAccent,
                  size: 20,
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: Text(
                    label,
                    style: TextStyle(
                      color: isPrimary ? Colors.white : Colors.white70,
                      fontWeight: isPrimary ? FontWeight.bold : FontWeight.normal,
                      fontSize: isPrimary ? 15 : 13,
                    ),
                  ),
                ),
                if (_focusErrors.contains(key))
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                    decoration: BoxDecoration(
                      color: const Color(0xFFFF6600).withAlpha(60),
                      borderRadius: BorderRadius.circular(4),
                    ),
                    child: const Text(
                      'FOCUS',
                      style: TextStyle(
                        color: Color(0xFFFF6600),
                        fontSize: 9,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
              ],
            ),
          );
        },
      );
    }

    return Container(
      color: const Color(0xFF0D0D1A),
      child: Column(
        children: [
          Expanded(child: content),
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(8),
            color: Colors.black.withAlpha(100),
            child: Text(
              '[Debug: ${_gatekeeper.state}]\n'
              'Knee Angle: ${_gatekeeper.lastReasons['knee_angle']?.toStringAsFixed(1) ?? 'N/A'} (Needs < 176)\n'
              'Turned Back: ${_gatekeeper.lastReasons['turned_back']}\n'
              'Moving: ${_gatekeeper.lastReasons['moving']}\n'
              'Step Ratio: ${_heuristics.lastStepRatio?.toStringAsFixed(2) ?? 'N/A'} | '
              'Step Width: ${_heuristics.lastStepWidth?.toStringAsFixed(3) ?? 'N/A'} | '
              'Pose FPS: ${_effectivePoseFps.toStringAsFixed(1)}',
              style: const TextStyle(color: Colors.white54, fontSize: 10),
              textAlign: TextAlign.center,
            ),
          ),
        ],
      ),
    );
  }

  // ──────────────────────────────────────────────────────────────────────────
  // TAB 2: SETTINGS
  // ──────────────────────────────────────────────────────────────────────────

  Widget _buildSettingsTab() {
    final allowedErrors = kErrorLabels.keys.where((key) {
      final modes = kErrorSupportedModes[key] ?? [];
      return modes.contains(_trainingMode);
    }).toList();

    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _sectionHeader('⚙️ 訓練設定 Training Settings'),
          const SizedBox(height: 12),

          // Target Side
          _buildDropdownField<String>(
            label: '擊劍手方向 Target Side',
            value: _targetSide,
            items: const [
              DropdownMenuItem(value: 'left', child: Text('左撇子擊劍手 (Left Fencer)')),
              DropdownMenuItem(value: 'right', child: Text('右撇子擊劍手 (Right Fencer)')),
            ],
            onChanged: (val) {
              if (val != null && val != _targetSide) {
                setState(() {
                  _targetSide = val;
                  _rebuildHeuristics();
                });
              }
            },
          ),
          const SizedBox(height: 12),

          // Training Mode
          _buildDropdownField<String>(
            label: '訓練模式 Training Mode',
            value: _trainingMode,
            items: const [
              DropdownMenuItem(value: 'Footwork', child: Text('步法訓練 Footwork')),
              DropdownMenuItem(value: 'Target Practice', child: Text('刺靶練習 Target Practice')),
              DropdownMenuItem(value: 'Free Bouting', child: Text('自由對練 Free Bouting')),
            ],
            onChanged: (val) {
              if (val != null && val != _trainingMode) {
                setState(() {
                  _trainingMode = val;
                  _focusErrors = _focusErrors
                      .where((e) => kErrorSupportedModes[e]!.contains(val))
                      .toList();
                  _muteErrors = _muteErrors
                      .where((e) => kErrorSupportedModes[e]!.contains(val))
                      .toList();
                  _rebuildHeuristics();
                });
              }
            },
          ),
          const SizedBox(height: 16),

          // Voice switch
          _buildCard(
            child: SwitchListTile(
              title: const Text('語音提示 Voice Coaching'),
              subtitle: const Text('即時語音播報錯誤 / Real-time voice alerts'),
              value: _voiceEnabled,
              activeThumbColor: const Color(0xFFFF6600),
              contentPadding: EdgeInsets.zero,
              onChanged: (val) => setState(() => _voiceEnabled = val),
            ),
          ),
          const SizedBox(height: 16),

          // Focus Errors
          _sectionHeader('🎯 重點錯誤 Focus Errors'),
          const SizedBox(height: 4),
          const Text(
            '選取後這些錯誤會優先提示',
            style: TextStyle(color: Colors.white38, fontSize: 12),
          ),
          const SizedBox(height: 8),
          _buildCard(
            child: Column(
              children: allowedErrors.map((key) {
                final label = kErrorLabels[key] ?? key;
                return CheckboxListTile(
                  title: Text(label, style: const TextStyle(fontSize: 13)),
                  value: _focusErrors.contains(key),
                  activeColor: const Color(0xFFFF6600),
                  contentPadding: EdgeInsets.zero,
                  dense: true,
                  onChanged: (val) {
                    setState(() {
                      if (val == true) {
                        _focusErrors.add(key);
                        _muteErrors.remove(key);
                      } else {
                        _focusErrors.remove(key);
                      }
                    });
                  },
                );
              }).toList(),
            ),
          ),
          const SizedBox(height: 16),

          // Mute Errors
          _sectionHeader('🔇 靜音錯誤 Mute Errors'),
          const SizedBox(height: 4),
          const Text(
            '選取後這些錯誤不會被提示',
            style: TextStyle(color: Colors.white38, fontSize: 12),
          ),
          const SizedBox(height: 8),
          _buildCard(
            child: Column(
              children: allowedErrors.map((key) {
                final label = kErrorLabels[key] ?? key;
                return CheckboxListTile(
                  title: Text(label, style: const TextStyle(fontSize: 13)),
                  value: _muteErrors.contains(key),
                  activeColor: Colors.blueGrey,
                  contentPadding: EdgeInsets.zero,
                  dense: true,
                  onChanged: (val) {
                    setState(() {
                      if (val == true) {
                        _muteErrors.add(key);
                        _focusErrors.remove(key);
                      } else {
                        _muteErrors.remove(key);
                      }
                    });
                  },
                );
              }).toList(),
            ),
          ),
          const SizedBox(height: 12),

          // Only selected
          _buildCard(
            child: CheckboxListTile(
              title: const Text('只顯示重點錯誤 Only show focused errors'),
              value: _onlySelected,
              activeColor: const Color(0xFFFF6600),
              contentPadding: EdgeInsets.zero,
              onChanged: (val) {
                if (val != null) setState(() => _onlySelected = val);
              },
            ),
          ),
          const SizedBox(height: 32),
        ],
      ),
    );
  }

  // ──────────────────────────────────────────────────────────────────────────
  // TAB 3: DEBUG METRICS
  // ──────────────────────────────────────────────────────────────────────────

  Widget _buildDebugTab() {
    return Column(
      children: [
        Container(
          width: double.infinity,
          padding: const EdgeInsets.all(14),
          color: const Color(0xFF141420),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  const Icon(Icons.memory, size: 16, color: Color(0xFFFF6600)),
                  const SizedBox(width: 6),
                  Text(
                    'Frame: $_frameCount  |  Buffer: ${_skeletonBuffer.length}/60',
                    style: const TextStyle(
                      color: Color(0xFFFF6600),
                      fontWeight: FontWeight.bold,
                      fontSize: 13,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 4),
              Text(
                'Action: $_currentAction  |  Mode: $_trainingMode  |  Side: $_targetSide',
                style: const TextStyle(color: Colors.white54, fontSize: 11),
              ),
              const SizedBox(height: 2),
              Text(
                'FenceNet: ${_fenceNetLoaded ? "CoreML ✓" : "Not loaded"}',
                style: TextStyle(
                  color: _fenceNetLoaded ? Colors.greenAccent : Colors.orange,
                  fontSize: 11,
                ),
              ),
            ],
          ),
        ),
        Expanded(
          child: ListView(
            padding: const EdgeInsets.all(12),
            children: kErrorLabels.keys.map((key) {
              final triggered = _heuristicTriggered[key] ?? false;
              final label = kErrorLabels[key] ?? key;
              return Container(
                margin: const EdgeInsets.only(bottom: 8),
                decoration: BoxDecoration(
                  color: triggered
                      ? const Color(0xFF3A0000)
                      : const Color(0xFF141420),
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(
                    color: triggered
                        ? Colors.redAccent.withAlpha(180)
                        : Colors.white12,
                  ),
                ),
                child: ListTile(
                  dense: true,
                  leading: Container(
                    padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                    decoration: BoxDecoration(
                      color: triggered
                          ? Colors.red
                          : Colors.green.shade800,
                      borderRadius: BorderRadius.circular(4),
                    ),
                    child: Text(
                      triggered ? 'TRIGGER' : 'OK',
                      style: const TextStyle(
                        fontSize: 9,
                        fontWeight: FontWeight.bold,
                        color: Colors.white,
                      ),
                    ),
                  ),
                  title: Text(
                    label,
                    style: TextStyle(
                      fontSize: 12,
                      color: triggered ? Colors.white : Colors.white60,
                    ),
                  ),
                  subtitle: Text(
                    key,
                    style: const TextStyle(
                      fontFamily: 'monospace',
                      fontSize: 10,
                      color: Colors.white30,
                    ),
                  ),
                ),
              );
            }).toList(),
          ),
        ),
      ],
    );
  }

  // ──────────────────────────────────────────────────────────────────────────
  // UI helpers
  // ──────────────────────────────────────────────────────────────────────────

  Widget _sectionHeader(String title) {
    return Text(
      title,
      style: const TextStyle(
        fontSize: 16,
        fontWeight: FontWeight.bold,
        color: Colors.white,
        letterSpacing: 0.3,
      ),
    );
  }

  Widget _buildCard({required Widget child}) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: const Color(0xFF141420),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.white10),
      ),
      child: child,
    );
  }

  Widget _buildDropdownField<T>({
    required String label,
    required T value,
    required List<DropdownMenuItem<T>> items,
    required ValueChanged<T?> onChanged,
  }) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label, style: const TextStyle(color: Colors.white54, fontSize: 12)),
        const SizedBox(height: 6),
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 12),
          decoration: BoxDecoration(
            color: const Color(0xFF141420),
            borderRadius: BorderRadius.circular(8),
            border: Border.all(color: Colors.white12),
          ),
          child: DropdownButtonHideUnderline(
            child: DropdownButton<T>(
              value: value,
              isExpanded: true,
              dropdownColor: const Color(0xFF1E1E30),
              items: items,
              onChanged: onChanged,
            ),
          ),
        ),
      ],
    );
  }
}

// ---------------------------------------------------------------------------
// Model joints constant (same as fencenet_classifier.dart)
// ---------------------------------------------------------------------------

const List<String> kModelJoints = [
  'front_wrist',
  'front_elbow',
  'front_shoulder',
  'left_hip',
  'right_hip',
  'left_knee',
  'right_knee',
  'left_ankle',
  'right_ankle',
];
