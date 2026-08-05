/// macOS video threshold-tuner for the heuristics engine.
///
/// Run with:  tool/run_tuner.sh   (or:
///   flutter run -d macos -t lib/tuner_main.dart \
///     --dart-define=TUNE_VIDEO_DIR=/abs/path/to/docs \
///     --dart-define=TUNE_MODEL_PATH=/abs/path/to/yolov8n_pose.mlpackage )
///
/// Workflow: pick a cue video (named after its error key) → poses are
/// extracted once via CoreML and cached → the video plays with a skeleton
/// overlay while the metric timeline below shows the cue's metric across the
/// whole clip, evaluated by the SAME HeuristicsEngine + window cadence the
/// app uses. Drag the threshold (slider or the dashed line itself) and
/// trigger regions update instantly. "複製全部參數" yields the Dart snippet to
/// paste back into HeuristicsConfig.

library;

import 'dart:io';
import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:flutter/scheduler.dart';
import 'package:flutter/services.dart';
import 'package:video_player/video_player.dart';

import 'heuristics/heuristics_engine.dart';
import 'pose/pose_painter.dart';
import 'pose/pose_service.dart';
import 'tuning/replay_evaluator.dart';
import 'tuning/tuner_cues.dart';
import 'tuning/tuner_settings.dart';
import 'tuning/tuning_config_store.dart';
import 'tuning/tuning_specs.dart';
import 'tuning/video_pose_extractor.dart';

// Chart colors (dark surface; roles are shape-coded too: solid data line,
// dashed threshold line, shaded trigger regions with tick strip).
const kMetricLineColor = Color(0xFF4FC3F7);
const kThresholdColor = Color(0xFFFFB74D);
const kTriggerColor = Color(0xFFEF5350);

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const TunerApp());
}

class TunerApp extends StatelessWidget {
  const TunerApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Heuristics 影片調參',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        brightness: Brightness.dark,
        colorSchemeSeed: kMetricLineColor,
        useMaterial3: true,
      ),
      home: const TunerScreen(),
    );
  }
}

class TunerScreen extends StatefulWidget {
  const TunerScreen({super.key});

  @override
  State<TunerScreen> createState() => _TunerScreenState();
}

class _TunerScreenState extends State<TunerScreen>
    with SingleTickerProviderStateMixin {
  TunerSettings? _settings;
  List<String> _videoPaths = [];

  String? _videoPath;
  VideoPlayerController? _player;
  late final Ticker _ticker;

  VideoPoseData? _poseData;
  List<PoseFrame> _poseFrames = [];
  ReplayResult? _replay;

  HeuristicsConfig _config = const HeuristicsConfig();
  String _targetSide = 'left';
  String _cueKey = kTuningSpecs.first.errorKey;

  bool _busy = false;
  String _busyLabel = '';
  double _busyProgress = 0;
  String? _errorMessage;

  final FocusNode _focusNode = FocusNode();

  @override
  void initState() {
    super.initState();
    _ticker = createTicker((_) {
      if (_player?.value.isPlaying == true && mounted) setState(() {});
    })
      ..start();
    _init();
  }

  Future<void> _init() async {
    final config = await TuningConfigStore.load();
    final settings = await TunerSettings.load();
    setState(() {
      _config = config;
      _settings = settings;
    });
    _scanVideos();
    // Headless batch mode: extract (cache-aware) every video in the folder,
    // then quit. Used after re-recording clips:
    //   flutter build macos --debug -t lib/tuner_main.dart \
    //     --dart-define=TUNE_EXTRACT_ALL=true ... && open <app>
    if (const bool.fromEnvironment('TUNE_EXTRACT_ALL')) {
      for (final path in _videoPaths) {
        setState(() {
          _busy = true;
          _busyLabel = '批次抽取 ${path.split('/').last}…';
          _busyProgress = 0;
        });
        try {
          await VideoPoseExtractor.extract(
            videoPath: path,
            modelPath: _settings!.modelPath,
            onProgress: (done, total) {
              if (mounted) {
                setState(() => _busyProgress = total == 0 ? 0 : done / total);
              }
            },
          );
          debugPrint('TUNE_EXTRACT_ALL done: $path');
        } catch (e) {
          debugPrint('TUNE_EXTRACT_ALL failed: $path → $e');
        }
      }
      exit(0);
    }
    if (_videoPaths.isNotEmpty) {
      await _selectVideo(_videoPaths.first);
    }
  }

  @override
  void dispose() {
    _ticker.dispose();
    _player?.dispose();
    _focusNode.dispose();
    super.dispose();
  }

  // ── Video list ────────────────────────────────────────────────────────────

  void _scanVideos() {
    final dirPath = _settings?.videoDir ?? '';
    final dir = Directory(dirPath);
    if (dirPath.isEmpty || !dir.existsSync()) {
      setState(() => _videoPaths = []);
      return;
    }
    final paths = dir
        .listSync()
        .whereType<File>()
        .map((f) => f.path)
        .where((p) {
          final lower = p.toLowerCase();
          return lower.endsWith('.mov') || lower.endsWith('.mp4');
        })
        .toList()
      ..sort();
    setState(() => _videoPaths = paths);
  }

  // ── Selection / extraction / evaluation ───────────────────────────────────

  Future<void> _selectVideo(String path, {bool forceReextract = false}) async {
    if (_busy) return;
    final modelPath = _settings?.modelPath ?? '';
    setState(() {
      _videoPath = path;
      _errorMessage = null;
      _busy = true;
      _busyLabel = '抽取骨架中…';
      _busyProgress = 0;
      _poseData = null;
      _poseFrames = [];
      _replay = null;
    });

    // (Re)build the player.
    await _player?.dispose();
    _player = null;
    final player = VideoPlayerController.file(File(path));
    try {
      await player.initialize();
      setState(() => _player = player);
    } catch (e) {
      setState(() {
        _busy = false;
        _errorMessage = '影片開啟失敗: $e';
      });
      return;
    }

    // Auto-select the cue from the filename.
    final matched = cuesForVideoName(path.split('/').last);
    if (matched.isNotEmpty && !matched.contains(_cueKey)) {
      _cueKey = matched.first;
    }

    try {
      final data = await VideoPoseExtractor.extract(
        videoPath: path,
        modelPath: modelPath,
        forceReextract: forceReextract,
        onProgress: (done, total) {
          if (mounted) {
            setState(() => _busyProgress = total == 0 ? 0 : done / total);
          }
        },
      );
      if (!mounted || _videoPath != path) return;
      _poseData = data;
      _retrack();
      setState(() => _busy = false);
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _busy = false;
        _errorMessage = '骨架抽取失敗: $e\n(model: $modelPath)';
      });
    }
  }

  void _retrack() {
    final data = _poseData;
    if (data == null) return;
    _poseFrames = trackSkeletons(data, _targetSide);
    _reevaluate();
  }

  void _reevaluate() {
    final data = _poseData;
    if (data == null) return;
    _replay = evaluateReplay(
      data: data,
      poseFrames: _poseFrames,
      config: _config,
      targetSide: _targetSide,
      action: kCueAction[_cueKey] ?? 'SF',
    );
    setState(() {});
  }

  // ── Config editing ────────────────────────────────────────────────────────

  void _setParam(String paramName, double value, {bool save = false}) {
    final map = _config.toMap();
    map[paramName] = value;
    _config = HeuristicsConfig.fromMap(map);
    _reevaluate();
    if (save) TuningConfigStore.save(_config);
  }

  void _resetCueParams() {
    const defaults = HeuristicsConfig();
    final defaultMap = defaults.toMap();
    final map = _config.toMap();
    final spec = specForError(_cueKey);
    map[spec.paramName] = defaultMap[spec.paramName]!;
    for (final aux in kCueAuxParams[_cueKey] ?? const <AuxParamSpec>[]) {
      map[aux.paramName] = defaultMap[aux.paramName]!;
    }
    _config = HeuristicsConfig.fromMap(map);
    _reevaluate();
    TuningConfigStore.save(_config);
  }

  // ── Playback helpers ──────────────────────────────────────────────────────

  int get _playheadMs => _player?.value.position.inMilliseconds ?? 0;
  int get _durationMs {
    final d = _player?.value.duration.inMilliseconds ?? 0;
    if (d > 0) return d;
    return _poseData?.frames.isNotEmpty == true ? _poseData!.frames.last.tMs : 1;
  }

  void _togglePlay() {
    final p = _player;
    if (p == null) return;
    setState(() {
      if (p.value.isPlaying) {
        p.pause();
      } else {
        if (p.value.position >= p.value.duration) p.seekTo(Duration.zero);
        p.play();
      }
    });
  }

  void _seekMs(int ms) {
    _player?.seekTo(Duration(milliseconds: ms.clamp(0, _durationMs).toInt()));
    setState(() {});
  }

  void _stepFrames(int frames) {
    _player?.pause();
    _seekMs(_playheadMs + (frames * 1000 / 30).round());
  }

  KeyEventResult _onKey(FocusNode node, KeyEvent event) {
    if (event is! KeyDownEvent && event is! KeyRepeatEvent) {
      return KeyEventResult.ignored;
    }
    final shift = HardwareKeyboard.instance.isShiftPressed;
    switch (event.logicalKey) {
      case LogicalKeyboardKey.space:
        _togglePlay();
        return KeyEventResult.handled;
      case LogicalKeyboardKey.arrowLeft:
        shift ? _seekMs(_playheadMs - 1000) : _stepFrames(-1);
        return KeyEventResult.handled;
      case LogicalKeyboardKey.arrowRight:
        shift ? _seekMs(_playheadMs + 1000) : _stepFrames(1);
        return KeyEventResult.handled;
    }
    return KeyEventResult.ignored;
  }

  // ── Derived state ─────────────────────────────────────────────────────────

  /// Skeleton nearest the playhead (≤150 ms away) for the overlay.
  FencingSkeleton? get _overlaySkeleton {
    if (_poseFrames.isEmpty) return null;
    final t = _playheadMs;
    int lo = 0, hi = _poseFrames.length - 1;
    while (lo < hi) {
      final mid = (lo + hi + 1) ~/ 2;
      if (_poseFrames[mid].tMs <= t) {
        lo = mid;
      } else {
        hi = mid - 1;
      }
    }
    PoseFrame best = _poseFrames[lo];
    if (lo + 1 < _poseFrames.length &&
        (_poseFrames[lo + 1].tMs - t).abs() < (best.tMs - t).abs()) {
      best = _poseFrames[lo + 1];
    }
    if ((best.tMs - t).abs() > 150) return null;
    return FencingSkeleton(
      joints: best.joints,
      nose: best.joints['nose'],
      scale: null,
      imageWidth: 1,
      imageHeight: 1,
    );
  }

  /// Evaluation window whose end is nearest the playhead (for the readout).
  WindowEval? get _currentWindow {
    final windows = _replay?.windows;
    if (windows == null || windows.isEmpty) return null;
    WindowEval best = windows.first;
    for (final w in windows) {
      if ((w.endMs - _playheadMs).abs() < (best.endMs - _playheadMs).abs()) {
        best = w;
      }
    }
    return best;
  }

  // ── UI ────────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    final spec = specForError(_cueKey);
    return Scaffold(
      body: Focus(
        focusNode: _focusNode,
        autofocus: true,
        onKeyEvent: _onKey,
        child: Column(
          children: [
            _buildTopBar(),
            if (_errorMessage != null)
              MaterialBanner(
                content: Text(_errorMessage!),
                leading: const Icon(Icons.error_outline, color: kTriggerColor),
                actions: [
                  TextButton(
                    onPressed: () => setState(() => _errorMessage = null),
                    child: const Text('關閉'),
                  ),
                ],
              ),
            Expanded(
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  Expanded(child: _buildVideoPane()),
                  SizedBox(width: 400, child: _buildControlPane(spec)),
                ],
              ),
            ),
            SizedBox(height: 250, child: _buildChart(spec)),
          ],
        ),
      ),
    );
  }

  Widget _buildTopBar() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(12, 8, 12, 4),
      child: Row(
        children: [
          const Text('Heuristics 影片調參',
              style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
          const SizedBox(width: 16),
          Expanded(
            child: SingleChildScrollView(
              scrollDirection: Axis.horizontal,
              child: Row(
                children: [
                  for (final path in _videoPaths)
                    Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 3),
                      child: ChoiceChip(
                        label: Text(
                          path.split('/').last.replaceAll(
                              RegExp(r'\.(mov|mp4)$', caseSensitive: false), ''),
                          style: const TextStyle(fontSize: 12),
                        ),
                        selected: _videoPath == path,
                        onSelected: (_) => _selectVideo(path),
                      ),
                    ),
                ],
              ),
            ),
          ),
          const SizedBox(width: 8),
          SegmentedButton<String>(
            segments: const [
              ButtonSegment(value: 'left', label: Text('左側', style: TextStyle(fontSize: 12))),
              ButtonSegment(value: 'right', label: Text('右側', style: TextStyle(fontSize: 12))),
            ],
            selected: {_targetSide},
            onSelectionChanged: (sel) {
              _targetSide = sel.first;
              _retrack();
            },
          ),
          IconButton(
            tooltip: '重新抽取骨架（忽略快取）',
            icon: const Icon(Icons.restart_alt),
            onPressed: _videoPath == null || _busy
                ? null
                : () => _selectVideo(_videoPath!, forceReextract: true),
          ),
          IconButton(
            tooltip: '路徑設定',
            icon: const Icon(Icons.settings_outlined),
            onPressed: _openSettings,
          ),
        ],
      ),
    );
  }

  Widget _buildVideoPane() {
    final player = _player;
    return Container(
      margin: const EdgeInsets.all(8),
      decoration: BoxDecoration(
        color: Colors.black,
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(
        children: [
          Expanded(
            child: player == null || !player.value.isInitialized
                ? Center(
                    child: _busy
                        ? const CircularProgressIndicator()
                        : const Text('選擇上方影片開始'),
                  )
                : Center(
                    child: AspectRatio(
                      aspectRatio: player.value.aspectRatio,
                      child: Stack(
                        fit: StackFit.expand,
                        children: [
                          VideoPlayer(player),
                          if (_overlaySkeleton != null)
                            IgnorePointer(
                              child: CustomPaint(
                                painter: PosePainter(
                                  skeleton: _overlaySkeleton!,
                                  imageSize: const Size(1, 1),
                                  currentAction: '',
                                ),
                              ),
                            ),
                        ],
                      ),
                    ),
                  ),
          ),
          if (_busy)
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
              child: Row(children: [
                Expanded(child: LinearProgressIndicator(value: _busyProgress)),
                const SizedBox(width: 8),
                Text('$_busyLabel ${(_busyProgress * 100).round()}%',
                    style: const TextStyle(fontSize: 12)),
              ]),
            ),
          _buildTransport(),
        ],
      ),
    );
  }

  Widget _buildTransport() {
    final playing = _player?.value.isPlaying == true;
    String fmt(int ms) {
      final s = ms / 1000;
      return '${s.toStringAsFixed(2)}s';
    }

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
      child: Row(
        children: [
          IconButton(
            tooltip: '上一幀 (←)',
            icon: const Icon(Icons.skip_previous),
            onPressed: () => _stepFrames(-1),
          ),
          IconButton(
            tooltip: '播放/暫停 (space)',
            icon: Icon(playing ? Icons.pause : Icons.play_arrow),
            onPressed: _togglePlay,
          ),
          IconButton(
            tooltip: '下一幀 (→)',
            icon: const Icon(Icons.skip_next),
            onPressed: () => _stepFrames(1),
          ),
          Expanded(
            child: Slider(
              value: _playheadMs.clamp(0, _durationMs).toDouble(),
              max: _durationMs.toDouble(),
              onChanged: (v) {
                _player?.pause();
                _seekMs(v.round());
              },
            ),
          ),
          Text('${fmt(_playheadMs)} / ${fmt(_durationMs)}',
              style: const TextStyle(fontSize: 12, fontFeatures: [
                FontFeature.tabularFigures(),
              ])),
        ],
      ),
    );
  }

  Widget _buildControlPane(TuningSpec spec) {
    final matched =
        _videoPath == null ? const <String>[] : cuesForVideoName(_videoPath!.split('/').last);
    final auxSpecs = kCueAuxParams[_cueKey] ?? const <AuxParamSpec>[];
    final window = _currentWindow;
    final metric = window?.metrics[spec.metricKey];
    final threshold = spec.thresholdOf(_config);
    final triggeredNow = window?.triggered.contains(_cueKey) == true;
    final triggerCount =
        _replay?.windows.where((w) => w.triggered.contains(_cueKey)).length ?? 0;

    return ListView(
      padding: const EdgeInsets.fromLTRB(4, 8, 12, 8),
      children: [
        // ── Cue selector ────────────────────────────────────────────────
        Wrap(
          spacing: 6,
          runSpacing: 2,
          children: [
            for (final key in [
              ...matched,
              ...kTuningSpecs.map((s) => s.errorKey).where((k) => !matched.contains(k)),
            ])
              ChoiceChip(
                label: Text(kCueLabels[key]?.split(' (').first ?? key,
                    style: const TextStyle(fontSize: 12)),
                selected: _cueKey == key,
                avatar: matched.contains(key)
                    ? const Icon(Icons.videocam, size: 14)
                    : null,
                onSelected: (_) {
                  _cueKey = key;
                  _reevaluate();
                },
              ),
          ],
        ),
        const SizedBox(height: 8),
        Text(
          '評估情境: action=${kCueAction[_cueKey]}   '
          '${spec.direction == TriggerDirection.above ? "指標 > 閾值 → 觸發" : "指標 < 閾值 → 觸發"}',
          style: TextStyle(fontSize: 12, color: Colors.grey.shade400),
        ),
        const SizedBox(height: 8),

        // ── Live readout ────────────────────────────────────────────────
        Card(
          color: triggeredNow
              ? kTriggerColor.withAlpha(50)
              : Theme.of(context).colorScheme.surfaceContainerHighest,
          child: Padding(
            padding: const EdgeInsets.all(10),
            child: Row(
              children: [
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(spec.metricKey,
                          style:
                              TextStyle(fontSize: 11, color: Colors.grey.shade400)),
                      Text(
                        metric == null
                            ? '—'
                            : '${metric.toStringAsFixed(spec.decimals)}${spec.unit}',
                        style: const TextStyle(
                            fontSize: 26, fontWeight: FontWeight.bold),
                      ),
                    ],
                  ),
                ),
                Column(
                  crossAxisAlignment: CrossAxisAlignment.end,
                  children: [
                    Text(triggeredNow ? '觸發' : '未觸發',
                        style: TextStyle(
                          fontSize: 14,
                          fontWeight: FontWeight.bold,
                          color: triggeredNow ? kTriggerColor : Colors.grey,
                        )),
                    Text('全片觸發窗數: $triggerCount',
                        style:
                            TextStyle(fontSize: 11, color: Colors.grey.shade400)),
                    if (window != null)
                      Text('fps≈${window.fps.toStringAsFixed(1)}',
                          style: TextStyle(
                              fontSize: 11, color: Colors.grey.shade500)),
                  ],
                ),
              ],
            ),
          ),
        ),
        const SizedBox(height: 8),

        // ── Primary threshold slider ────────────────────────────────────
        _paramSlider(
          label: spec.paramName,
          value: threshold,
          min: spec.min,
          max: spec.max,
          decimals: spec.decimals,
          unit: spec.unit,
          hint: spec.hint,
          accent: kThresholdColor,
          onChanged: (v) => _setParam(spec.paramName, v),
          onChangeEnd: (v) => _setParam(spec.paramName, v, save: true),
        ),

        // ── Aux sliders ─────────────────────────────────────────────────
        for (final aux in auxSpecs)
          _paramSlider(
            label: aux.paramName,
            value: _config.toMap()[aux.paramName]!,
            min: aux.min,
            max: aux.max,
            decimals: aux.decimals,
            unit: '',
            hint: aux.hint,
            accent: Colors.grey.shade500,
            onChanged: (v) => _setParam(aux.paramName, v),
            onChangeEnd: (v) => _setParam(aux.paramName, v, save: true),
          ),

        const SizedBox(height: 12),
        Row(
          children: [
            Expanded(
              child: FilledButton.icon(
                icon: const Icon(Icons.copy, size: 16),
                label: const Text('複製全部參數'),
                onPressed: () {
                  Clipboard.setData(ClipboardData(
                      text: TuningConfigStore.asDartSnippet(_config)));
                  ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
                    content: Text('已複製 Dart 參數片段，貼回 HeuristicsConfig 預設值'),
                    duration: Duration(seconds: 2),
                  ));
                },
              ),
            ),
            const SizedBox(width: 8),
            OutlinedButton(
              onPressed: _resetCueParams,
              child: const Text('重設此 cue'),
            ),
          ],
        ),
      ],
    );
  }

  Widget _paramSlider({
    required String label,
    required double value,
    required double min,
    required double max,
    required int decimals,
    required String unit,
    required String hint,
    required Color accent,
    required ValueChanged<double> onChanged,
    required ValueChanged<double> onChangeEnd,
  }) {
    final step = math.pow(10.0, -decimals).toDouble();
    return Padding(
      padding: const EdgeInsets.only(bottom: 4),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Expanded(
                child: Text(label,
                    style: TextStyle(
                        fontSize: 12,
                        fontWeight: FontWeight.w600,
                        color: accent)),
              ),
              IconButton(
                visualDensity: VisualDensity.compact,
                iconSize: 14,
                icon: const Icon(Icons.remove),
                onPressed: () =>
                    onChangeEnd((value - step).clamp(min, max)),
              ),
              Text('${value.toStringAsFixed(decimals)}$unit',
                  style: const TextStyle(
                      fontSize: 13,
                      fontWeight: FontWeight.bold,
                      fontFeatures: [FontFeature.tabularFigures()])),
              IconButton(
                visualDensity: VisualDensity.compact,
                iconSize: 14,
                icon: const Icon(Icons.add),
                onPressed: () =>
                    onChangeEnd((value + step).clamp(min, max)),
              ),
            ],
          ),
          SliderTheme(
            data: SliderTheme.of(context).copyWith(
              trackHeight: 2,
              thumbShape: const RoundSliderThumbShape(enabledThumbRadius: 7),
            ),
            child: Slider(
              value: value.clamp(min, max),
              min: min,
              max: max,
              activeColor: accent,
              onChanged: onChanged,
              onChangeEnd: onChangeEnd,
            ),
          ),
          Text(hint, style: TextStyle(fontSize: 11, color: Colors.grey.shade500)),
        ],
      ),
    );
  }

  Widget _buildChart(TuningSpec spec) {
    final windows = _replay?.windows ?? const <WindowEval>[];
    return Padding(
      padding: const EdgeInsets.fromLTRB(12, 0, 12, 10),
      child: MetricTimelineChart(
        windows: windows,
        spec: spec,
        cueKey: _cueKey,
        threshold: spec.thresholdOf(_config),
        durationMs: _durationMs,
        playheadMs: _playheadMs,
        onSeek: (ms) {
          _player?.pause();
          _seekMs(ms);
        },
        onThresholdChanged: (v) => _setParam(spec.paramName, v),
        onThresholdChangeEnd: (v) => _setParam(spec.paramName, v, save: true),
      ),
    );
  }

  Future<void> _openSettings() async {
    final settings = _settings ?? TunerSettings(videoDir: '', modelPath: '');
    final videoDirCtl = TextEditingController(text: settings.videoDir);
    final modelPathCtl = TextEditingController(text: settings.modelPath);
    final saved = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('路徑設定'),
        content: SizedBox(
          width: 560,
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              TextField(
                controller: videoDirCtl,
                decoration: const InputDecoration(
                  labelText: '影片資料夾 (docs/)',
                ),
              ),
              const SizedBox(height: 12),
              TextField(
                controller: modelPathCtl,
                decoration: const InputDecoration(
                  labelText: 'YOLO mlpackage 路徑 (ios/Runner/yolov8n_pose.mlpackage)',
                ),
              ),
            ],
          ),
        ),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(ctx, false),
              child: const Text('取消')),
          FilledButton(
              onPressed: () => Navigator.pop(ctx, true),
              child: const Text('儲存')),
        ],
      ),
    );
    if (saved == true) {
      settings.videoDir = videoDirCtl.text.trim();
      settings.modelPath = modelPathCtl.text.trim();
      await settings.save();
      setState(() => _settings = settings);
      _scanVideos();
    }
  }
}

// ---------------------------------------------------------------------------
// Metric timeline chart
// ---------------------------------------------------------------------------

class MetricTimelineChart extends StatefulWidget {
  final List<WindowEval> windows;
  final TuningSpec spec;
  final String cueKey;
  final double threshold;
  final int durationMs;
  final int playheadMs;
  final ValueChanged<int> onSeek;
  final ValueChanged<double> onThresholdChanged;
  final ValueChanged<double> onThresholdChangeEnd;

  const MetricTimelineChart({
    super.key,
    required this.windows,
    required this.spec,
    required this.cueKey,
    required this.threshold,
    required this.durationMs,
    required this.playheadMs,
    required this.onSeek,
    required this.onThresholdChanged,
    required this.onThresholdChangeEnd,
  });

  @override
  State<MetricTimelineChart> createState() => _MetricTimelineChartState();
}

class _MetricTimelineChartState extends State<MetricTimelineChart> {
  Offset? _hover;
  bool _draggingThreshold = false;

  static const _padL = 56.0;
  static const _padR = 14.0;
  static const _padT = 10.0;
  static const _padB = 30.0;

  (double, double) _valueRange() {
    final values = [
      for (final w in widget.windows)
        if (w.metrics[widget.spec.metricKey] != null)
          w.metrics[widget.spec.metricKey]!,
      widget.threshold,
    ];
    var lo = values.reduce(math.min);
    var hi = values.reduce(math.max);
    if (hi - lo < 1e-9) {
      lo -= 1;
      hi += 1;
    }
    final pad = (hi - lo) * 0.12;
    return (lo - pad, hi + pad);
  }

  double _yToValue(double y, Size size) {
    final (lo, hi) = _valueRange();
    final plotH = size.height - _padT - _padB;
    final frac = 1 - ((y - _padT) / plotH);
    return (lo + frac * (hi - lo)).clamp(
      math.min(widget.spec.min, lo),
      math.max(widget.spec.max, hi),
    );
  }

  double _valueToY(double v, Size size) {
    final (lo, hi) = _valueRange();
    final plotH = size.height - _padT - _padB;
    return _padT + (1 - (v - lo) / (hi - lo)) * plotH;
  }

  int _xToMs(double x, Size size) {
    final plotW = size.width - _padL - _padR;
    final frac = ((x - _padL) / plotW).clamp(0.0, 1.0);
    return (frac * widget.durationMs).round();
  }

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(builder: (context, constraints) {
      final size = Size(constraints.maxWidth, constraints.maxHeight);
      return MouseRegion(
        onHover: (e) => setState(() => _hover = e.localPosition),
        onExit: (_) => setState(() => _hover = null),
        child: GestureDetector(
          onPanStart: (d) {
            final thresholdY = _valueToY(widget.threshold, size);
            _draggingThreshold =
                (d.localPosition.dy - thresholdY).abs() < 12 &&
                    d.localPosition.dx > _padL;
            if (!_draggingThreshold) {
              widget.onSeek(_xToMs(d.localPosition.dx, size));
            }
          },
          onPanUpdate: (d) {
            if (_draggingThreshold) {
              widget.onThresholdChanged(_yToValue(d.localPosition.dy, size));
            } else {
              widget.onSeek(_xToMs(d.localPosition.dx, size));
            }
          },
          onPanEnd: (_) {
            if (_draggingThreshold) {
              widget.onThresholdChangeEnd(widget.threshold);
              _draggingThreshold = false;
            }
          },
          onTapDown: (d) {
            final thresholdY = _valueToY(widget.threshold, size);
            if ((d.localPosition.dy - thresholdY).abs() >= 12) {
              widget.onSeek(_xToMs(d.localPosition.dx, size));
            }
          },
          child: CustomPaint(
            size: size,
            painter: _TimelinePainter(
              windows: widget.windows,
              spec: widget.spec,
              cueKey: widget.cueKey,
              threshold: widget.threshold,
              durationMs: widget.durationMs,
              playheadMs: widget.playheadMs,
              hover: _hover,
              valueRange: _valueRange(),
              padL: _padL,
              padR: _padR,
              padT: _padT,
              padB: _padB,
            ),
          ),
        ),
      );
    });
  }
}

class _TimelinePainter extends CustomPainter {
  final List<WindowEval> windows;
  final TuningSpec spec;
  final String cueKey;
  final double threshold;
  final int durationMs;
  final int playheadMs;
  final Offset? hover;
  final (double, double) valueRange;
  final double padL, padR, padT, padB;

  _TimelinePainter({
    required this.windows,
    required this.spec,
    required this.cueKey,
    required this.threshold,
    required this.durationMs,
    required this.playheadMs,
    required this.hover,
    required this.valueRange,
    required this.padL,
    required this.padR,
    required this.padT,
    required this.padB,
  });

  double _x(int ms, Size size) =>
      padL + (ms / math.max(1, durationMs)) * (size.width - padL - padR);

  double _y(double v, Size size) {
    final (lo, hi) = valueRange;
    return padT + (1 - (v - lo) / (hi - lo)) * (size.height - padT - padB);
  }

  void _text(Canvas canvas, String s, Offset pos, Color color,
      {double fontSize = 10, TextAlign align = TextAlign.left}) {
    final tp = TextPainter(
      text: TextSpan(
          text: s,
          style: TextStyle(
              color: color,
              fontSize: fontSize,
              fontFeatures: const [FontFeature.tabularFigures()])),
      textDirection: TextDirection.ltr,
      textAlign: align,
    )..layout();
    var dx = pos.dx;
    if (align == TextAlign.right) dx -= tp.width;
    if (align == TextAlign.center) dx -= tp.width / 2;
    tp.paint(canvas, Offset(dx, pos.dy));
  }

  @override
  void paint(Canvas canvas, Size size) {
    final plotRect = Rect.fromLTRB(
        padL, padT, size.width - padR, size.height - padB);

    // Panel background
    canvas.drawRRect(
      RRect.fromRectAndRadius(
          Rect.fromLTWH(0, 0, size.width, size.height), const Radius.circular(8)),
      Paint()..color = const Color(0xFF1C1F24),
    );

    if (windows.isEmpty) {
      _text(canvas, '（尚無評估資料）', Offset(size.width / 2, size.height / 2 - 6),
          Colors.grey, fontSize: 12, align: TextAlign.center);
      return;
    }

    final (lo, hi) = valueRange;

    // ── Trigger regions (for the selected cue) ───────────────────────────
    final triggerPaint = Paint()..color = kTriggerColor.withAlpha(38);
    for (int i = 0; i < windows.length; i++) {
      if (!windows[i].triggered.contains(cueKey)) continue;
      final x0 = _x(i == 0 ? windows[i].startMs : windows[i - 1].endMs, size);
      final x1 = _x(windows[i].endMs, size);
      canvas.drawRect(
          Rect.fromLTRB(x0, plotRect.top, x1, plotRect.bottom), triggerPaint);
    }

    // ── Grid + y labels ──────────────────────────────────────────────────
    final gridPaint = Paint()
      ..color = Colors.white.withAlpha(18)
      ..strokeWidth = 1;
    for (int i = 0; i <= 4; i++) {
      final v = lo + (hi - lo) * i / 4;
      final y = _y(v, size);
      canvas.drawLine(Offset(plotRect.left, y), Offset(plotRect.right, y), gridPaint);
      _text(canvas, v.toStringAsFixed(spec.decimals), Offset(padL - 6, y - 6),
          Colors.grey, align: TextAlign.right);
    }

    // ── X labels every 2s ────────────────────────────────────────────────
    for (int ms = 0; ms <= durationMs; ms += 2000) {
      final x = _x(ms, size);
      canvas.drawLine(Offset(x, plotRect.bottom),
          Offset(x, plotRect.bottom + 4), gridPaint..strokeWidth = 1);
      _text(canvas, '${ms ~/ 1000}s', Offset(x, plotRect.bottom + 8),
          Colors.grey, align: TextAlign.center);
    }

    // ── Other-error tick strip (bottom of plot) ──────────────────────────
    for (int i = 0; i < windows.length; i++) {
      final others =
          windows[i].triggered.where((t) => t != cueKey).isNotEmpty;
      if (!others) continue;
      final x = _x(windows[i].endMs, size);
      canvas.drawLine(
        Offset(x, plotRect.bottom - 6),
        Offset(x, plotRect.bottom),
        Paint()
          ..color = Colors.grey.shade500
          ..strokeWidth = 2,
      );
    }

    // ── Metric line ──────────────────────────────────────────────────────
    final linePaint = Paint()
      ..color = kMetricLineColor
      ..strokeWidth = 2
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round;
    Path? path;
    for (final w in windows) {
      final v = w.metrics[spec.metricKey];
      if (v == null) {
        if (path != null) canvas.drawPath(path, linePaint);
        path = null;
        continue;
      }
      final p = Offset(_x(w.endMs, size), _y(v, size));
      if (path == null) {
        path = Path()..moveTo(p.dx, p.dy);
      } else {
        path.lineTo(p.dx, p.dy);
      }
    }
    if (path != null) canvas.drawPath(path, linePaint);

    // ── Threshold line (dashed, draggable) ───────────────────────────────
    final ty = _y(threshold, size);
    final dashPaint = Paint()
      ..color = kThresholdColor
      ..strokeWidth = 2;
    for (double x = plotRect.left; x < plotRect.right; x += 10) {
      canvas.drawLine(
          Offset(x, ty), Offset(math.min(x + 5, plotRect.right), ty), dashPaint);
    }
    _text(
        canvas,
        '閾值 ${threshold.toStringAsFixed(spec.decimals)}',
        Offset(plotRect.right - 4, ty - 14),
        kThresholdColor,
        align: TextAlign.right);

    // ── Playhead ─────────────────────────────────────────────────────────
    final px = _x(playheadMs, size);
    canvas.drawLine(
      Offset(px, plotRect.top),
      Offset(px, plotRect.bottom),
      Paint()
        ..color = Colors.white70
        ..strokeWidth = 1.5,
    );

    // ── Hover tooltip ────────────────────────────────────────────────────
    final h = hover;
    if (h != null && plotRect.contains(h)) {
      WindowEval? nearest;
      double bestDx = double.infinity;
      for (final w in windows) {
        final dx = (_x(w.endMs, size) - h.dx).abs();
        if (dx < bestDx) {
          bestDx = dx;
          nearest = w;
        }
      }
      if (nearest != null && bestDx < 30) {
        final v = nearest.metrics[spec.metricKey];
        final x = _x(nearest.endMs, size);
        if (v != null) {
          final y = _y(v, size);
          canvas.drawCircle(Offset(x, y), 4,
              Paint()..color = kMetricLineColor);
          canvas.drawCircle(
              Offset(x, y),
              5.5,
              Paint()
                ..color = const Color(0xFF1C1F24)
                ..style = PaintingStyle.stroke
                ..strokeWidth = 2);
        }
        final lines = [
          '${(nearest.endMs / 1000).toStringAsFixed(2)}s',
          v == null
              ? '${spec.metricKey}: —'
              : '${spec.metricKey}: ${v.toStringAsFixed(spec.decimals)}${spec.unit}',
          if (nearest.triggered.isNotEmpty) '觸發: ${nearest.triggered.join(", ")}',
        ];
        final tp = TextPainter(
          text: TextSpan(
            text: lines.join('\n'),
            style: const TextStyle(color: Colors.white, fontSize: 11, height: 1.4),
          ),
          textDirection: TextDirection.ltr,
        )..layout(maxWidth: 260);
        var boxX = x + 10;
        if (boxX + tp.width + 12 > plotRect.right) boxX = x - tp.width - 22;
        final boxY = math.max(plotRect.top + 4, h.dy - tp.height - 16);
        canvas.drawRRect(
          RRect.fromRectAndRadius(
              Rect.fromLTWH(boxX, boxY, tp.width + 12, tp.height + 10),
              const Radius.circular(5)),
          Paint()..color = const Color(0xE6262B33),
        );
        tp.paint(canvas, Offset(boxX + 6, boxY + 5));
      }
    }
  }

  @override
  bool shouldRepaint(_TimelinePainter old) =>
      old.windows != windows ||
      old.threshold != threshold ||
      old.playheadMs != playheadMs ||
      old.cueKey != cueKey ||
      old.hover != hover ||
      old.durationMs != durationMs;
}
