import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

class PostgameScreen extends StatefulWidget {
  const PostgameScreen({super.key});

  @override
  State<PostgameScreen> createState() => _PostgameScreenState();
}

class _PostgameScreenState extends State<PostgameScreen> {
  final ImagePicker _picker = ImagePicker();
  File? _selectedVideo;
  bool _isProcessing = false;
  String _statusMessage = 'Select a video to begin analysis.';

  Future<void> _pickVideo() async {
    final XFile? video = await _picker.pickVideo(source: ImageSource.gallery);
    if (video != null) {
      setState(() {
        _selectedVideo = File(video.path);
        _statusMessage = 'Video selected: \${video.name}';
      });
    }
  }

  Future<void> _analyzeVideo() async {
    if (_selectedVideo == null) return;
    
    setState(() {
      _isProcessing = true;
      _statusMessage = 'Analyzing video frames... (Mocked for MVP)';
    });

    // In a full implementation, you would:
    // 1. Extract frames using FFmpegKit
    // 2. Pass frames through YoloPoseService and FenceNetChannel
    // 3. Collect errors using HeuristicsEngine
    // 4. Save a PracticeReport to AppDatabase
    
    await Future.delayed(const Duration(seconds: 3));

    setState(() {
      _isProcessing = false;
      _statusMessage = 'Analysis complete! Check the History tab.';
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0A0A0F),
      appBar: AppBar(
        title: const Text('Postgame Analysis'),
        backgroundColor: const Color(0xFF14141F),
        elevation: 0,
      ),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(24.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(Icons.video_library, size: 80, color: Color(0xFFFF6600)),
              const SizedBox(height: 24),
              Text(
                _statusMessage,
                textAlign: TextAlign.center,
                style: const TextStyle(color: Colors.white70, fontSize: 16),
              ),
              const SizedBox(height: 40),
              if (_isProcessing)
                const CircularProgressIndicator(color: Color(0xFFFF6600))
              else ...[
                ElevatedButton.icon(
                  icon: const Icon(Icons.file_upload),
                  label: const Text('Choose Video'),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: const Color(0xFF1C1C28),
                    foregroundColor: Colors.white,
                    padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
                  ),
                  onPressed: _pickVideo,
                ),
                const SizedBox(height: 16),
                ElevatedButton.icon(
                  icon: const Icon(Icons.analytics),
                  label: const Text('Run Analysis'),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: _selectedVideo == null ? Colors.grey : const Color(0xFFFF6600),
                    foregroundColor: Colors.white,
                    padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
                  ),
                  onPressed: _selectedVideo == null ? null : _analyzeVideo,
                ),
              ]
            ],
          ),
        ),
      ),
    );
  }
}
