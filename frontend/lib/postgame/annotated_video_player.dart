import 'dart:io';
import 'package:flutter/material.dart';
import 'package:video_player/video_player.dart';

import '../postgame/postgame_analyzer.dart';
import '../services/video_exporter.dart';

class AnnotatedVideoPlayerScreen extends StatefulWidget {
  final String videoPath;
  final PostgameReport report;

  const AnnotatedVideoPlayerScreen({
    super.key,
    required this.videoPath,
    required this.report,
  });

  @override
  State<AnnotatedVideoPlayerScreen> createState() => _AnnotatedVideoPlayerScreenState();
}

class _AnnotatedVideoPlayerScreenState extends State<AnnotatedVideoPlayerScreen> {
  late VideoPlayerController _controller;
  bool _isPlaying = false;
  bool _isExporting = false;

  @override
  void initState() {
    super.initState();
    _controller = VideoPlayerController.file(File(widget.videoPath))
      ..initialize().then((_) {
        setState(() {}); // Update to show first frame
        _controller.addListener(() {
          if (mounted) setState(() {});
        });
      });
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  void _togglePlayPause() {
    setState(() {
      if (_controller.value.isPlaying) {
        _controller.pause();
        _isPlaying = false;
      } else {
        _controller.play();
        _isPlaying = true;
      }
    });
  }

  Future<void> _exportToGallery() async {
    setState(() => _isExporting = true);
    try {
      final exportedPath = await VideoExporter.exportAnnotatedVideo(
        originalVideoPath: widget.videoPath,
        report: widget.report,
      );

      if (exportedPath != null) {
        final success = await VideoExporter.saveToGallery(exportedPath);
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text(success ? '已成功儲存至相簿 / Saved to Gallery!' : '儲存失敗 / Save failed.'),
              backgroundColor: success ? Colors.green : Colors.red,
            ),
          );
        }
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('發生錯誤 / Error: $e'),
            backgroundColor: Colors.red,
          ),
        );
      }
    } finally {
      if (mounted) setState(() => _isExporting = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    if (!_controller.value.isInitialized) {
      return const Scaffold(
        backgroundColor: Colors.black,
        body: Center(child: CircularProgressIndicator(color: Color(0xFFFF6600))),
      );
    }

    // Find current action / errors based on playback time
    final currentSec = _controller.value.position.inMilliseconds / 1000.0;
    PostgameTimelineItem? currentItem;
    for (var i = 0; i < widget.report.timeline.length; i++) {
      final item = widget.report.timeline[i];
      if (currentSec >= item.timeSeconds) {
        currentItem = item;
      } else {
        break; // timeline is sorted by timeSeconds
      }
    }

    final hasErrors = currentItem != null && currentItem.errors.isNotEmpty;
    final actionText = currentItem?.action ?? 'Idle';
    final actionColor = hasErrors ? Colors.redAccent : const Color(0xFFFF6600);

    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: const Color(0xFF141420),
        title: const Text('播放分析影片 / Annotated Playback', style: TextStyle(fontSize: 16)),
        actions: [
          _isExporting
              ? const Center(
                  child: Padding(
                    padding: EdgeInsets.symmetric(horizontal: 20),
                    child: SizedBox(
                      width: 20,
                      height: 20,
                      child: CircularProgressIndicator(color: Colors.white, strokeWidth: 2),
                    ),
                  ),
                )
              : IconButton(
                  icon: const Icon(Icons.download, color: Colors.blueAccent),
                  tooltip: '下載至相簿 (Download to Gallery)',
                  onPressed: _exportToGallery,
                ),
        ],
      ),
      body: SafeArea(
        child: Column(
          children: [
            Expanded(
              child: Center(
                child: AspectRatio(
                  aspectRatio: _controller.value.aspectRatio,
                  child: Stack(
                    alignment: Alignment.center,
                    children: [
                      VideoPlayer(_controller),
                      
                      // Draw current action text overlay
                      Positioned(
                        top: 20,
                        left: 20,
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Container(
                              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                              decoration: BoxDecoration(
                                color: actionColor.withAlpha(200),
                                borderRadius: BorderRadius.circular(8),
                              ),
                              child: Text(
                                actionText,
                                style: const TextStyle(
                                  color: Colors.white,
                                  fontWeight: FontWeight.bold,
                                  fontSize: 18,
                                ),
                              ),
                            ),
                            if (hasErrors)
                              ...currentItem.errors.map((e) => Container(
                                    margin: const EdgeInsets.only(top: 4),
                                    padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                                    decoration: BoxDecoration(
                                      color: Colors.redAccent.withAlpha(180),
                                      borderRadius: BorderRadius.circular(4),
                                    ),
                                    child: Text(
                                      e,
                                      style: const TextStyle(color: Colors.white, fontSize: 12),
                                    ),
                                  )),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
            _buildControls(),
          ],
        ),
      ),
    );
  }

  Widget _buildControls() {
    return Container(
      color: const Color(0xFF141420),
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: Row(
        children: [
          IconButton(
            icon: Icon(_isPlaying ? Icons.pause : Icons.play_arrow, color: Colors.white),
            onPressed: _togglePlayPause,
          ),
          Expanded(
            child: VideoProgressIndicator(
              _controller,
              allowScrubbing: true,
              colors: const VideoProgressColors(
                playedColor: Color(0xFFFF6600),
                bufferedColor: Colors.white24,
                backgroundColor: Colors.white10,
              ),
            ),
          ),
          const SizedBox(width: 16),
          Text(
            _formatDuration(_controller.value.position),
            style: const TextStyle(color: Colors.white, fontSize: 12),
          ),
        ],
      ),
    );
  }

  String _formatDuration(Duration duration) {
    String twoDigits(int n) => n.toString().padLeft(2, '0');
    final minutes = twoDigits(duration.inMinutes.remainder(60));
    final seconds = twoDigits(duration.inSeconds.remainder(60));
    return '$minutes:$seconds';
  }
}
