import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import '../database/app_database.dart';
import '../database/entities.dart';

class HistoryScreen extends StatefulWidget {
  const HistoryScreen({super.key});

  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  List<PracticeReport> _reports = [];
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _loadHistory();
  }

  Future<void> _loadHistory() async {
    final reports = await AppDatabase.instance.getAllPracticeReports();
    setState(() {
      _reports = reports;
      _isLoading = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0A0A0F),
      appBar: AppBar(
        title: const Text('Training History'),
        backgroundColor: const Color(0xFF14141F),
        elevation: 0,
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : _reports.isEmpty
              ? const Center(
                  child: Text(
                    'No training sessions recorded yet.',
                    style: TextStyle(color: Colors.white70),
                  ),
                )
              : ListView.builder(
                  padding: const EdgeInsets.all(16),
                  itemCount: _reports.length,
                  itemBuilder: (context, index) {
                    final report = _reports[index];
                    final date = DateTime.fromMillisecondsSinceEpoch(report.startTimeMs);
                    final formattedDate = DateFormat('MMM d, yyyy - h:mm a').format(date);
                    final duration = Duration(seconds: report.elapsedSeconds);
                    
                    return Card(
                      color: const Color(0xFF1C1C28),
                      margin: const EdgeInsets.only(bottom: 16),
                      child: ExpansionTile(
                        title: Text(
                          formattedDate,
                          style: const TextStyle(
                            color: Colors.white,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        subtitle: Text(
                          'Duration: ${duration.inMinutes}m ${duration.inSeconds % 60}s | Cues: ${report.cueTimeline.length}',
                          style: const TextStyle(color: Colors.white54),
                        ),
                        childrenPadding: const EdgeInsets.all(16),
                        children: [
                          if (report.llmSummary.isNotEmpty) ...[
                            const Align(
                              alignment: Alignment.centerLeft,
                              child: Text(
                                'AI Coach Summary',
                                style: TextStyle(
                                  color: Color(0xFFFF6600),
                                  fontWeight: FontWeight.bold,
                                  fontSize: 16,
                                ),
                              ),
                            ),
                            const SizedBox(height: 8),
                            Align(
                              alignment: Alignment.centerLeft,
                              child: Text(
                                report.llmSummary,
                                style: const TextStyle(color: Colors.white70, height: 1.5),
                              ),
                            ),
                            const SizedBox(height: 16),
                          ],
                          const Align(
                            alignment: Alignment.centerLeft,
                            child: Text(
                              'Action Counts',
                              style: TextStyle(
                                color: Colors.blueAccent,
                                fontWeight: FontWeight.bold,
                                fontSize: 16,
                              ),
                            ),
                          ),
                          const SizedBox(height: 8),
                          ...report.actionCounts.map((a) => Padding(
                                padding: const EdgeInsets.symmetric(vertical: 2),
                                child: Row(
                                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                                  children: [
                                    Text(a.action, style: const TextStyle(color: Colors.white70)),
                                    Text('${a.count}', style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
                                  ],
                                ),
                              )),
                        ],
                      ),
                    );
                  },
                ),
    );
  }
}
