import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import '../database/database_helper.dart';
import '../services/gemini_service.dart';

// ---------------------------------------------------------------------------
// History Screen — Flutter port of Android's HistoryScreen.kt
// ---------------------------------------------------------------------------

class HistoryScreen extends StatefulWidget {
  const HistoryScreen({super.key});

  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  String? _selectedUser;

  @override
  Widget build(BuildContext context) {
    if (_selectedUser == null) {
      return UserSelectionScreen(
        onBack: null, // no back button — this is the root of the History tab
        onUserSelected: (user) => setState(() => _selectedUser = user),
      );
    }
    return UserHistoryScreen(
      userName: _selectedUser!,
      onBack: () => setState(() => _selectedUser = null),
    );
  }
}

// ---------------------------------------------------------------------------
// User Selection Screen
// ---------------------------------------------------------------------------

class UserSelectionScreen extends StatefulWidget {
  final VoidCallback? onBack; // null = no back button
  final void Function(String user) onUserSelected;

  const UserSelectionScreen({
    super.key,
    required this.onBack,
    required this.onUserSelected,
  });

  @override
  State<UserSelectionScreen> createState() => _UserSelectionScreenState();
}

class _UserSelectionScreenState extends State<UserSelectionScreen> {
  List<String> _users = [];
  bool _loading = true;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    final users = await DatabaseHelper().getDistinctUsers();
    if (mounted) setState(() { _users = users; _loading = false; });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF101418),
      appBar: AppBar(
        backgroundColor: const Color(0xFF1E262F),
        automaticallyImplyLeading: false,
        leading: widget.onBack != null
            ? IconButton(
                icon: const Icon(Icons.arrow_back, color: Colors.white),
                onPressed: widget.onBack,
              )
            : null,
        title: const Text('Select User', style: TextStyle(color: Colors.white)),
      ),
      body: _loading
          ? const Center(child: CircularProgressIndicator(color: Color(0xFF2E6DD1)))
          : _users.isEmpty
              ? const Center(
                  child: Text('No history yet.', style: TextStyle(color: Colors.grey, fontSize: 18)),
                )
              : ListView.separated(
                  padding: const EdgeInsets.all(16),
                  itemCount: _users.length,
                  separatorBuilder: (_, __) => const SizedBox(height: 12),
                  itemBuilder: (context, i) {
                    final user = _users[i];
                    return Card(
                      color: const Color(0xFF1E262F),
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                      child: ListTile(
                        onTap: () => widget.onUserSelected(user),
                        leading: CircleAvatar(
                          backgroundColor: const Color(0xFF2E6DD1),
                          child: Text(
                            user.isNotEmpty ? user[0].toUpperCase() : '?',
                            style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
                          ),
                        ),
                        title: Text(user, style: const TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.w600)),
                        trailing: const Icon(Icons.chevron_right, color: Colors.white54),
                      ),
                    );
                  },
                ),
    );
  }
}

// ---------------------------------------------------------------------------
// User History Screen
// ---------------------------------------------------------------------------

class UserHistoryScreen extends StatefulWidget {
  final String userName;
  final VoidCallback onBack;

  const UserHistoryScreen({super.key, required this.userName, required this.onBack});

  @override
  State<UserHistoryScreen> createState() => _UserHistoryScreenState();
}

class _UserHistoryScreenState extends State<UserHistoryScreen> {
  final _db = DatabaseHelper();
  List<SessionRecord> _sessions = [];
  bool _loading = true;
  int _refreshToken = 0;
  bool _selectionMode = false;
  Set<int> _selectedIds = {};
  bool _historyExpanded = true;
  int _recapCount = 5;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    setState(() => _loading = true);
    final sessions = await _db.getSessionsByUser(widget.userName);
    if (mounted) setState(() { _sessions = sessions; _loading = false; });
  }

  Future<void> _deleteSelected() async {
    for (final id in _selectedIds) {
      await _db.deleteSession(id);
    }
    setState(() {
      _selectedIds = {};
      _selectionMode = false;
      _refreshToken++;
    });
    await _load();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF101418),
      appBar: AppBar(
        backgroundColor: const Color(0xFF1E262F),
        leading: IconButton(
          icon: const Icon(Icons.arrow_back, color: Colors.white),
          onPressed: () {
            if (_selectionMode) {
              setState(() { _selectionMode = false; _selectedIds = {}; });
            } else {
              widget.onBack();
            }
          },
        ),
        title: Text(
          _selectionMode ? '${_selectedIds.length} selected' : "${widget.userName}'s History",
          style: const TextStyle(color: Colors.white),
        ),
        actions: [
          if (_selectionMode) ...[
            TextButton(
              onPressed: _sessions.isEmpty ? null : () => setState(() => _selectedIds = _sessions.map((s) => s.id).toSet()),
              child: const Text('All', style: TextStyle(color: Colors.white)),
            ),
            TextButton(
              onPressed: _selectedIds.isEmpty ? null : () {
                showDialog(
                  context: context,
                  builder: (_) => AlertDialog(
                    title: const Text('Delete selected history?'),
                    content: Text('Delete ${_selectedIds.length} session(s). This cannot be undone.'),
                    actions: [
                      TextButton(onPressed: () => Navigator.pop(context), child: const Text('Cancel')),
                      TextButton(
                        onPressed: () { Navigator.pop(context); _deleteSelected(); },
                        child: const Text('Delete', style: TextStyle(color: Color(0xFFFF6B6B))),
                      ),
                    ],
                  ),
                );
              },
              child: Text(
                'Delete',
                style: TextStyle(color: _selectedIds.isEmpty ? Colors.grey : const Color(0xFFFF6B6B)),
              ),
            ),
          ] else
            TextButton(
              onPressed: _sessions.isEmpty ? null : () => setState(() => _selectionMode = true),
              child: Text('Select', style: TextStyle(color: _sessions.isEmpty ? Colors.grey : Colors.white)),
            ),
        ],
      ),
      body: _loading
          ? const Center(child: CircularProgressIndicator(color: Color(0xFF2E6DD1)))
          : _sessions.isEmpty
              ? const Center(child: Text('No sessions found.', style: TextStyle(color: Colors.grey, fontSize: 18)))
              : ListView(
                  padding: const EdgeInsets.all(16),
                  children: [
                    _RecapCard(
                      sessions: _sessions.take(_recapCount).toList(),
                      recapCount: _recapCount,
                      onRecapCountChanged: (v) => setState(() => _recapCount = v),
                    ),
                    const SizedBox(height: 16),
                    // Session History header
                    GestureDetector(
                      onTap: () => setState(() => _historyExpanded = !_historyExpanded),
                      child: Container(
                        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                        decoration: BoxDecoration(
                          color: const Color(0xFF1E262F),
                          borderRadius: BorderRadius.circular(8),
                        ),
                        child: Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            const Text(
                              'Session History',
                              style: TextStyle(color: Color(0xFF64B5F6), fontSize: 18, fontWeight: FontWeight.w800, letterSpacing: 0.5),
                            ),
                            Icon(
                              _historyExpanded ? Icons.keyboard_arrow_up : Icons.keyboard_arrow_down,
                              color: const Color(0xFF64B5F6),
                            ),
                          ],
                        ),
                      ),
                    ),
                    if (_historyExpanded) ...[
                      const SizedBox(height: 12),
                      ..._sessions.map((session) {
                        final selected = _selectedIds.contains(session.id);
                        return Padding(
                          padding: const EdgeInsets.only(bottom: 12),
                          child: _SessionCard(
                            session: session,
                            selectionMode: _selectionMode,
                            selected: selected,
                            onSelectedChange: (v) => setState(() {
                              if (v) { _selectedIds.add(session.id); } else { _selectedIds.remove(session.id); }
                            }),
                            onTap: () async {
                              if (_selectionMode) {
                                setState(() {
                                  if (selected) { _selectedIds.remove(session.id); } else { _selectedIds.add(session.id); }
                                });
                              } else {
                                final full = await _db.getFullSession(session.id);
                                if (full != null && context.mounted) {
                                  Navigator.of(context).push(
                                    MaterialPageRoute(builder: (_) => SessionDetailScreen(data: full)),
                                  );
                                }
                              }
                            },
                          ),
                        );
                      }),
                    ],
                  ],
                ),
    );
  }
}

// ---------------------------------------------------------------------------
// Recap Card
// ---------------------------------------------------------------------------

class _RecapCard extends StatefulWidget {
  final List<SessionRecord> sessions;
  final int recapCount;
  final void Function(int) onRecapCountChanged;

  const _RecapCard({required this.sessions, required this.recapCount, required this.onRecapCountChanged});

  @override
  State<_RecapCard> createState() => _RecapCardState();
}

class _RecapCardState extends State<_RecapCard> {
  String? _aiAnalysis;
  bool _generating = false;

  @override
  void initState() {
    super.initState();
    _generateRecap();
  }

  @override
  void didUpdateWidget(_RecapCard old) {
    super.didUpdateWidget(old);
    if (old.sessions.length != widget.sessions.length ||
        old.recapCount != widget.recapCount) {
      _generateRecap();
    }
  }

  String _buildFallback() {
    if (widget.sessions.isEmpty) return '最近沒有明顯重複錯誤，保持目前節奏。';

    // Aggregate error counts across sessions
    final Map<String, int> totals = {};
    for (final s in widget.sessions) {
      for (final e in s.errorCounts.entries) {
        totals[e.key] = (totals[e.key] ?? 0) + e.value;
      }
    }
    if (totals.isEmpty) return '最近沒有明顯重複錯誤，保持目前節奏。';

    final topError = totals.entries.reduce((a, b) => a.value > b.value ? a : b);
    final half = math.max(1, (widget.sessions.length * 0.5).round());
    final recent = widget.sessions.take(half).fold(0, (s, r) => s + (r.errorCounts[topError.key] ?? 0));
    final old = widget.sessions.skip(half).fold(0, (s, r) => s + (r.errorCounts[topError.key] ?? 0));

    String trend;
    if (widget.sessions.length < 2) {
      trend = '資料還少，先專注在這個問題就好！';
    } else if (old == 0) {
      trend = '近期才開始出現，需要關注。';
    } else {
      final pct = (((old - recent) / old) * 100).round();
      trend = pct > 0 ? '近期比之前改善 $pct%。' : (pct < 0 ? '近期比之前增加 ${-pct}%，需要優先處理。' : '近期和之前大致持平。');
    }
    return '【 近期重點: ${topError.key}】\n$trend';
  }

  Future<void> _generateRecap() async {
    final fallback = _buildFallback();
    setState(() { _aiAnalysis = fallback; _generating = false; });

    final gemini = GeminiService();
    if (!gemini.isEnabled || widget.sessions.isEmpty) return;

    setState(() => _generating = true);
    final errorsText = widget.sessions.map((s) {
      final counts = s.errorCounts.entries.map((e) => '${e.key}: ${e.value}').join(', ');
      return '${DateFormat('yyyy/MM/dd').format(s.date)}: ${counts.isEmpty ? 'no errors' : counts}';
    }).join('\n');

    try {
      final result = await gemini.generateRecapAnalysis(
        userName: widget.sessions.first.userName,
        recentErrorsText: errorsText,
        recapCount: widget.recapCount,
        fallback: fallback,
      );
      if (mounted) setState(() { _aiAnalysis = result; _generating = false; });
    } catch (_) {
      if (mounted) setState(() => _generating = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Card(
      color: const Color(0xFF141420),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      elevation: 8,
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Expanded(
                  child: Text(
                    'Last ${widget.recapCount} Sessions Recap',
                    style: const TextStyle(color: Color(0xFF64B5F6), fontSize: 18, fontWeight: FontWeight.w800, letterSpacing: 0.5),
                  ),
                ),
                PopupMenuButton<int>(
                  icon: const Icon(Icons.settings, color: Colors.white),
                  color: const Color(0xFF263647),
                  onSelected: widget.onRecapCountChanged,
                  itemBuilder: (_) => [5, 10, 15, 20].map((n) =>
                    PopupMenuItem<int>(value: n, child: Text('Last $n Sessions', style: const TextStyle(color: Colors.white)))
                  ).toList(),
                ),
              ],
            ),
            const SizedBox(height: 16),
            if (widget.sessions.isEmpty)
              const Text('No sessions found for this filter.', style: TextStyle(color: Colors.grey))
            else if (_generating)
              const Row(children: [
                SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2, color: Color(0xFF64B5F6))),
                SizedBox(width: 12),
                Text('Generating Recap...', style: TextStyle(color: Colors.grey, fontSize: 15)),
              ])
            else if (_aiAnalysis != null)
              Text(_aiAnalysis!, style: const TextStyle(color: Color(0xFFB6C2CC), fontSize: 15, height: 1.6)),
            const SizedBox(height: 16),
            // Mini error breakdown chart
            _ErrorBreakdown(sessions: widget.sessions),
          ],
        ),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Error breakdown bar chart
// ---------------------------------------------------------------------------

class _ErrorBreakdown extends StatelessWidget {
  final List<SessionRecord> sessions;
  const _ErrorBreakdown({required this.sessions});

  @override
  Widget build(BuildContext context) {
    final totals = <String, int>{};
    for (final s in sessions) {
      for (final e in s.errorCounts.entries) {
        totals[e.key] = (totals[e.key] ?? 0) + e.value;
      }
    }
    if (totals.isEmpty) return const SizedBox.shrink();

    final sorted = totals.entries.toList()..sort((a, b) => b.value.compareTo(a.value));
    final maxVal = sorted.first.value.toDouble();

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text('Mistake Breakdown', style: TextStyle(color: Color(0xFF64B5F6), fontSize: 16, fontWeight: FontWeight.w800, letterSpacing: 0.5)),
        const SizedBox(height: 10),
        ...sorted.take(5).map((entry) {
          final ratio = entry.value / maxVal;
          return Padding(
            padding: const EdgeInsets.only(bottom: 8),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('${entry.key}  (${entry.value}x)', style: const TextStyle(color: Colors.white70, fontSize: 12)),
                const SizedBox(height: 4),
                ClipRRect(
                  borderRadius: BorderRadius.circular(4),
                  child: LinearProgressIndicator(
                    value: ratio,
                    backgroundColor: const Color(0xFF263647),
                    valueColor: const AlwaysStoppedAnimation<Color>(Color(0xFF2E6DD1)),
                    minHeight: 8,
                  ),
                ),
              ],
            ),
          );
        }),
      ],
    );
  }
}

// ---------------------------------------------------------------------------
// Session Card
// ---------------------------------------------------------------------------

class _SessionCard extends StatelessWidget {
  final SessionRecord session;
  final bool selectionMode;
  final bool selected;
  final void Function(bool) onSelectedChange;
  final VoidCallback onTap;

  const _SessionCard({
    required this.session,
    required this.selectionMode,
    required this.selected,
    required this.onSelectedChange,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final fmt = DateFormat('yyyy/MM/dd HH:mm');
    final durationSec = session.durationMs ~/ 1000;

    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 150),
        decoration: BoxDecoration(
          color: selected ? const Color(0xFF263647) : const Color(0xFF1E262F),
          borderRadius: BorderRadius.circular(12),
          border: selected ? Border.all(color: const Color(0xFF00D4FF), width: 1.5) : null,
        ),
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Row(
            children: [
              if (selectionMode) ...[
                Checkbox(
                  value: selected,
                  onChanged: (v) => onSelectedChange(v ?? false),
                  activeColor: const Color(0xFF00D4FF),
                ),
                const SizedBox(width: 8),
              ],
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Text(fmt.format(session.date), style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 15)),
                        Text('${durationSec}s', style: const TextStyle(color: Colors.grey, fontSize: 13)),
                      ],
                    ),
                    const SizedBox(height: 8),
                    Wrap(
                      spacing: 8,
                      children: [
                        _Chip(session.trainingMode),
                        _Chip(session.source),
                        if (session.cueCount > 0) _Chip('${session.cueCount} cues', color: const Color(0xFF4CAF50)),
                      ],
                    ),
                    if (session.llmSummary != null) ...[
                      const SizedBox(height: 6),
                      const Text('✓ AI Summary', style: TextStyle(color: Color(0xFFE57373), fontSize: 12, fontWeight: FontWeight.bold)),
                    ],
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _Chip extends StatelessWidget {
  final String label;
  final Color? color;
  const _Chip(this.label, {this.color});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(
        color: (color ?? const Color(0xFF2E6DD1)).withValues(alpha: 0.2),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: (color ?? const Color(0xFF2E6DD1)).withValues(alpha: 0.5), width: 1),
      ),
      child: Text(label, style: TextStyle(color: color ?? const Color(0xFF90CAF9), fontSize: 12, fontWeight: FontWeight.w600)),
    );
  }
}

// ---------------------------------------------------------------------------
// Session Detail Screen
// ---------------------------------------------------------------------------

class SessionDetailScreen extends StatelessWidget {
  final FullSessionData data;
  const SessionDetailScreen({super.key, required this.data});

  @override
  Widget build(BuildContext context) {
    final session = data.session;
    final fmt = DateFormat('yyyy/MM/dd HH:mm');
    final durationSec = session.durationMs ~/ 1000;

    return Scaffold(
      backgroundColor: const Color(0xFF101418),
      appBar: AppBar(
        backgroundColor: const Color(0xFF1E262F),
        title: Text(fmt.format(session.date), style: const TextStyle(color: Colors.white, fontSize: 16)),
        leading: IconButton(
          icon: const Icon(Icons.arrow_back, color: Colors.white),
          onPressed: () => Navigator.pop(context),
        ),
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          // Overview card
          _InfoCard(children: [
            _InfoRow('Training Mode', session.trainingMode),
            _InfoRow('Target Side', session.targetSide),
            _InfoRow('Duration', '${durationSec}s'),
            _InfoRow('Source', session.source),
            _InfoRow('Top Action', session.topAction),
            _InfoRow('Cues Fired', '${session.cueCount}'),
          ]),
          const SizedBox(height: 16),
          // AI Summary
          if (session.llmSummary != null) ...[
            _SectionTitle('AI Summary'),
            Card(
              color: const Color(0xFF1E262F),
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Text(session.llmSummary!, style: const TextStyle(color: Color(0xFFB6C2CC), fontSize: 14, height: 1.6)),
              ),
            ),
            const SizedBox(height: 16),
          ],
          // Error counts
          if (session.errorCounts.isNotEmpty) ...[
            _SectionTitle('Error Counts'),
            _InfoCard(
              children: (session.errorCounts.entries.toList()
                    ..sort((a, b) => b.value.compareTo(a.value)))
                  .map((e) => _InfoRow(e.key, '${e.value}x'))
                  .toList(),
            ),
            const SizedBox(height: 16),
          ],
          // Cue timeline
          if (data.cues.isNotEmpty) ...[
            _SectionTitle('Feedback Timeline'),
            ...data.cues.map((cue) => Padding(
              padding: const EdgeInsets.only(bottom: 10),
              child: Card(
                color: const Color(0xFF1E262F),
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                child: ListTile(
                  dense: true,
                  leading: Text(
                    '${cue.timeSeconds.toStringAsFixed(1)}s',
                    style: const TextStyle(color: Color(0xFF64B5F6), fontWeight: FontWeight.bold, fontSize: 13),
                  ),
                  title: Text(cue.errorName, style: const TextStyle(color: Colors.white, fontSize: 14)),
                  subtitle: cue.practiceSuggestion.isNotEmpty
                      ? Text(cue.practiceSuggestion, style: const TextStyle(color: Colors.grey, fontSize: 12))
                      : null,
                ),
              ),
            )),
          ],
        ],
      ),
    );
  }
}

class _SectionTitle extends StatelessWidget {
  final String text;
  const _SectionTitle(this.text);

  @override
  Widget build(BuildContext context) => Padding(
    padding: const EdgeInsets.only(bottom: 10),
    child: Text(
      text,
      style: const TextStyle(color: Color(0xFF64B5F6), fontSize: 16, fontWeight: FontWeight.w800, letterSpacing: 0.5),
    ),
  );
}

class _InfoCard extends StatelessWidget {
  final List<Widget> children;
  const _InfoCard({required this.children});

  @override
  Widget build(BuildContext context) {
    return Card(
      color: const Color(0xFF1E262F),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 8),
        child: Column(children: children),
      ),
    );
  }
}

class _InfoRow extends StatelessWidget {
  final String label;
  final String value;
  const _InfoRow(this.label, this.value);

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: const TextStyle(color: Colors.grey, fontSize: 14)),
          Text(value, style: const TextStyle(color: Colors.white, fontSize: 14, fontWeight: FontWeight.w600)),
        ],
      ),
    );
  }
}
