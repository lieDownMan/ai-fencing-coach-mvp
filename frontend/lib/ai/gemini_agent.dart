import 'package:google_generative_ai/google_generative_ai.dart';
import '../database/entities.dart';

class GeminiAgent {
  // Replace with actual API key or read from environment
  static const String _apiKey = String.fromEnvironment('GEMINI_API_KEY', defaultValue: '');
  
  bool get isEnabled => _apiKey.isNotEmpty;

  late final GenerativeModel _model;

  GeminiAgent() {
    if (isEnabled) {
      _model = GenerativeModel(
        model: 'gemini-1.5-flash',
        apiKey: _apiKey,
      );
    }
  }

  Future<String> generateSummary({
    required String trainingMode,
    required String targetSide,
    required List<ActionCountItem> actionCounts,
    required List<CueHistoryItem> cuesFired,
    required String userSettingsName,
  }) async {
    if (!isEnabled) {
      return "*(No Gemini API Key found. Offline fallback summary)*\n\n"
          "Training Mode: \$trainingMode\n"
          "Actions tracked: \${actionCounts.fold(0, (sum, item) => sum + item.count)}\n"
          "Feedback cues: \${cuesFired.length}";
    }

    final actionSummary = actionCounts.map((e) => "\${e.action}: \${e.count} times").join("\n");
    
    final Map<String, int> cueMap = {};
    for (var c in cuesFired) {
      cueMap[c.label] = (cueMap[c.label] ?? 0) + 1;
    }
    final cueSummary = cueMap.entries.map((e) => "- \${e.key}: \${e.value} times").join("\n");

    final prompt = '''
You are an expert, professional fencing coach. The user just completed a training session.
Provide a brief, encouraging, and highly technical summary of their session in Traditional Chinese.
Do not exceed 3 paragraphs. Use markdown formatting.

## Session Details
User: \$userSettingsName
Mode: \$trainingMode
Targeting: \$targetSide

## Actions Performed
\$actionSummary

## Detected Mistakes (Frequency)
\$cueSummary
''';

    try {
      final response = await _model.generateContent([Content.text(prompt)]);
      return response.text ?? "無法生成總結 (API 回傳空值)。";
    } catch (e) {
      return "**Gemini API Error:** \$e";
    }
  }
}
