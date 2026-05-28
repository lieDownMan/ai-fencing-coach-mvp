import 'package:google_generative_ai/google_generative_ai.dart';

/// Gemini-based fencing coach summary generator.
///
/// Mirrors the behaviour of the web backend's `llm_agent.py`.
/// The API key is injected at compile time via:
///   `flutter run --dart-define=GEMINI_API_KEY=your_key`
class GeminiService {
  static const _apiKey = String.fromEnvironment('GEMINI_API_KEY');

  bool get isEnabled => _apiKey.isNotEmpty;

  GenerativeModel? _model;

  GenerativeModel get _generativeModel {
    _model ??= GenerativeModel(
      model: 'gemini-1.5-flash',
      apiKey: _apiKey,
    );
    return _model!;
  }

  /// Generate a coaching summary from postgame data.
  ///
  /// Returns the generated text, or a fallback string if the API key
  /// is missing or the call fails.
  Future<String> generateSummary({
    required String trainingMode,
    required String targetSide,
    required Map<String, int> actionCounts,
    required Map<String, int> errorCounts,
    required Map<String, String> errorLabels,
    required String userName,
  }) async {
    if (!isEnabled) {
      return _offlineFallback(trainingMode, actionCounts, errorCounts);
    }

    final actionSummary = actionCounts.entries
        .map((e) => '${e.key}: ${e.value} times')
        .join('\n');

    final cueSummary = errorCounts.entries
        .map((e) => '- ${errorLabels[e.key] ?? e.key}: ${e.value} times')
        .join('\n');

    final prompt = '''
You are an expert, professional fencing coach. The user just completed a training session.
Provide a brief, encouraging, and highly technical summary of their session in Traditional Chinese.
Do not exceed 3 paragraphs. Use markdown formatting.

## Session Details
User: $userName
Mode: $trainingMode
Targeting: $targetSide

## Actions Performed
$actionSummary

## Detected Mistakes (Frequency)
$cueSummary
''';

    try {
      final content = [Content.text(prompt)];
      final response = await _generativeModel.generateContent(content);
      return response.text ?? '無法生成總結 (API 回傳空值)。';
    } catch (e) {
      return '**Gemini API Error:** $e';
    }
  }

  String _offlineFallback(
    String trainingMode,
    Map<String, int> actionCounts,
    Map<String, int> errorCounts,
  ) {
    final totalActions = actionCounts.values.fold<int>(0, (a, b) => a + b);
    return '*(No Gemini API Key found. Offline fallback summary)*\n'
        'Training Mode: $trainingMode\n'
        'Actions tracked: $totalActions\n'
        'Feedback cues: ${errorCounts.length}';
  }
}
