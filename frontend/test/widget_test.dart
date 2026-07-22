import 'package:flutter_test/flutter_test.dart';

import 'package:frontend/main.dart';

void main() {
  testWidgets('App renders the main tab bar', (WidgetTester tester) async {
    await tester.pumpWidget(const FencingCoachApp());
    await tester.pump();

    expect(find.text('AI Fencing Coach'), findsOneWidget);
    expect(find.text('Live'), findsOneWidget);
    expect(find.text('Settings'), findsOneWidget);
    expect(find.text('History'), findsOneWidget);
  });
}
