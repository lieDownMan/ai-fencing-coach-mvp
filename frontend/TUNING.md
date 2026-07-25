# Heuristics 調參流程 (Threshold Tuning Workflow)

所有可調閾值集中在 [`HeuristicsConfig`](lib/heuristics/heuristics_engine.dart)（16 個參數，
含目前預設值與註解）。動作分類的信心閾值另外在
[`FenceNetBridge.swift`](ios/Runner/FenceNetBridge.swift)（`confidenceThreshold = 0.6`）。
`lib/heuristics/fencenet_classifier.dart` 是 TFLite 備用路徑，iOS 上**沒有被使用**，
改它的 0.6 不會有任何效果。

調參不是直接改數字試手感——流程是「錄數據 → 看分佈 → 挑分離點」：

## 1. 錄數據（手機上，5 分鐘）

1. 真機跑 app，切到 **Debug** tab，打開「錄製調參數據」。
2. 做動作。每段錄一種內容，10–20 秒就夠：
   - 一段**全部做對**的（正確 en garde、正確步伐、正確弓步）
   - 每種要調的錯誤各錄一段**刻意做錯**的（站太高、步太小、防守揮太大…）
3. 關掉開關 → 檔案存在 檔案App ▸ AI Fencing Coach ▸ tuning/，AirDrop 到電腦。

格式是 JSONL，每行一幀：時間戳、當下動作分類+信心值、全部關節座標。

## 2. 標註（電腦上，改檔名就是標註）

```
good__enGarde_luther.jsonl
good__footwork_luther.jsonl
stance_too_high__luther.jsonl
narrow_step__luther.jsonl
over_parrying__retreat_test.jsonl
```

`__` 前面的字＝標籤：`good` 或任一 error key。

## 3. 回放 + 看分佈

```bash
cd frontend
flutter test test/tuning/replay_recordings_test.dart \
  --dart-define=TUNE_DIR=/path/to/recordings
```

（`--dart-define=TUNE_TARGET_SIDE=right`、`--dart-define=TUNE_MODE="Target Practice"`
可覆寫回放時的側別/模式。）

輸出三塊：

- **每檔觸發了什麼**——good 檔觸發任何東西＝誤報，錯誤檔沒觸發自己的 key＝漏報
- **每個指標在每個標籤下的分佈**（p05/p50/p95/min/max）——調參的核心依據
- **目前的閾值**——對照用

## 4. 挑閾值

原則：**先保精確率**。語音教練誤報比漏報煩十倍。

- 閾值放在 `good` 分佈與錯誤分佈之間的分離帶上，偏向 good 那側留餘裕
  （例：good 的膝角 p95=152°、stance_too_high 的 p05=171° → 閾值 165 比 170 好）
- 兩個分佈重疊嚴重＝這條規則的指標本身不夠好，調閾值救不了，要改指標
- 改 `HeuristicsConfig` 的預設值 → 跑 `flutter test` 確認單元測試還過 → 重跑回放確認
  good 誤報率下降/歸零

**或者直接把錄好的 .jsonl 丟給 AI**（放進 repo 或給路徑），說「幫我 tune」——
上面整條分析迴圈 AI 可以自己跑完並直接改好 config。

## 5. 動作分類（FenceNet）的信心閾值

錄製檔每幀都帶當下的 `action` + `conf`（原始信心值，未過閾值前）。如果發現
Idle 誤判多（conf 常在 0.5–0.6 徘徊被砍成 Idle）或亂報動作（低 conf 卻過了 0.6），
改 `FenceNetBridge.swift` 的 `confidenceThreshold`。分佈同樣從錄製檔看得到。

## 指標 ↔ 閾值對照表

| 回放輸出的指標 | 對應閾值 (HeuristicsConfig) | 觸發方向 |
|---|---|---|
| `avg_front_knee_angle_deg` | `stanceTooHighAngleDeg` (170) | 大於 |
| `lunge_knee_angle_deg` | `lungeKneeMinAngleDeg` (90) | 小於 |
| `arm_extension_angle_deg` | `incompleteArmExtensionAngleDeg` (155) | 小於 |
| `bounce_ratio` | `bounceRatioThreshold` (0.33) | 大於 |
| `guard_below_pelvis_max_run_s` | `guardDroppedSeconds` (0.35 / 0.70) | 大於 |
| `parry_sweep_torso_ratio` | `overParryTorsoRatioThreshold` (1.2) | 大於 |
| `step_ratio_min/max/median` | `narrowStepRatioThreshold` (1.0) / `wideStepRatioThreshold` (3.0) | 小於 / 大於 |
| `com_ratio_min/max/median` | `comLeaningBackRatioThreshold` (0.35) / `comInFrontRatioThreshold` (0.65) | 小於 / 大於 |

優先驗證順序：`overParryTorsoRatioThreshold`（1.2 是新設的估計值）→
`bounceRatioThreshold`（Dart 0.33 vs Python 0.25 尚未用數據對齊）→
step/CoM 比值 → 膝角/手臂角（有生物力學依據，最不急）。
