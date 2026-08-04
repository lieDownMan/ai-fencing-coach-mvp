# Heuristics 調參流程 (Threshold Tuning Workflow)

## 免走動路徑：Mac 鏡頭 + 手機遙控（tuning_server）

不想在手機腳架和站位之間來回走：讓 **Mac 的鏡頭拍你**，手機拿在手上當遙控器。

```bash
# Mac 上（repo 根目錄；第一次會跳相機權限）
venv/bin/python backend/tuning_server.py
```

手機瀏覽器開它印出的 `http://<Mac-IP>:8123`（同一個 Wi-Fi）——手機上就有：
即時畫面+骨架、大字指標、閾值滑桿、觸發語音提示、「複製 Dart 參數」。
指標計算與 App 引擎**逐行對齊**（同名參數、同數學、同預設值，
`--self-test` 可驗證），調出來的數字直接貼回 `HeuristicsConfig`。
調整值存在 `backend/tuning_overrides.json`（gitignored）。

## App 內建 Tuning 專區（手機單機、人工即時調）

App 的 **Tuning** tab 一次選一個錯誤：即時顯示該錯誤的指標數字與觸發狀態
（不經動作分類過濾），滑桿拖動閾值**立刻生效**並自動保存
（存於 Documents/tuning/config_overrides.json，重開 App 仍在，檔案App可見）。
流程：擺出「剛好該被唸」的動作 → 看當下數字 → 把滑桿拉到那附近 → 反覆驗證
good 動作不觸發、錯誤動作會觸發 → 按「複製全部參數」把 Dart 片段貼回
`HeuristicsConfig` 的預設值（或貼給 AI 代寫），tuned 值就固化進程式。

以下的錄製+回放流程是數據派做法，作為補充仍然可用。

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

**推薦：一鍵版（Python，直接建議閾值）**

```bash
venv/bin/python backend/tune_from_recordings.py /path/to/recordings
```

自動回放全部錄製檔（60 幀窗、每 10 幀滑動，與 App 相同）、印出 good vs
錯誤的指標分佈（p05/p50/p95）、在分離帶上偏 good 側建議閾值、輸出可直接
貼回 `HeuristicsConfig` 的 Dart snippet。分佈重疊時會標示「調閾值救不了」
而不是硬給數字。因為輸入是手機自己的關節數據，調出來的值就是手機口徑，
沒有 Mac↔手機的系統差。

**或者 Dart 版（多印每檔觸發了什麼）：**

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
| `lunge_knee_angle_deg` | `lungeKneeMinAngleDeg` (153.5) | 小於 |
| `arm_extension_angle_deg` | `incompleteArmExtensionAngleDeg` (101) | 小於 |
| `bounce_ratio` | `bounceRatioThreshold` (0.13) | 大於 |
| `guard_below_pelvis_max_run_s` | `guardDroppedSeconds` (3.0) | 大於 |
| `parry_sweep_torso_ratio` | `overParryTorsoRatioThreshold` (0.54) | 大於 |
| `step_ratio_min/max/median` | `narrowStepRatioThreshold` (0.9) / `wideStepRatioThreshold` (3.0) | 小於 / 大於 |
| `torso_lean_deg_min/max/median`（軀幹傾角，°，前傾為正） | `comBackwardLeanDeg` (3.0) / `comForwardLeanDeg` (21) | 小於 / 大於 |
| `foot_hand_lead_s`（腳比手早動幾秒） | `footBeforeHandLeadSeconds` (0.37) | 大於等於 |

預設值為 2026-07 於實機人工調校的結果（Tuning tab / Mac tuning server），
全部 11 種錯誤皆已實測調校。注意後傾閾值為正值——自然 en garde 本身
帶有約 10° 前傾，低於 3° 即視為後仰。

另有一道不可調的守門：窗口內過半幀偵測不到人（骨盆＋前腳踝缺失）時，
引擎不評估也不輸出任何警示；App 端在連續約 1 秒偵測不到人後會清空
骨架緩衝，避免人回到畫面時拿舊窗口誤判。

foot_before_hand 的 onset 偵測（Dart/Python 同步）：手腳的身體相對前進序列
先過 5 幀滑動中位數濾波（固定常數，殺單幀關節抖動），riseOnset 的 baseline
用「峰前中位數」而非最小值——避免後撤造成的真實低點把腳的起跑點誤錨到
後撤段（retreat–lunge 誤報）。已知極限：後撤無停頓直接流入弓步時，2D 腳踝
序列原理上不可分割，由 leadSeconds（0.37，即腳早於手 0.37 秒才算錯）兜底。
