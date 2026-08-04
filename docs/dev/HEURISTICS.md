# 擊劍幾何姿勢啟發式規則引擎 (Geometric Posture Heuristics Engine)

此文件詳細說明了 AI Fencing Coach 中幾何姿勢啟發式規則（Heuristics）的計算邏輯、偵錯規則與調參參數。

本系統包含兩個實作版本：
1. **Dart 版本 (前端)**：[`frontend/lib/heuristics/heuristics_engine.dart`](../../frontend/lib/heuristics/heuristics_engine.dart) —— 用於手機端 App 的即時姿態判定與語音/視覺回饋。
2. **Python 版本 (後端)**：[`backend/inference/heuristics_engine.py`](../../backend/inference/heuristics_engine.py) —— 用於影片片段分析與離線調參工具。

> [!NOTE]
> 手機 App（Dart 版本）針對行動裝置的效能波動與實時性進行了優化，相較於 Python 離線版本引入了時間窗平滑、身體相對特徵縮放以及持續時間判定的設計。

---

## 核心設計原則

為了讓幾何規則在多變的手機拍攝場景下保持穩定，Dart 引擎採用了以下設計：

1. **時間基底閾值 (Time-based Thresholds)**：
   手機上的 Pose FPS 會因硬體負載而波動。因此，判定框長度（Frame Count）均改用「秒數 × 實際量測 FPS」動態計算，避免相同動作在不同手機上觸發靈敏度不一。
2. **身體相對坐標與縮放 (Body-relative & Scaling)**：
   * 排除選手在畫面中的平移影響，許多指標（如手腕移動、步幅）皆相對於骨盆中心（Pelvis Center）或躯幹長度（Torso Length）計算。
   * 躯幹長度定義為**前肩到骨盆中心的歐氏距離**。在側身站立的擊劍視野中，雙肩寬度常塌陷至接近 0，因此使用軀幹長度作為像素空間的歸一化基準。
3. **朝向自適應 (Facing Auto-detection)**：
   系統透過前後腳踝的相對 X 坐標中位數（`_windowFacingSign`）自動判斷選手朝向（向左為 `+1`，向右為 `-1`），無需依賴寫死的設定。
4. **持續時間判定 (Sustained Duration)**：
   步幅過大/過小或重心過度傾斜等步法錯誤，必須在時間窗內**持續滿足**特定影格數（例如持續 0.3 秒）才觸發，避免選手在步伐切換的過渡影格中產生誤報。

---

## 啟發式判定規則詳解

以下為目前系統支援的 11 個幾何規則及其在 [`coach_playbook.json`](../../backend/coach_playbook.json) 中對應的教練提示詞與建議練習：

### 1. 過度彈跳 (`bounce_excessive`)
* **目的**：偵測實戰姿勢（En Garde）或步法移動時，身體重心是否產生過大的垂直起伏。
* **計算邏輯**：
  計算時間窗內骨盆中心 Y 軸的最大與最小差值（垂直起伏幅），除以整個人體邊界框（Bounding Box）的高度。
* **觸發條件**：
  $$\frac{\text{Pelvis } Y_{\max} - \text{Pelvis } Y_{\min}}{\text{Bounding Box Height}} > \text{bounceRatioThreshold } (0.13)$$
* **執行時機**：僅在步法動作（`SF` 進步、`SB` 退步）時評估。
* **教練提示與診斷**：
  * **錯誤名稱**：步伐上下浮動
  * **語音提示 (Short Cue)**：`"重心壓低蹲好，不要上下移動!"`
  * **診斷分析 (Diagnosis)**：`"在走腳步的時候大腿肌力或是控制力不足導致身體重心不穩，影響攻擊刺點"`
  * **建議自主練習 (Practice)**：`"連續前進、後退以穩定腳步"`

### 2. 弓步過度伸展 (`lunge_overextension`)
* **目的**：避免弓步（Lunge）時前腳膝蓋彎曲角度過大（小於 90°），導致重心過度前傾並造成關節受傷。
* **計算邏輯**：
  尋找時間窗內前腳踝相對於第一影格位移最大（即弓步落地最遠）的那一影格，計算該影格中前腳的 **髖—膝—踝角度**（Angle ABC，頂點在膝蓋）。
* **觸發條件**：
  $$\text{前膝角度} < \text{lungeKneeMinAngleDeg } (153.5^\circ)$$
* **執行時機**：僅在進攻動作（`R` 弓步、`IS` 躍步刺、`WW` 閃躲刺、`JS` 跳步刺）時評估。
* **教練提示與診斷**：
  * **錯誤名稱**：長刺過度前傾
  * **語音提示 (Short Cue)**：`"注意前腳不要壓膝蓋，長刺打到90度就好"`
  * **診斷分析 (Diagnosis)**：`"重心衝太快，前腳發力不夠，容易傷膝蓋且無法快速收步。"`
  * **建議自主練習 (Practice)**：`"長刺（lunge）重複練習"`

### 3. 防守崩潰/護手下垂 (`guard_dropped`)
* **目的**：警告選手實戰中劍手手腕是否低於骨盆，暴露有效擊中部位。
* **計算邏輯**：
  檢查前手手腕（Wrist）的 Y 坐標是否低於骨盆中心（Pelvis Center）（在影像空間中 Y 越大代表越低）。
* **觸發條件**：
  手腕 Y > 骨盆 Y 且持續影格數超過時間閾值（自由實戰模式預設為 3.0 秒，其餘模式預設為 3.0 秒）。
* **執行時機**：所有模式及動作均會持續評估。
* **教練提示與診斷**：
  * **錯誤名稱**：持劍手掉落
  * **語音提示 (Short Cue)**：`"手抬起來，劍尖指著對手"`
  * **診斷分析 (Diagnosis)**：`"攻擊或防守時手放太低，劍尖沒有威脅對手，且有效部位完全暴露給對手。"`
  * **建議自主練習 (Practice)**：無

### 4. 腳比手先動 (`foot_before_hand`)
* **目的**：經典擊劍錯誤。正確的進攻順序應該是「手先延伸，腳再蹬出」，若腳先動會暴露進攻意圖且容易被對手反擊。
* **計算邏輯**：
  1. 提取前手腕與前腳踝相對於骨盆中心的水平位移序列，並使用 5 影格中位數濾波器平滑。
  2. 從位移的全局峰值向回尋找**動作起點 (Onset)**：位移值首次達到「基準線 + 10% 總位移量」的影格。
  3. 比較前腳踝起點與前手腕起點的時間差。
* **觸發條件**：
  $$\text{腳踝動作起點影格} + \text{leadFrames} \le \text{手腕動作起點影格}$$
  （其中 `leadSeconds` 預設為 0.37 秒，即腳比手先起跑超過 0.37 秒時觸發）
* **執行時機**：僅在進攻動作（`R`, `IS`, `WW`, `JS`）時評估。
* **教練提示與診斷**：
  * **錯誤名稱**：手腳順序錯誤
  * **語音提示 (Short Cue)**：`"手要先伸"`
  * **診斷分析 (Diagnosis)**：`"腳先動了，劍尖還沒出去，這樣會先把自己的身體送出去給對方打，容易被得分。"`
  * **建議自主練習 (Practice)**：`"長刺（lunge）前進長刺（step lunge）重複練習"`

### 5. 站姿過高 (`stance_too_high`)
* **目的**：提醒選手實戰姿勢或步法移動中，雙膝未適度彎曲，重心太高。
* **計算邏輯**：
  計算時間窗內所有影格前膝關節角度（髖—膝—踝）的平均值。
* **觸發條件**：
  $$\text{平均膝關節角度} > \text{stanceTooHighAngleDeg } (170.0^\circ)$$
* **執行時機**：僅在步法動作（`SF`, `SB`）時評估。
* **教練提示與診斷**：
  * **錯誤名稱**：預備姿勢沒蹲好
  * **語音提示 (Short Cue)**：`"重心壓低，蹲好！"`
  * **診斷分析 (Diagnosis)**：`"重心太高會導致啟動速度變慢，無法瞬間爆發，也容易在快速移動中失去平衡。"`
  * **建議自主練習 (Practice)**：`"預備姿勢（en guard）練習"`

### 6. 手臂未完全伸展 (`incomplete_arm_extension`)
* **目的**：偵測進攻或刺擊時，劍手手臂是否彎曲、未完全伸直。
* **計算邏輯**：
  定位前手腕水平位移最大的影格，計算該影格中前臂的 **肩—肘—腕角度**。
* **觸發條件**：
  $$\text{手臂角度} < \text{incompleteArmExtensionAngleDeg } (101.0^\circ)$$
* **執行時機**：僅在進攻動作（`R`, `IS`, `WW`, `JS`）時評估。
* **教練提示與診斷**：
  * **錯誤名稱**：刺的時候手沒有伸直
  * **語音提示 (Short Cue)**：`"刺的時候手伸直！"`
  * **診斷分析 (Diagnosis)**：`"手臂沒有完全伸直會讓你平白損失攻擊距離，導致原本能擊中的攻擊落空。"`
  * **建議自主練習 (Practice)**：`"重複長刺（lunge）練習"`

### 7. 過度防守 (`over_parrying`)
* **目的**：警示撥劍（Parry）防守時，前手擺動幅度過大，導致門戶大開且難以快速還擊。
* **計算邏輯**：
  量測前手腕相對於骨盆中心的 X 軸擺動範圍，並除以軀幹長度（Torso Length）進行歸一化。
* **觸發條件**：
  $$\frac{\text{Wrist } X_{\max} - \text{Wrist } X_{\min}}{\text{Torso Length}} > \text{overParryTorsoRatioThreshold } (0.54)$$
* **執行時機**：在退步防守（`SB`）或自由實戰中的任何步法動作下評估。
* **教練提示與診斷**：
  * **錯誤名稱**：防守動作太大且太頻繁
  * **語音提示 (Short Cue)**：`"防守小一點，不要亂撈劍，用手指頭控制劍尖不要用整個手腕"`
  * **診斷分析 (Diagnosis)**：`"撥擋（Parry）的動作超過身體輪廓太多，導致防守後露出太多有效部位。對手很容易做一個簡單的轉移刺（Disengage）擊中你，且動作太大會導致回防速度變慢。"`
  * **建議自主練習 (Practice)**：`"重複防守（Parry）練習"`

### 8. 步幅過大/過小 (`wide_step` / `narrow_step`)
* **目的**：維持穩定且具備爆發力的前後腳間距。
* **計算邏輯**：
  1. **肩膀代理寬度 (Shoulder Proxy Width, `sw`)**：前肩與骨盆中心的水平距離 $\times 2.5$（用於模擬正面肩膀寬度）。
  2. **步幅比值 (Step Ratio)**：前後腳踝的水平距離除以 `sw`。
* **觸發條件**：
  * **步幅過大 (`wide_step`)**：步幅比值 $> \text{wideStepRatioThreshold } (3.0)$ 且持續時間超過 0.30 秒。
  * **步幅過小 (`narrow_step`)**：步幅比值 $< \text{narrowStepRatioThreshold } (0.9)$ 且持續時間超過 0.30 秒。
* **執行時機**：僅在步法動作（`SF`, `SB`）時評估。
* **教練提示與診斷**：
  * **步伐太大 (`wide_step`)**：
    * **語音提示 (Short Cue)**：`"腳打太開"`
    * **診斷分析 (Diagnosis)**：`"雙腳間距過大影響移動和動作發動"`
  * **步伐太小 (`narrow_step`)**：
    * **語音提示 (Short Cue)**：`"腳打太窄"`
    * **診斷分析 (Diagnosis)**：`"雙腳間距過小會影響移動和動作發動"`

### 9. 重心前傾/後仰 (`center_of_mass_in_front` / `center_of_mass_leaning_backward`)
* **目的**：檢視實戰步法中，上身躯幹是否過度前傾或後傾。
* **計算邏輯**：
  計算骨盆中心至前肩的連線相對於垂直線的夾角（Lean Angle）。向對手方向傾斜為正，背向對手為負。
* **觸發條件**：
  * **重心過度前傾**：夾角 $> \text{comForwardLeanDeg } (21.0^\circ)$ 且持續時間超過 0.30 秒。
  * **重心過度後仰**：夾角 $< \text{comBackwardLeanDeg } (3.0^\circ)$ 且持續時間超過 0.30 秒。
* **執行時機**：僅在步法動作（`SF`, `SB`）時評估。
* **教練提示與診斷**：
  * **重心過度前傾 (`center_of_mass_in_front`)**：
    * **語音提示 (Short Cue)**：`"重心太靠前"`
    * **診斷分析 (Diagnosis)**：`"姿勢不標準，會無法有正確迅速確實的發力過程"`
    * **建議自主練習 (Practice)**：`"提供身體應有基準給選手參考"`
  * **重心過度後仰 (`center_of_mass_leaning_backward`)**：
    * **語音提示 (Short Cue)**：`"重心太靠後"`
    * **診斷分析 (Diagnosis)**：`"姿勢不標準，會無法有正確迅速確實的發力過程"`
    * **建議自主練習 (Practice)**：`"提供身體應有基準給選手參考"`

---

## 參數對照表 (Tunable Config Parameters)

所有的啟發式閾值都封裝在 `HeuristicsConfig` 類別中，其預設值（根據 iOS/Android 真機手調優化）如下：

| 參數名稱 | 預設值 | 單位 | 說明 | 對應判定錯誤 |
| :--- | :--- | :--- | :--- | :--- |
| `bounceRatioThreshold` | `0.13` | 比例 | 骨盆 Y 起伏佔 Bounding Box 高度之最大比例 | `bounce_excessive` |
| `lungeKneeMinAngleDeg` | `153.5` | 度 (°) | 弓步落地時，前腳膝關節允許的最小角度 | `lunge_overextension` |
| `guardDroppedSeconds` | `3.0` | 秒 (s) | 一般模式下手腕低於骨盆觸發警告的持續秒數 | `guard_dropped` |
| `guardDroppedFreeBoutingSeconds` | `3.0` | 秒 (s) | 自由實戰下手腕低於骨盆觸發警告的持續秒數 | `guard_dropped` |
| `footBeforeHandMinDisplacement` | `0.03` | 比例 | 手腳位移被視為「動作開始」的最小歸一化位移量 | 動作起點判定 |
| `footBeforeHandLeadSeconds` | `0.37` | 秒 (s) | 腳踝動作起點領先手腕起點的最小秒數 | `foot_before_hand` |
| `stance_too_high` | `170.0` | 度 (°) | 實戰姿勢中膝關節平均角度上限 | `stance_too_high` |
| `incompleteArmExtensionAngleDeg` | `101.0` | 度 (°) | 刺擊最遠點時，前手臂（肩肘腕）允許的最小角度 | `incomplete_arm_extension` |
| `overParryTorsoRatioThreshold` | `0.54` | 比例 | 手腕水平掃掠範圍佔軀幹長度的最大比例 | `over_parrying` |
| `stepShoulderProxyMultiplier` | `2.5` | 倍數 | 肩膀水平投影轉換為模擬肩寬的乘數 | 步幅計算 |
| `wideStepRatioThreshold` | `3.0` | 比例 | 步幅比值（步寬 / 模擬肩寬）的上限 | `wide_step` |
| `narrowStepRatioThreshold` | `0.9` | 比例 | 步幅比值（步寬 / 模擬肩寬）的下限 | `narrow_step` |
| `stepSustainedSeconds` | `0.30` | 秒 (s) | 步幅異常狀態必須持續的最短時間 | 步幅判定 |
| `comForwardLeanDeg` | `21.0` | 度 (°) | 軀幹前傾允許的最大角度（正值為面向對手） | `center_of_mass_in_front` |
| `comBackwardLeanDeg` | `3.0` | 度 (°) | 軀幹後仰允許的最小角度（正值為面向對手） | `center_of_mass_leaning_backward` |
| `comSustainedSeconds` | `0.30` | 秒 (s) | 軀幹傾斜異常狀態必須持續的最短時間 | 重心判定 |

---

## 相關開發指引

* **如何調整這些閾值**：請參考 [`frontend/TUNING.md`](../../frontend/TUNING.md) 了解如何透過手機實機 App 或 Mac 的 `tuning_server.py` 工具進行可視化調參。
* **如何新增啟發式規則**：請閱讀 [`docs/dev/ADDING_HEURISTIC.md`](ADDING_HEURISTIC.md) 以遵循正確的流程修改前後端規則定義、評估函數與測試案例。
* **語音提示音檔 (M4A)**：系統中每個錯誤對應的語音提示音檔已預先生成於 [`docs/cues_audio/`](cues_audio) 資料夾中。音檔採用 macOS 的 `say -v Mei-Jia`（台灣中文）語音引擎進行離線合成。
