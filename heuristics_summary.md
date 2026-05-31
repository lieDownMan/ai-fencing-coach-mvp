# 擊劍姿勢偵測 (Heuristics Engine) 判斷因子總覽

這份文件整理了目前 `HeuristicsEngine` 中所有支援的動作檢測項目、它們的判斷邏輯（判斷因子），以及相關的門檻參數。所有的計算都是基於 2D 骨架節點 (Skeleton Landmarks) 在正規化座標系（`[0, 1]`）下進行的。

## 1. 預備與步伐姿勢 (Stance & Footwork)

| 錯誤名稱 | UI 標籤 | 判斷因子 (Factor) | 觸發條件 (Condition) | 參數變數 (常數) |
| :--- | :--- | :--- | :--- | :--- |
| **stance_too_high** | 預備姿勢沒蹲好 | **後腳膝蓋角度** (`calcAngle(hip, knee, ankle)`) | 當後腳膝蓋角度大於門檻，代表站得太直、沒有確實蹲低。 | `kStanceTooHighAngleDeg` = 160.0° |
| **bounce_excessive** | 步伐上下浮動 | **骨盆的垂直 (Y) 軸變異係數** (Standard Deviation / Mean) | 骨盆在一段時間內上下震動的幅度與平均高度的比例超過門檻。 | `kBounceRatioThreshold` = 0.33 |
| **wide_step** | 步伐太大 | **步距與肩膀寬度的比例** (`stepWidth / shoulderWidthProxy`) | 兩腳腳踝的水平距離超過肩膀寬度的 N 倍。 | `kWideStepRatioThreshold` = 3.0 |
| **narrow_step** | 步伐太小 | **步距與肩膀寬度的比例** (`stepWidth / shoulderWidthProxy`) | 兩腳腳踝的水平距離小於肩膀寬度的 N 倍。 | `kNarrowStepRatioThreshold` = 1.2 |

## 2. 重心平衡 (Center of Mass)

> [!NOTE]
> 重心的判斷結合了「脊椎」與「肩膀連線」兩個因子的邏輯，只要任一條件觸發，即判定為重心錯誤。

| 錯誤名稱 | UI 標籤 | 判斷因子 (Factor) | 觸發條件 (Condition) | 參數變數 (常數) |
| :--- | :--- | :--- | :--- | :--- |
| **center_of_mass_in_front** | 重心向前 | **脊椎前傾角** (Neck to Pelvis) 或 **肩膀前傾角** | 1. 脊椎（頸部相對於骨盆）向前傾斜超過門檻。<br>2. 或是前肩膀低於後肩膀的連線角度超過門檻。 | `kSpineForwardTiltThresholdDeg` = 15.0°<br>`kShoulderForwardTiltThresholdDeg` = 15.0° |
| **center_of_mass_leaning_backward** | 重心向後 | **脊椎後仰角** (Neck to Pelvis) 或 **肩膀後仰角** | 1. 脊椎（頸部相對於骨盆）向後仰超過門檻。<br>2. 或是前肩膀高於後肩膀的連線角度超過門檻。 | `kSpineBackwardTiltThresholdDeg` = 10.0°<br>`kShoulderBackwardTiltThresholdDeg` = 15.0° |

## 3. 手臂與持劍動作 (Arm & Guard)

| 錯誤名稱 | UI 標籤 | 判斷因子 (Factor) | 觸發條件 (Condition) | 參數變數 (常數) |
| :--- | :--- | :--- | :--- | :--- |
| **guard_dropped** | 持劍手掉落 | **持劍手腕與手肘的相對高度** (Wrist Y vs Elbow Y) | 持劍手腕的垂直高度低於手肘，且持續累積超過 N 個影格。 | `kGuardDroppedThresholdFrames` = 5<br>`kGuardDroppedFreeBoutingThresholdFrames` = 10 |
| **hand_too_high** | 手抬太高 | **前手臂與水平線夾角** (Angle of Elbow to Wrist) | 前手臂太靠近垂直方向（手腕高於手肘，且與水平線夾角大於門檻）。 | `kHandTooHighMinAngleDeg` = 60.0° |
| **over_parrying** | 防守動作太大 | **持劍手腕的 2D 揮動距離** (Max 2D Euclidean Sweep) | 手腕在觀察視窗內移動的最大距離，大於肩膀寬度的 N 倍。 | `kOverParryRatioThreshold` = 3.0 |

## 4. 攻擊動作 (Offensive / Lunge)

> [!IMPORTANT]
> 這些動作主要在特定狀態（如 Target Practice 的攻擊動作、或偵測到 Lunge 時）觸發。

| 錯誤名稱 | UI 標籤 | 判斷因子 (Factor) | 觸發條件 (Condition) | 參數變數 (常數) |
| :--- | :--- | :--- | :--- | :--- |
| **lunge_overextension** | 長刺過度前傾 (超伸) | **前腳膝蓋的最小角度** (`calcAngle(hip, knee, ankle)`) | 整個動作視窗中，前腳膝蓋的最彎角度小於門檻（膝蓋過度超出腳踝）。 | `kLungeKneeMinAngleDeg` = 120.0° |
| **incomplete_arm_extension** | 手沒有伸直 | **前手臂伸直角度** (`calcAngle(shoulder, elbow, wrist)`) | 進行攻擊時，前手臂的關節角度小於門檻，代表手臂呈彎曲狀態。 | `kIncompleteArmExtensionAngleDeg` = 155.0° |
| **foot_before_hand** | 手腳順序錯誤 | **手腕位移與腳踝位移的比較** | 攻擊發起時，腳踝移動的距離超過門檻，但手腕移動的距離卻沒有顯著增加。 | `kFootBeforeHandMinDisplacementPx` = 0.01 |
