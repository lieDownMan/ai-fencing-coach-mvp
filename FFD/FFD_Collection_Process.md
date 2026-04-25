# 輕量級純視覺擊劍步法資料集 (Vision-Only FFD) 收集流程說明書

本文件詳述如何建立一個針對「擊劍步法識別」設計的輕量級、純視覺 (RGB-based) 訓練資料集。此流程參考了 Malawski 等人提出的 Fencing Footwork Dataset (FFD)  與 FenceNet 的 2D 骨架分析方法 。

## 1. 環境與硬體設定 (Hardware & Setup)

* **錄製設備**：智慧型手機或相機，建議設定為 **1080p / 60fps**。較高的幀率能更精準捕捉弓步的加速度變化 。
* **視角設定 (Piste View)**：將相機架設於劍道側邊，高度約與選手腰部平齊 。
* **距離**：相機距選手約 **3 公尺**，確保畫面涵蓋選手前進與後退的完整移動範圍 。
* **環境要求**：光線充足，背景應盡量簡潔，避免其他移動物體干擾骨架偵測 。

## 2. 動作分類清單 (Action Classes)

根據擊劍理論與 FFD 論文 ，資料集應包含以下具備不同「動態特徵」的動作：

1.  **Step Forward (SF)**: 標準前進步。
2.  **Step Backward (SB)**: 標準後退步。
3.  **Rapid Lunge (R)**: 極速弓步，強調瞬間爆發力。
4.  **Incremental Speed Lunge (IS)**: 漸速弓步，起步稍慢後段加速。
5.  **Lunge with Waiting (WW)**: 帶停頓弓步，動作中途有短暫觀察停頓。
6.  **Jumping-Sliding Lunge (JS)**: 躍步弓步，距離最長的攻擊步法。
7.  **Idle / En Garde**: 原地準備姿勢（作為模型負樣本）。

## 3. 收集協議 (Recording Protocol)

為簡化後續標記工作，採取「指令觸發式」錄製：
* **單一動作錄製**：選手聽取指令執行單一動作（如：前進步），執行完畢後回到 En Garde 並靜止 1 秒。
* **樣本數量**：建議每位選手針對每種動作重複 **20-30 次**。
* **多樣性**：邀請不同技術等級（初階與職業）的選手參與，以提升模型的泛用性 。

## 4. 資料預處理流程 (Data Preprocessing Pipeline)

### Step 1: 影像切割與標記
* 將影片切割為 **1.0 至 2.0 秒** 的短片（約 30-60 幀）。
* 使用「資料夾分類法」，將影片直接存入對應名稱的資料夾（如 `/1_advance/`）完成標記。

### Step 2: 骨架特徵提取 (Pose Extraction)
* 使用 **MediaPipe** 或 **ViTPose** 提取每幀的關鍵點 。
* **選用節點**：鼻子、雙肩、雙髖、雙膝、雙腳踝、雙腳尖及持劍手手腕（共 12 點）。

### Step 3: 座標正規化 (Normalization) - 核心關鍵
* 為消除選手在畫面中位置的差異，將每幀的所有座標減去「後腳踝」或「骨盆中心」的座標 。
* 將座標值除以選手身高的代理值（如肩膀到腳踝的距離），實現尺度縮放 (Scaling)。

### Step 4: 時間維度對齊 (Temporal Alignment)
* **固定長度**：將所有樣本統一對齊至 **30 幀 (1秒)** 或 **60 幀**。
* **處理方式**：短於標準長度者採用「末幀填充 (Last-frame padding)」，長於標準者則截斷 (Trimming)。

## 5. 資料存檔格式 (Data Export)

最終輸出的資料應轉換為 Numpy 矩陣供模型訓練：
* **特徵 (X)**: 形狀為 `(N_samples, N_frames, N_features)`，例如 `(500, 30, 24)`。
* **標籤 (Y)**: 形狀為 `(N_samples,)`，儲存 `0-6` 的分類編號。

---
**引用與參考文獻：**
*  Malawski & Kwolek, "Recognition of action dynamics in fencing using multimodal cues", 2018.
*  Zhu et al., "FenceNet: Fine-grained Footwork Recognition in Fencing", 2022.
*  Sawahata et al., "Instance Segmentation-Based Markerless Tracking of Fencing Sword Tips", 2024.
