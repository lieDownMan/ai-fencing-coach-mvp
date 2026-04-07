# AI 擊劍教練與裁判系統

## 項目概述

這是一個綜合遠程擊劍教練和裁判輔助系統，利用 **FenceNet**（基於TCN的2D骨架擊劍動作識別）和開源多模態大型語言模型（如 LLaVA-NeXT、Qwen2-VL）來充當虛擬教練。

系統監控擊劍比賽，分析擊劍動作模式，提供分層反饋（實時反饋、休息時策略、賽後總結），並追蹤運動員的長期進度。

## 核心特性

- **實時骨架檢測**: 從視頻提取2D姿勢關鍵點
- **擊劍動作分類**: 識別6類擊劍基本動作
  - **R**: 快速刺擊（Rapid lunge）
  - **IS**: 遞增速度刺擊（Incremental speed lunge）
  - **WW**: 等待中刺擊（With waiting lunge）
  - **JS**: 跳躍滑動刺擊（Jumping sliding lunge）
  - **SF**: 向前步法（Step forward）
  - **SB**: 向後步法（Step backward）

- **動作模式分析**: 計算進攻比例、防守比例、JS/SF比率等
- **虛擬教練反饋**: 
  - 實時反饋（比賽中）
  - 休息策略建議（1分鐘休息時）
  - 賽後綜合分析
- **運動員檔案**: 長期性能跟蹤和進度分析
- **交互式UI**: 實時儀表板顯示視頻、分類和教練建議

## 項目結構

```
fencing_coach_system/
├── app.py                              # 主應用入口
├── requirements.txt                    # Python依賴
├── README.md                          # 本文件
├── detailedstructure.md               # 詳細規格
├── data/
│   ├── videos/                        # 原始比賽視頻
│   └── fencer_profiles/               # 運動員檔案（JSON）
├── weights/
│   ├── pose_estimator/                # 2D姿態估計模型
│   ├── fencenet/                      # FenceNet訓練權重
│   └── llm_models/                    # 本地LLM權重
└── src/
    ├── pose_estimation/               # 第1階段：視頻到2D骨架
    ├── preprocessing/                 # 第2階段：骨架正規化和時序採樣
    ├── models/                        # 第3階段：FenceNet分類
    ├── tracking/                      # 第4階段：模式分析和進度跟蹤
    ├── llm_agent/                     # 第5階段：虛擬教練引擎
    └── app_interface/                 # 第6階段：系統編排和HCI
```

## 安裝步驟

### 1. 克隆或下載項目
```bash
cd ai-fencing-coach-mvp
```

### 2. 創建Python虛擬環境（推薦）
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. 安裝依賴
```bash
pip install -r requirements.txt
```

### 4. 驗證安裝
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"
```

## 使用指南

### 交互模式（推薦用於初次使用）
```bash
python app.py --interactive
```

這將啟動交互式菜單，讓您：
1. 處理比賽視頻
2. 配置運動員
3. 查看運動員檔案
4. 退出

### 從命令行處理視頻
```bash
python app.py --video path/to/video.mp4 --fencer-id fencer_001 --fencer-name "王明"
```

### 使用BiFenceNet（雙向分析）
```bash
python app.py --video path/to/video.mp4 --use-bifencenet --fencer-id fencer_001
```

### 指定計算設備
```bash
# 自動檢測（推薦）
python app.py --video video.mp4 --device auto

# 強制使用GPU
python app.py --video video.mp4 --device cuda

# 使用CPU
python app.py --video video.mp4 --device cpu
```

### 加載預訓練模型
```bash
python app.py --video video.mp4 --model weights/fencenet/model_best.pth
```

## 命令行選項

```
用法: app.py [選項]

選項:
  --video VIDEO              視頻文件路徑
  --fencer-id ID            運動員ID (默認: fencer_001)
  --fencer-name NAME        運動員全名
  --opponent-id ID          對手ID（可選）
  --opponent-name NAME      對手全名（可選）
  --use-bifencenet          使用BiFenceNet而不是FenceNet
  --device {auto,cuda,cpu}  計算設備 (默認: auto)
  --model PATH              預訓練模型路徑
  --interactive             交互模式
  --help                    顯示幫助信息
```

## 模塊說明

### 1. 姿勢估計 (`src/pose_estimation/`)
- 使用YOLO-Pose等模型從視頻幀提取2D骨架
- 輸出17個關鍵點，包括鼻子、肩膀、肘、手腕、臀部、膝蓋、踝關節
- **PoseEstimator**: 管理模型加載和推理

### 2. 預處理 (`src/preprocessing/`)
- **SpatialNormalizer**: 
  - 以第一幀的鼻尖位置作為原點
  - 以頭部到踝關節的垂直距離進行縮放歸一化
- **TemporalSampler**: 
  - 將所有序列重採樣為28幀
  - 支持插值（上採樣）和下採樣

### 3. FenceNet模型 (`src/models/`)
- **TCNBlock**: 因果卷積 + 膨脹卷積 + 殘差連接
- **FenceNet**: 6個TCN塊堆疊，用於6分類
- **BiFenceNet**: 雙向TCN（前向+反向），用於更好的時序建模

### 4. 追蹤和分析 (`src/tracking/`)
- **PatternAnalyzer**: 
  - 聚合FenceNet輸出
  - 計算動作頻率、進攻/防守比例、JS/SF比率
  - 檢測重複模式
- **ProfileManager**: 
  - 保存和加載運動員檔案
  - 追蹤歷史統計和進度

### 5. LLM教練引擎 (`src/llm_agent/`)
- **PromptTemplates**: 3種反饋類型的系統提示
  - 實時反饋（1-2句）
  - 休息策略（2-3句）
  - 賽後分析（結構化）
- **ModelLoader**: 加載開源MLLM（LLaVA-NeXT、Qwen2-VL等）
- **CoachEngine**: 協調反饋生成

### 6. 應用界面 (`src/app_interface/`)
- **SystemPipeline**: 協調所有處理階段
- **FencingCoachUI**: OpenCV實時儀表板
  - 顯示實時視頻
  - 動作分類日志
  - 動作頻率分布
  - 教練反饋面板

## 工作流程

### 單個比賽處理
```
視頻輸入
    ↓
[第1階段] 姿勢估計 → 2D骨架序列
    ↓
[第2階段] 預處理 → 歸一化 + 採樣到28幀
    ↓
[第3階段] FenceNet推理 → 動作分類
    ↓
[第4階段] 模式分析 → 統計指標
    ↓
[第5階段] LLM教練 → 反饋生成
    ↓
[第6階段] UI顯示 + 檔案保存
    ↓
運動員檔案更新
```

## 配置文件

### 運動員檔案格式
```json
{
  "fencer_id": "fencer_001",
  "name": "王明",
  "created_at": "2026-04-07T...",
  "bouts": [
    {
      "timestamp": "2026-04-07T...",
      "opponent_id": "fencer_002",
      "result": "win",
      "statistics": {
        "offensive_ratio": 0.65,
        "defensive_ratio": 0.15,
        "js_sf_ratio": 2.1,
        "action_frequencies": {...}
      }
    }
  ],
  "overall_stats": {
    "total_bouts": 10,
    "wins": 7,
    "losses": 3,
    "average_offensive_ratio": 0.62
  }
}
```

## 性能優化

### GPU加速
- 確保安裝了CUDA-enabled PyTorch
- 使用 `--device cuda` 選項
- 模型會自動在GPU上運行

### 模型量化
- requirements.txt 包含 `bitsandbytes`
- BiFenceNet 支持4-bit量化以節省內存

### 批處理
- SystemPipeline 支持幀批處理
- 調整 `batch_size` 以平衡速度和內存

## 故障排除

### 問題：CUDA不可用
**解決方案**: 
- 檢查 `python -c "import torch; print(torch.cuda.is_available())"`
- 重新安裝 CUDA-compatible PyTorch: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

### 問題：模型加載緩慢
**解決方案**:
- 第一次加載會下載模型權重，請耐心等待
- 後續加載會使用緩存（`~/.cache/huggingface/hub/`）

### 問題：內存不足
**解決方案**:
- 使用 CPU 進行推理（較慢但內存少）
- 使用模型量化：BiFenceNet 支持 4-bit
- 減小 batch_size

## 依賴項說明

| 包名 | 版本 | 用途 |
|------|------|------|
| torch | >=2.0.0 | 深度學習框架 |
| torchvision | >=0.15.0 | 計算機視覺工具 |
| opencv-python | >=4.8.0 | 視頻處理和UI |
| transformers | >=4.30.0 | LLM模型加載 |
| numpy | >=1.24.0 | 數值計算 |
| pandas | >=2.0.0 | 數據處理 |
| bitsandbytes | >=0.40.0 | 模型量化（可選） |

## 未來改進方向

- [ ] 實時視頻流支持（WebRTC）
- [ ] 多運動員同時追蹤
- [ ] 視頻記錄和回放功能
- [ ] Web界面（Flask/Vue.js）
- [ ] 模型微調工具
- [ ] 更多語言支持
- [ ] 詳細的統計分析儀表板

## 許可證

MIT License

## 聯繫方式

如有問題或建議，請聯繫項目維護人員。

---

**版本**: 0.1.0  
**最後更新**: 2026年4月7日
