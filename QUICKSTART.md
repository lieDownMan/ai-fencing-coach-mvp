# Quick Start Guide - AI Fencing Coach System

## 1分鐘快速開始

### 最簡單的方式（交互模式）

```bash
# 1. 進入項目目錄
cd ai-fencing-coach-mvp

# 2. 安裝依賴（首次使用）
pip install -r requirements.txt

# 3. 運行應用
python app.py --interactive
```

按照菜單提示操作即可。

---

## 5分鐘開始處理視頻

```bash
# 處理一個視頻文件並得到反饋
python app.py --video /path/to/bout.mp4 --fencer-id athlete_001 --fencer-name "John Smith"
```

這將：
1. ✅ 提取骨架信息
2. ✅ 分類擊劍動作
3. ✅ 生成統計數據
4. ✅ 創建/更新運動員檔案
5. ✅ 生成教練反饋

---

## 文件說明

| 文件 | 說明 |
|------|------|
| `app.py` | 主程序入口 |
| `requirements.txt` | 依賴列表 |
| `config.yaml` | 配置文件（可選） |
| `README.md` | 英文文檔 |
| `README_zh.md` | 中文文檔 |
| `detailedstructure.md` | 完整規格 |

---

## 常見命令

### 查看所有選項
```bash
python app.py --help
```

### 使用 GPU 加速（推薦）
```bash
python app.py --video bout.mp4 --device cuda --fencer-id athlete_001
```

### 使用更準確的模型（BiFenceNet）
```bash
python app.py --video bout.mp4 --use-bifencenet --fencer-id athlete_001
```

### 加載預訓練模型
```bash
python app.py --video bout.mp4 --model weights/fencenet/best_model.pth
```

---

## 項目結構速覽

```
ai-fencing-coach-mvp/
├── app.py                    ⭐ 主程序
├── requirements.txt          📦 依賴
├── config.yaml               ⚙️ 配置
├── README.md                 📖 英文文檔
├── README_zh.md              📖 中文文檔
├── detailedstructure.md      📘 完整規格
│
├── data/                     💾 數據文件夾
│   ├── videos/              📹 視頻存儲
│   └── fencer_profiles/     👤 運動員檔案
│
├── weights/                 🎯 模型權重
│   ├── pose_estimator/
│   ├── fencenet/
│   └── llm_models/
│
└── src/                     🔧 源代碼
    ├── pose_estimation/    [階段1] 骨架提取
    ├── preprocessing/      [階段2] 數據預處理
    ├── models/            [階段3] FenceNet 模型
    ├── tracking/          [階段4] 模式分析
    ├── llm_agent/         [階段5] AI 教練
    └── app_interface/     [階段6] 用戶界面
```

---

## 輸出文件

運行後你會得到：

1. **運動員檔案** (`data/fencer_profiles/athlete_001.json`)
   - 比賽統計
   - 歷史數據
   - 技術特點

2. **處理後的視頻** (可選)
   - 骨架可視化
   - 標籤overlay
   - 分類結果

3. **報告** (可選)
   - JSON 統計
   - PDF 分析

---

## 常見問題

**Q: 運行很慢，怎麼辦？**
A: 使用 GPU: `python app.py --video bout.mp4 --device cuda`

**Q: CUDA 不可用？**
A: 檢查 PyTorch 安裝: `python -c "import torch; print(torch.cuda.is_available())"`
或從CPU版本開始: `python app.py --video bout.mp4 --device cpu`

**Q: 模型加載失敗？**
A: 第一次運行會下載模型（較慢），請耐心等待~

**Q: 需要視頻文件嗎？**
A: 是的，需要 `.mp4`, `.avi`, `.mov` 等視頻格式

---

## 下一步

- 📖 閱讀 [README.md](README.md) 了解更多功能
- 🔧 查看 [config.yaml](config.yaml) 調整配置
- 📚 閱讀 [detailedstructure.md](detailedstructure.md) 深入了解技術細節
- 💻 瀏覽 `src/` 文件夾理解代碼結構

---

## 技術支持

遇到問題？
1. 檢查 [README.md](README.md) 的故障排除部分
2. 查看源代碼注釋
3. 檢查日誌文件 `logs/fencing_coach.log`

祝你使用愉快！ 🤺
