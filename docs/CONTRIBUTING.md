# Contributing to AI Fencing Coach System

感謝你對本項目的興趣！我們歡迎各種形式的貢獻。

## 貢獻方式

### 1. 報告 Bug

如果你發現了 bug，請提交 Issue，包含：
- ✅ 問題描述
- ✅ 重現步驟
- ✅ 預期行為 vs 實際行為
- ✅ 系統環境（Python 版本、OS 等）
- ✅ 相關日誌或錯誤信息

### 2. 功能請求

有想法改進系統？提交 Feature Request：
- ✅ 清楚的功能描述
- ✅ 使用場景
- ✅ 可能的實現方式（可選）

### 3. 代碼提交

#### 準備開發環境

```bash
# 克隆倉庫
git clone <repo>
cd ai-fencing-coach-mvp

# 創建虛擬環境
python3 -m venv venv
source venv/bin/activate

# 安裝依賴+開發工具
pip install -r requirements.txt
pip install pytest black flake8 mypy
```

#### 代碼風格

我們遵循 PEP 8：

```bash
# 格式化代碼
black src/

# 檢查風格
flake8 src/

# 類型檢查
mypy src/
```

#### 提交流程

1. **創建功能分支**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **編寫代碼和測試**
   ```bash
   # 編輯代碼
   vim src/your_module.py
   
   # 運行測試
   pytest
   ```

3. **提交更改**
   ```bash
   git add src/
   git commit -m "Add: brief description of changes"
   ```

4. **推送到分支**
   ```bash
   git push origin feature/your-feature-name
   ```

5. **提交 Pull Request**
   - 在 GitHub 上創建 PR
   - 填寫詳細描述
   - 等待審核

### 4. 改進文檔

幫助我們改進文檔：
- 修正錯誤或過時內容
- 添加示例或教程
- 改進代碼注釋
- 翻譯文檔

---

## 開發指南

### 項目結構

```
src/
├── pose_estimation/     [Phase 1] 骨架提取
├── preprocessing/       [Phase 2] 數據預處理  
├── models/             [Phase 3] 模型架構
├── tracking/           [Phase 4] 模式分析
├── llm_agent/          [Phase 5] AI 教練
└── app_interface/      [Phase 6] 用戶界面
```

### 命名規範

- **類名**: `PascalCase` (e.g., `FenceNet`, `PatternAnalyzer`)
- **函數名**: `snake_case` (e.g., `extract_skeleton`, `normalize_data`)
- **常量**: `UPPER_CASE` (e.g., `SEQUENCE_LENGTH = 28`)
- **私有方法**: 前綴 `_` (e.g., `_normalize_skeleton`)

### 代碼注釋

```python
def process_video(video_path: str) -> Dict[str, Any]:
    """
    Process a fencing video through the entire pipeline.
    
    Args:
        video_path: Path to input video file
        
    Returns:
        Dictionary containing processing results:
        - 'frames_processed': number of frames
        - 'classifications': list of predicted actions
        - 'statistics': summary statistics
        
    Raises:
        ValueError: If video file not found
        RuntimeError: If model inference fails
    """
```

### 類型提示

使用完整的類型提示：

```python
from typing import Dict, List, Optional, Tuple

def classify_actions(
    skeleton_sequence: List[Dict[str, Tuple[float, float]]],
    confidence_threshold: float = 0.5
) -> List[Tuple[int, float]]:
    """Classify fencing actions from skeleton sequence."""
    pass
```

---

## 測試指南

### 運行測試

```bash
# 運行所有測試
pytest

# 運行特定測試文件
pytest tests/test_models.py

# 運行特定測試函數
pytest tests/test_models.py::test_fencenet_forward

# 顯示覆蓋率
pytest --cov=src
```

### 編寫測試

```python
# tests/test_preprocessing.py
import pytest
from src.preprocessing import SpatialNormalizer

def test_spatial_normalizer_fit():
    """Test spatial normalizer fitting."""
    normalizer = SpatialNormalizer()
    skeleton_seq = [
        {"nose": (100, 100), "front_ankle": (100, 200)},
        {"nose": (105, 100), "front_ankle": (105, 200)},
    ]
    normalizer.fit(skeleton_seq)
    assert normalizer.scale_factor == 100.0

def test_spatial_normalizer_normalize():
    """Test skeleton normalization."""
    # ... test implementation
    pass
```

---

## 性能優化建議

對於各模塊的優化方向：

### Phase 1: Pose Estimation
- 考慮使用更輕量級的模型
- 實現 batch 推理
- GPU 加速

### Phase 3: FenceNet
- 模型量化（INT8/FP16）
- 知識蒸餾
- Mobile/edge 部署

### Phase 5: LLM Coaching
- 本地 LLM 應用
- Prompt 優化
- 緩存常用反饋

---

## 提交信息規範

```
<type>: <subject>

<body>

<footer>
```

類型包括：
- `feat`: 新功能
- `fix`: 修復 bug
- `docs`: 文檔更新
- `style`: 代碼風格
- `refactor`: 代碼重構
- `perf`: 性能優化
- `test`: 添加測試

例子：
```
feat: add sliding window inference for real-time processing

- Implement sliding window with configurable stride
- Add temporal context handling
- Improve throughput by 3x

Fixes #123
```

---

## 社區準則

- 尊重所有貢獻者
- 建設性的反饋
- 不騷擾、歧視或欺凌
- 遵守開源社區規范

---

## 聯繫方式

有疑問？
- 📧 Email: support@fencing-coach.app
- 💬 Discussion: GitHub Discussions
- 🐛 Issues: GitHub Issues

感謝你的貢獻！🙏 🤺
