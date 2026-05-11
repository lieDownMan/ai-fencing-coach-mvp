# AI Fencing Coach MVP - 中文摘要

此文件是簡短中文導覽。主要規格與工作流程請以英文文件為準：

- [../../README.md](../../README.md)：GitHub 專案入口頁。
- [../README.md](../README.md)：文件總索引。
- [README.md](README.md)：完整開發文件總覽。
- [CURRENT_STATUS.md](CURRENT_STATUS.md)：目前可執行狀態與交接摘要。
- [mvpspec.md](mvpspec.md)：MVP 範圍、系統工作流程與 HCI 研究定位。
- [QUICKSTART.md](QUICKSTART.md)：最小安裝與執行命令。
- [CONTRIBUTING.md](CONTRIBUTING.md)：開發與貢獻規範。

## 專案定位

這個 repo 是一個桌面版 AI 擊劍教練 MVP。它使用影片輸入、姿態估計、擊劍動作分析與教練回饋，協助初學到中階擊劍選手理解距離、步法與練習模式。

目前應把它定位為「教練輔助工具」，不是正式裁判系統，也不是完整得分系統。研究價值應該來自使用者經驗：選手或教練是否能因為系統回饋而更快理解問題、調整練習或進行賽後檢討。

## 目前主要入口

- `app.py`：Gradio 片段分析 UI，適合上傳或錄製短片後做完整分析。
- `realtime_app.py`：本機 OpenCV 即時模式，適合筆電 webcam + 本機語音回饋。
- `web_realtime_app.py`：瀏覽器 webcam 串流模式，適合網頁即時顯示。

## 核心工作流程

```text
影片輸入
  -> 姿態估計
     -> 目前程式：雙人候選 + 單一 target 骨架進分類器
     -> 研究目標：更穩定的身份追蹤與左右持劍手處理
  -> 骨架正規化為 9 個關節 / 18 個 channel
  -> FenceNet/BiFenceNet 六類步法辨識
  -> heuristics 判斷與 coaching cue
  -> 教練回饋
     -> 目前程式：可選 Gemini session summary，另有本機 pyttsx3 即時語音 cue
     -> 研究目標：更穩定的左手選手支援與更完整的即時 coaching
  -> Gradio、OpenCV 視窗或瀏覽器串流
```

## 目前實作狀態

- 可用 `--pose-backend mock` 進行穩定的本機與測試流程。
- `--pose-backend ultralytics` 是預留給 YOLO pose 的真實姿態估計路徑，但目前 venv 尚未安裝 `ultralytics`。
- 目前還不是完整雙人擊劍追蹤，也不是正式裁判或得分系統。
- 如果沒有訓練好的模型權重，分類標籤只適合做 pipeline smoke test，不應解讀為可靠教練判斷。

## 快速開始

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

啟動本機即時 webcam 模式：

```bash
python realtime_app.py --source 0 --mode "Free Bouting"
```

啟動瀏覽器 webcam 串流模式：

```bash
python web_realtime_app.py
```

## 下一步研究問題

- 具體是哪一類選手或教練正在受這個問題困擾？
- 他們現在如何處理步法與距離回饋？
- 我們是否已經觀察到真實練習或賽後檢討場景？
- 這個系統的貢獻是新能力、較好的教練體驗，還是較低成本的替代方案？
- 我們能否在專案時程內接觸真實目標使用者進行測試？
