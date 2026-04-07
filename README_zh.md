# AI Fencing Coach MVP - 中文摘要

此文件是簡短中文導覽。主要規格與工作流程請以英文文件為準：

- [README.md](README.md)：專案入口與文件地圖。
- [mvpspec.md](mvpspec.md)：MVP 範圍、系統工作流程與 HCI 研究定位。
- [QUICKSTART.md](QUICKSTART.md)：最小安裝與執行命令。
- [CONTRIBUTING.md](CONTRIBUTING.md)：開發與貢獻規範。

## 專案定位

這個 repo 是一個桌面版 AI 擊劍教練 MVP。它使用影片輸入、姿態估計、擊劍動作分析與教練回饋，協助初學到中階擊劍選手理解距離、步法與練習模式。

目前應把它定位為「教練輔助工具」，不是正式裁判系統，也不是完整得分系統。研究價值應該來自使用者經驗：選手或教練是否能因為系統回饋而更快理解問題、調整練習或進行賽後檢討。

## 核心工作流程

```text
影片輸入
  -> 姿態估計與雙人追蹤
  -> 骨架正規化與動作特徵擷取
  -> 動作理解
     -> MVP 路徑：距離與站姿等啟發式規則
     -> 研究路徑：FenceNet/BiFenceNet 六類步法辨識
  -> 模式分析與選手檔案更新
  -> 教練回饋
  -> OpenCV 儀表板或處理後影片輸出
```

## 快速開始

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py --interactive
```

處理影片：

```bash
python app.py --video path/to/bout.mp4 --fencer-id athlete_001 --device auto
```

## 下一步研究問題

- 具體是哪一類選手或教練正在受這個問題困擾？
- 他們現在如何處理步法與距離回饋？
- 我們是否已經觀察到真實練習或賽後檢討場景？
- 這個系統的貢獻是新能力、較好的教練體驗，還是較低成本的替代方案？
- 我們能否在專案時程內接觸真實目標使用者進行測試？
