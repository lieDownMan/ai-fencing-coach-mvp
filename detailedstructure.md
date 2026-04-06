fencing_referee_system/
├── data/                      # 存放原始比賽影片與萃取出的 JSON/CSV 骨架資料
├── weights/                   # 存放各個模型的預訓練權重
│   ├── pose_estimator/        # 存放 2D Pose 模型權重 (如 YOLO-Pose)
│   └── fencenet/              # 存放訓練好的 FenceNet 或 BiFenceNet 權重
├── src/
│   ├── pose_estimation/       # 第一階段：影像轉骨架
│   │   ├── __init__.py
│   │   └── pose_extractor.py  # 呼叫現成的 2D 人體姿勢估計模型 [cite: 79]
│   │
│   ├── preprocessing/         # 第二階段：骨架資料正規化
│   │   ├── __init__.py
│   │   ├── normalizer.py      # 實作論文中基於鼻子與腳踝的位移與縮放正規化 [cite: 151, 152]
│   │   └── sampler.py         # 實作將序列採樣至 28 個連續影格的邏輯 [cite: 141, 143]
│   │
│   ├── models/                # 第三階段：腳步動作分類模型
│   │   ├── __init__.py
│   │   ├── tcn_blocks.py      # 實作因果卷積 (Causal) 與擴張卷積 (Dilated) 的底層 Block [cite: 160, 161, 167]
│   │   ├── fencenet.py        # 實作包含 6 個 TCN blocks 與 Dense layers 的主模型 [cite: 159]
│   │   └── bifencenet.py      # 實作結合 causal 與 anti-causal 網路的雙向版本 
│   │
│   └── referee_app/           # 第四階段：系統整合與 HCI 介面
│       ├── __init__.py
│       ├── system_pipeline.py # 串接 Extract -> Preprocess -> Predict 的端到端腳本
│       └── main_ui.py         # 呈現預測結果 (例如：Step Forward, Rapid Lunge 等) 給裁判的介面 [cite: 126, 130]
│
├── train_fencenet.py          # 模型訓練腳本
├── requirements.txt           
└── README.md