epee-ai-coach/
├── api/                       # 遠端教練系統的核心 (Feature 1)
│   ├── main.py                # FastAPI 伺服器入口
│   ├── routes.py              # 處理影像串流與數據請求
│   └── websocket.py           # 負責即時推送資料給遠端前端
│
├── src/
│   ├── vision/                # 負責看 (YOLO, Pose)
│   ├── logic/                 # 負責算
│   │   ├── epee_distance.py   # 銳劍特化的距離與角度計算 (Feature 2)
│   │   └── bout_analyzer.py   # 統計回合內的數據 (Feature 3)
│   │
│   ├── services/
│   │   ├── bout_session.py    # 管理比賽的開始、暫停與休息狀態 (Feature 3)
│   │   └── llm_advisor.py     # 整合 LLM 產出賽後文字總結 (Feature 4)
│   │
│   └── db/                    # 負責記憶 (Feature 5)
│       ├── models.py          # 定義 Fencer, Bout, ActionLog 的資料表結構
│       └── crud.py            # 處理資料庫的讀寫操作
│
├── ui/                        # 給教練或選手看的介面
│   ├── dashboard/             # React/Vue 或 Streamlit 寫的網頁介面
│   └── utils_overlay.py       # 畫面上的 AR 線條疊加
│
└── tests/