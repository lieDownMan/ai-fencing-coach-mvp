import sqlite3
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

class Database:
    def __init__(self, db_path: str = "fencing_coach.db"):
        self.db_path = db_path
        self._init_db()
        
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            
            c.execute('''
                CREATE TABLE IF NOT EXISTS Users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    handedness TEXT,
                    height_cm INTEGER
                )
            ''')
            
            c.execute('''
                CREATE TABLE IF NOT EXISTS Sessions (
                    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    date TIMESTAMP,
                    training_mode TEXT,
                    video_path TEXT,
                    llm_summary TEXT,
                    FOREIGN KEY (user_id) REFERENCES Users (id)
                )
            ''')
            
            c.execute('''
                CREATE TABLE IF NOT EXISTS ActionLogs (
                    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER,
                    start_frame INTEGER,
                    action_label TEXT,
                    heuristic_warning TEXT,
                    FOREIGN KEY (session_id) REFERENCES Sessions (session_id)
                )
            ''')
            conn.commit()
            
    def get_users(self) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            c.execute('SELECT * FROM Users')
            return [dict(row) for row in c.fetchall()]
            
    def create_user(self, name: str, handedness: str, height_cm: int) -> int:
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute('INSERT INTO Users (name, handedness, height_cm) VALUES (?, ?, ?)',
                      (name, handedness, height_cm))
            return c.lastrowid
            
    def create_session(self, user_id: int, training_mode: str, video_path: str) -> int:
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            date = datetime.datetime.now().isoformat()
            c.execute('INSERT INTO Sessions (user_id, date, training_mode, video_path) VALUES (?, ?, ?, ?)',
                      (user_id, date, training_mode, video_path))
            return c.lastrowid
            
    def update_session_summary(self, session_id: int, summary: str):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute('UPDATE Sessions SET llm_summary = ? WHERE session_id = ?', (summary, session_id))
            
    def save_action_logs(self, session_id: int, action_segments: List[Dict], posture_errors: List[Dict]):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            
            # Create a lookup for errors
            error_map = {}
            for err in posture_errors:
                # Naive matching by segment_index if available, or just frame
                if "segment_index" in err:
                    error_map[err["segment_index"]] = err["error"]
                else:
                    error_map[err["start_frame"]] = err["error"]
                    
            for idx, seg in enumerate(action_segments):
                warning = error_map.get(idx) or error_map.get(seg["start_frame"])
                c.execute('INSERT INTO ActionLogs (session_id, start_frame, action_label, heuristic_warning) VALUES (?, ?, ?, ?)',
                          (session_id, seg["video_start_frame"], seg["action"], warning))
                          
    def get_sessions(self) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            c.execute('''
                SELECT s.*, u.name as user_name 
                FROM Sessions s 
                LEFT JOIN Users u ON s.user_id = u.id 
                ORDER BY s.date DESC
            ''')
            return [dict(row) for row in c.fetchall()]
