from database import Database
import sqlite3


def test_save_action_logs_accepts_error_key_schema(tmp_path):
    db_path = tmp_path / "coach.db"
    db = Database(str(db_path))
    user_id = db.create_user("Test User", "right", 180)
    session_id = db.create_session(user_id, "Free Bouting", "annotated.mp4")

    db.save_action_logs(
        session_id,
        [
            {
                "start_frame": 0,
                "video_start_frame": 30,
                "action": "R",
            }
        ],
        [
            {
                "segment_index": 0,
                "start_frame": 30,
                "error_key": "guard_dropped",
            }
        ],
    )

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT start_frame, action_label, heuristic_warning FROM ActionLogs"
        ).fetchone()

    assert dict(row) == {
        "start_frame": 30,
        "action_label": "R",
        "heuristic_warning": "guard_dropped",
    }


def test_save_action_logs_keeps_first_warning_for_segment(tmp_path):
    db_path = tmp_path / "coach.db"
    db = Database(str(db_path))
    user_id = db.create_user("Test User", "right", 180)
    session_id = db.create_session(user_id, "Free Bouting", "annotated.mp4")

    db.save_action_logs(
        session_id,
        [{"start_frame": 0, "video_start_frame": 30, "action": "R"}],
        [
            {"segment_index": 0, "error_key": "focused_error"},
            {"segment_index": 0, "error_key": "lower_priority_error"},
        ],
    )

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT heuristic_warning FROM ActionLogs"
        ).fetchone()

    assert row[0] == "focused_error"
