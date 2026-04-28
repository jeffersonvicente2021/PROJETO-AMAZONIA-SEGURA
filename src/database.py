import json
import sqlite3
from pathlib import Path

from .config import now_ts


EVENT_COLUMNS = (
    "id",
    "ts_start",
    "ts_end",
    "duration",
    "classes",
    "conf_max",
    "video_path",
    "snapshot_path",
    "meta_path",
)


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_start TEXT,
                ts_end TEXT,
                duration REAL,
                classes TEXT,
                conf_max REAL,
                video_path TEXT,
                snapshot_path TEXT,
                meta_path TEXT
            )
            """
        )
        con.commit()


def insert_event(db_path: Path, row: dict) -> None:
    with sqlite3.connect(db_path) as con:
        con.execute(
            """
            INSERT INTO events (
                ts_start, ts_end, duration, classes, conf_max,
                video_path, snapshot_path, meta_path
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["ts_start"],
                row["ts_end"],
                row["duration"],
                row["classes"],
                row["conf_max"],
                row["video_path"],
                row["snapshot_path"],
                row["meta_path"],
            ),
        )
        con.commit()


def save_meta(meta_dir: Path, meta: dict) -> str:
    meta_dir.mkdir(parents=True, exist_ok=True)
    path = meta_dir / f"meta_{now_ts()}.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)
    return str(path)
