import sqlite3

from src.database import EVENT_COLUMNS, init_db, insert_event


def test_init_db_creates_events_schema(tmp_path):
    db_path = tmp_path / "events.db"

    init_db(db_path)

    with sqlite3.connect(db_path) as con:
        cols = [row[1] for row in con.execute("PRAGMA table_info(events);").fetchall()]

    for column in EVENT_COLUMNS:
        assert column in cols


def test_insert_event_persists_row(tmp_path):
    db_path = tmp_path / "events.db"
    init_db(db_path)

    insert_event(
        db_path,
        {
            "ts_start": "2026-01-01T10:00:00",
            "ts_end": "2026-01-01T10:00:03",
            "duration": 3.0,
            "classes": "barco",
            "conf_max": 0.91,
            "video_path": "video.mp4",
            "snapshot_path": "snapshot.jpg",
            "meta_path": "meta.json",
        },
    )

    with sqlite3.connect(db_path) as con:
        count = con.execute("SELECT COUNT(*) FROM events;").fetchone()[0]

    assert count == 1
