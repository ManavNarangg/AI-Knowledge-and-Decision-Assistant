import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .config import settings

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(settings.db_path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def get_conn():
    """One short-lived connection per unit of work; commits on success."""
    conn = _connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    with get_conn() as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        _migrate(conn)


def _migrate(conn: sqlite3.Connection) -> None:
    """Additive migrations for databases created by earlier versions."""
    columns = {r["name"] for r in conn.execute("PRAGMA table_info(documents)")}
    if "ocr_pages" not in columns:
        conn.execute("ALTER TABLE documents ADD COLUMN ocr_pages INTEGER NOT NULL DEFAULT 0")
