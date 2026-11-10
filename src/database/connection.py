"""Database connection and session management for SQLite.

Enforces Write-Ahead Logging (WAL) and foreign key constraints on every connection.
Provides context-managed transactions with automatic rollback on error.
"""

from contextlib import contextmanager
import sqlite3
from typing import Iterator, Optional
from pathlib import Path

from src.core.config import config


def create_connection(db_path: Optional[Path | str] = None) -> sqlite3.Connection:
    """Create and configure a new SQLite connection with strict pragmas."""
    target_path = Path(db_path) if db_path else config.db_path
    
    # Ensure parent directory exists for file-based databases
    if str(target_path) != ":memory:":
        target_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(
        str(target_path),
        check_same_thread=False,
        timeout=10.0,
        isolation_level=None  # Explicit transaction control
    )
    
    conn.row_factory = sqlite3.Row
    
    # Enforce SQLite foreign key constraints
    conn.execute("PRAGMA foreign_keys = ON;")
    
    # Enable WAL mode for high concurrency if file-based
    if str(target_path) != ":memory:":
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        
    return conn


@contextmanager
def get_db(db_path: Optional[Path | str] = None) -> Iterator[sqlite3.Connection]:
    """Context manager yielding a transactional SQLite connection."""
    conn = create_connection(db_path)
    try:
        conn.execute("BEGIN;")
        yield conn
        conn.execute("COMMIT;")
    except Exception:
        conn.execute("ROLLBACK;")
        raise
    finally:
        conn.close()
