"""SQLite helpers: users, settings, event log."""
import sqlite3
import threading
import time
from contextlib import contextmanager

from werkzeug.security import generate_password_hash

import config

_cache_lock = threading.Lock()
_settings_cache = {}


@contextmanager
def db():
    c = sqlite3.connect(config.DB_PATH, timeout=10)
    c.row_factory = sqlite3.Row
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init_db():
    with db() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS users(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'viewer');
            CREATE TABLE IF NOT EXISTS settings(
                key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS events(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL, lane INTEGER, kind TEXT NOT NULL, message TEXT NOT NULL);
            """
        )
        if not c.execute("SELECT 1 FROM users WHERE role='admin'").fetchone():
            c.execute(
                "INSERT INTO users(username,password_hash,role) VALUES(?,?,?)",
                ("admin", generate_password_hash("admin123"), "admin"),
            )
        for k, v in config.DEFAULT_SETTINGS.items():
            c.execute("INSERT OR IGNORE INTO settings(key,value) VALUES(?,?)", (k, str(v)))
    reload_settings()


def reload_settings():
    with db() as c:
        rows = c.execute("SELECT key,value FROM settings").fetchall()
    parsed = {}
    for r in rows:
        default = config.DEFAULT_SETTINGS.get(r["key"])
        if default is None:
            continue
        try:
            parsed[r["key"]] = type(default)(float(r["value"])) if isinstance(default, int) else float(r["value"])
        except ValueError:
            parsed[r["key"]] = default
    with _cache_lock:
        _settings_cache.clear()
        _settings_cache.update(parsed)


def get_settings():
    with _cache_lock:
        return dict(_settings_cache)


def save_settings(values):
    with db() as c:
        for k, v in values.items():
            c.execute("INSERT OR REPLACE INTO settings(key,value) VALUES(?,?)", (k, str(v)))
    reload_settings()


def log_event(lane, kind, message):
    with db() as c:
        c.execute("INSERT INTO events(ts,lane,kind,message) VALUES(?,?,?,?)",
                  (time.time(), lane, kind, message))
        c.execute("DELETE FROM events WHERE id NOT IN (SELECT id FROM events ORDER BY id DESC LIMIT 1000)")


def recent_events(limit=30):
    with db() as c:
        rows = c.execute("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]
