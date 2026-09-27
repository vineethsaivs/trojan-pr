"""SQLite on VM1: runs, cells, receipts, prs. The run page polls this every second."""
import json, os, sqlite3

PATH = os.environ.get("PLUMBLINE_DB", "/var/lib/plumbline/plumbline.db")
SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, kind TEXT, case_id TEXT, subject TEXT, status TEXT,
  verdict TEXT, reason TEXT, plan TEXT, created_at TEXT, finished_at TEXT, wall_ms INTEGER);
CREATE TABLE IF NOT EXISTS cells (run_id TEXT, check_id TEXT, commit_label TEXT, sha TEXT, family TEXT,
  target TEXT, status TEXT, metric TEXT, threshold TEXT, detail TEXT, witness TEXT, decision TEXT,
  PRIMARY KEY (run_id, check_id, commit_label));
CREATE TABLE IF NOT EXISTS receipts (run_id TEXT PRIMARY KEY, body TEXT);
CREATE TABLE IF NOT EXISTS prs (id TEXT PRIMARY KEY, kind TEXT, title TEXT, body TEXT, diff TEXT,
  base_sha TEXT, head_sha TEXT, meta TEXT);
"""


def connect():
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    c = sqlite3.connect(PATH, timeout=30, check_same_thread=False)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    c.executescript(SCHEMA)
    return c


def _j(v):
    return json.dumps(v) if isinstance(v, (dict, list)) else v


def upsert(c, table, **row):
    cols = ", ".join(row)
    c.execute(f"INSERT OR REPLACE INTO {table} ({cols}) VALUES ({', '.join('?' * len(row))})",
              [_j(v) for v in row.values()])
    c.commit()


def update_run(c, run_id, **kw):
    c.execute(f"UPDATE runs SET {', '.join(k + '=?' for k in kw)} WHERE id=?", [_j(v) for v in kw.values()] + [run_id])
    c.commit()
