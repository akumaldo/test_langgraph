"""
PROJECT 11 — MCP SERVER: DATABASE LAYER
========================================

SQLite database for the job search MCP server. Two responsibilities:

1. IMPORT: Read job-history.json from the Obsidian vault and load it
   into a `jobs` SQLite table. This is a one-time operation on startup
   (or on --sync).

2. QUERY: Provide functions that the MCP server calls to read/write data.
   Resources call read-only functions. The tool calls the write function.

CONCEPT — Why SQLite instead of reading JSON directly?
  Three reasons:
  a) SQL makes filtering easy (resource templates: WHERE fit_score = ?)
  b) We need a second table (status_updates) for the tool's write log
  c) It teaches a real-world pattern: MCP servers often wrap databases

CONCEPT — Security through function design
  Notice how the query functions are shaped:
  - get_all_jobs() / get_jobs_by_score() → SELECT only, no parameters that
    could be SQL-injected
  - update_job_status() → UPDATE one row + INSERT one log row, using
    parameterized queries. No way to delete or bulk-update.

  The MCP server can only do what these functions allow. That's
  least-privilege at the code level.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .models import ALL_STATUSES, JobListing, StatusUpdate

# Default path to the job history JSON in the Obsidian vault
DEFAULT_JSON_PATH = Path.home() / "Downloads" / "obsidian_brain" / "Brain's Analyst" / "linkedin-jobs" / "job-history.json"

# Default SQLite database path (next to the server code, or overridden)
DEFAULT_DB_PATH = Path(__file__).parent / "jobs.db"


def get_connection(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Create a SQLite connection with row_factory for dict-like access."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # better concurrent reads
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def create_tables(conn: sqlite3.Connection) -> None:
    """Create the jobs and status_updates tables if they don't exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS jobs (
            id                  TEXT PRIMARY KEY,
            title               TEXT NOT NULL,
            company             TEXT NOT NULL,
            location            TEXT DEFAULT '',
            fit_score           TEXT DEFAULT '',
            status              TEXT DEFAULT 'active',
            platforms           TEXT DEFAULT '[]',
            urls                TEXT DEFAULT '{}',
            description_snippet TEXT DEFAULT '',
            first_seen          TEXT DEFAULT '',
            last_seen           TEXT DEFAULT '',
            times_seen          INTEGER DEFAULT 0,
            reported_as_great_fit INTEGER DEFAULT 0,
            first_reported_date TEXT
        );

        CREATE TABLE IF NOT EXISTS status_updates (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id      TEXT NOT NULL REFERENCES jobs(id),
            old_status  TEXT,
            new_status  TEXT NOT NULL,
            notes       TEXT DEFAULT '',
            updated_at  TEXT NOT NULL
        );
    """)


def import_from_json(
    conn: sqlite3.Connection,
    json_path: Path = DEFAULT_JSON_PATH,
) -> int:
    """Import jobs from job-history.json into the jobs table.

    Uses INSERT OR REPLACE so re-importing (--sync) updates existing rows.
    Returns the number of jobs imported.
    """
    data = json.loads(json_path.read_text(encoding="utf-8"))
    jobs = data.get("jobs", [])

    for job in jobs:
        conn.execute(
            """
            INSERT OR REPLACE INTO jobs
                (id, title, company, location, fit_score, status,
                 platforms, urls, description_snippet,
                 first_seen, last_seen, times_seen,
                 reported_as_great_fit, first_reported_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job["id"],
                job["title"],
                job["company"],
                job.get("location", ""),
                job.get("fit_score", ""),
                job.get("status", "active"),
                json.dumps(job.get("platforms", [])),
                json.dumps(job.get("urls", {})),
                job.get("description_snippet", ""),
                job.get("first_seen", ""),
                job.get("last_seen", ""),
                job.get("times_seen", 0),
                1 if job.get("reported_as_great_fit", False) else 0,
                job.get("first_reported_date"),
            ),
        )

    conn.commit()
    return len(jobs)


# ── READ functions (used by resources — SELECT only) ─────────────────────

def get_all_jobs(conn: sqlite3.Connection) -> list[JobListing]:
    """Return all jobs. Used by the jobs://listings resource."""
    rows = conn.execute("SELECT * FROM jobs ORDER BY last_seen DESC").fetchall()
    return [_row_to_job(row) for row in rows]


def get_jobs_by_score(conn: sqlite3.Connection, fit_score: str) -> list[JobListing]:
    """Return jobs filtered by fit_score. Used by jobs://listings/{fit_score}."""
    rows = conn.execute(
        "SELECT * FROM jobs WHERE fit_score = ? ORDER BY last_seen DESC",
        (fit_score,),
    ).fetchall()
    return [_row_to_job(row) for row in rows]


def get_job_by_id(conn: sqlite3.Connection, job_id: str) -> JobListing | None:
    """Return a single job by ID, or None if not found."""
    row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return _row_to_job(row) if row else None


# ── WRITE function (used by the tool — scoped to one job) ────────────────

def update_job_status(
    conn: sqlite3.Connection,
    job_id: str,
    new_status: str,
    notes: str = "",
) -> StatusUpdate:
    """Update a job's status and log the change.

    This is the ONLY write path in the entire server:
    1. Read current status
    2. UPDATE jobs SET status = ? WHERE id = ?
    3. INSERT INTO status_updates (log the change)

    Returns the StatusUpdate record.
    """
    # Read current status
    row = conn.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()
    old_status = row["status"] if row else "unknown"

    # Update the job
    conn.execute(
        "UPDATE jobs SET status = ? WHERE id = ?",
        (new_status, job_id),
    )

    # Log the change
    now = datetime.now(timezone.utc).isoformat()
    cursor = conn.execute(
        """
        INSERT INTO status_updates (job_id, old_status, new_status, notes, updated_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (job_id, old_status, new_status, notes, now),
    )
    conn.commit()

    return StatusUpdate(
        id=cursor.lastrowid,
        job_id=job_id,
        old_status=old_status,
        new_status=new_status,
        notes=notes,
        updated_at=now,
    )


# ── Helper ───────────────────────────────────────────────────────────────

def _row_to_job(row: sqlite3.Row) -> JobListing:
    """Convert a sqlite3.Row to a JobListing model."""
    return JobListing(
        id=row["id"],
        title=row["title"],
        company=row["company"],
        location=row["location"],
        fit_score=row["fit_score"],
        status=row["status"],
        platforms=json.loads(row["platforms"]),
        urls=json.loads(row["urls"]),
        description_snippet=row["description_snippet"],
        first_seen=row["first_seen"],
        last_seen=row["last_seen"],
        times_seen=row["times_seen"],
        reported_as_great_fit=bool(row["reported_as_great_fit"]),
        first_reported_date=row["first_reported_date"],
    )


def init_db(
    db_path: Path = DEFAULT_DB_PATH,
    json_path: Path = DEFAULT_JSON_PATH,
) -> sqlite3.Connection:
    """Full initialization: create tables + import JSON. Returns the connection."""
    conn = get_connection(db_path)
    create_tables(conn)
    count = import_from_json(conn, json_path)
    print(f"[db] Imported {count} jobs from {json_path.name}")
    return conn
