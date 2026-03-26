"""
PROJECT 11 — MCP SERVER: DATA MODELS
=====================================

Pydantic models for the job search MCP server. These define the shape of
data flowing through the system:

- JobListing: one job from the JSON import (what the resources return)
- StatusUpdate: a status change log entry (what the tool writes)

CONCEPT — Why Pydantic here?
  In previous projects (P4, P5), we used Pydantic for structured LLM output.
  Here, Pydantic serves a different purpose: it validates data at the BOUNDARY
  between the JSON file and our SQLite database (import), and between SQLite
  and the MCP response (serialization). Same tool, different use case.

CONCEPT — Enum for valid statuses
  The VALID_STATUSES set defines the only status values the tool accepts.
  This is the "predefined enum" part of least-privilege design: the tool
  can't set arbitrary strings, only values from this set.
"""

from __future__ import annotations

from pydantic import BaseModel

# ── Valid status values (the tool's "enum") ──────────────────────────────
# These are the ONLY statuses the update_status tool will accept.
# "possibly_filled" is set by the linkedin-jobs skill, not by the user.
VALID_STATUSES = {"active", "applied", "interviewing", "rejected", "offer", "closed"}

# Also allow "possibly_filled" when importing from JSON, but the tool can't set it.
ALL_STATUSES = VALID_STATUSES | {"possibly_filled"}

VALID_FIT_SCORES = {"great", "good", "stretch", "discard"}


# ── Job listing (imported from JSON, served via resources) ───────────────
class JobListing(BaseModel):
    """One job listing. Maps 1:1 to a row in the `jobs` table."""

    id: str
    title: str
    company: str
    location: str = ""
    fit_score: str = ""
    status: str = "active"
    platforms: list[str] = []
    urls: dict[str, str] = {}
    description_snippet: str = ""
    first_seen: str = ""
    last_seen: str = ""
    times_seen: int = 0
    reported_as_great_fit: bool = False
    first_reported_date: str | None = None


# ── Status update log entry (written by the tool) ───────────────────────
class StatusUpdate(BaseModel):
    """A status change record. One row in the `status_updates` table."""

    id: int | None = None  # auto-increment, None before insert
    job_id: str
    old_status: str
    new_status: str
    notes: str = ""
    updated_at: str = ""
