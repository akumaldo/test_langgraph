"""
PROJECT 11 — MCP SERVER: INPUT VALIDATION
==========================================

Server-side validation for every input the MCP tool receives.

CONCEPT — Why validate on the server?
  In a REST API, you validate because users can send anything. In MCP,
  the "user" is an LLM — and LLMs can also send anything. The MCP schema
  tells the client what parameters exist and their types, but the server
  must still verify:
    1. The value is within allowed bounds (status enum, notes length)
    2. The referenced data exists (job_id points to a real job)
    3. Business rules are met (can't transition to same status)

  This is "trust but verify" — the schema handles the obvious cases,
  validation catches everything else.

CONCEPT — Clear error messages
  When validation fails, the error message should tell the client exactly
  what went wrong AND what the valid options are. This helps the LLM
  self-correct on the next attempt.
"""

from __future__ import annotations

from .models import VALID_FIT_SCORES, VALID_STATUSES

MAX_NOTES_LENGTH = 500


class ValidationError(Exception):
    """Raised when input validation fails. Message is returned to the client."""

    pass


def validate_status(status: str) -> str:
    """Validate and normalize a status value.

    Returns the cleaned status or raises ValidationError.
    """
    cleaned = status.strip().lower()
    if cleaned not in VALID_STATUSES:
        raise ValidationError(
            f"Invalid status: '{status}'. "
            f"Valid statuses: {', '.join(sorted(VALID_STATUSES))}"
        )
    return cleaned


def validate_notes(notes: str) -> str:
    """Validate and clean notes text.

    Returns trimmed notes or raises ValidationError if too long.
    """
    cleaned = notes.strip()
    if len(cleaned) > MAX_NOTES_LENGTH:
        raise ValidationError(
            f"Notes too long: {len(cleaned)} characters "
            f"(max {MAX_NOTES_LENGTH})."
        )
    return cleaned


def validate_fit_score(fit_score: str) -> str:
    """Validate a fit_score filter value for resource templates.

    Returns the cleaned fit_score or raises ValidationError.
    """
    cleaned = fit_score.strip().lower()
    if cleaned not in VALID_FIT_SCORES:
        raise ValidationError(
            f"Invalid fit_score: '{fit_score}'. "
            f"Valid values: {', '.join(sorted(VALID_FIT_SCORES))}"
        )
    return cleaned
