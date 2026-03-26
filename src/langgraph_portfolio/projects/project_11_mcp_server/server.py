"""
PROJECT 11 — MCP SERVER: FASTMCP SERVER
========================================

The MCP server that exposes job search data through three primitives:

1. RESOURCES (read-only data the client can pull):
   - jobs://listings          → all jobs
   - jobs://listings/{fit_score} → filtered by fit score (resource template)

2. TOOL (write action the client can invoke):
   - update_status            → change one job's status

3. PROMPT (reusable template the client can use):
   - summarize_application    → structured application brief for a job

CONCEPT — FastMCP and the decorator pattern
  FastMCP works like Flask: you create a server instance and decorate
  functions to register them as capabilities. The MCP SDK handles all
  the protocol plumbing:
  - stdio transport (stdin/stdout, no HTTP)
  - JSON-RPC message format
  - Capability discovery (the client asks "what can you do?" and gets
    a list of resources, tools, and prompts)

  Compare this to LangGraph's @tool decorator (P2) or CrewAI's BaseTool (P6).
  Same idea — declare what a function does, let the framework handle the rest.

CONCEPT — Resource vs Resource Template
  A plain resource has a fixed URI: jobs://listings always returns all jobs.
  A resource TEMPLATE has a parameterized URI: jobs://listings/{fit_score}
  is like a REST path parameter (/jobs/:score). The client fills in the
  parameter, and the server receives it as a function argument.

  This is similar to LangGraph's conditional routing (P1) — the same
  endpoint behaves differently based on input — but at the protocol level.

CONCEPT — Least-privilege in practice
  Look at what each primitive CAN and CANNOT do:
  - Resources: SELECT only. Cannot write to the database.
  - Tool: UPDATE one row + INSERT one log row. Cannot delete or bulk-update.
  - Prompt: No database access at all. Pure text template.

  This is enforced by CODE STRUCTURE, not configuration. The resource
  handlers don't even have access to write functions. The tool handler
  calls exactly one write function with exactly one job_id.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from . import db
from .models import VALID_FIT_SCORES, VALID_STATUSES
from .validation import ValidationError, validate_fit_score, validate_notes, validate_status

# ── Server instance ──────────────────────────────────────────────────────
# The name appears in Claude Code's MCP server list.
server = FastMCP("job-search")

# ── Database connection (initialized in create_server) ───────────────────
_conn: sqlite3.Connection | None = None


def get_conn() -> sqlite3.Connection:
    """Get the database connection. Raises if not initialized."""
    if _conn is None:
        raise RuntimeError("Database not initialized. Call create_server() first.")
    return _conn


# ── RESOURCE: jobs://listings ────────────────────────────────────────────
# Plain resource — fixed URI, returns all jobs.
# Direction: Server → Client (client pulls data)
# Analogy: GET /jobs
# Security: SELECT only. No writes.

@server.resource("jobs://listings")
async def get_listings() -> str:
    """All job listings from the job search history.

    Returns a JSON array of job objects with fields:
    id, title, company, location, fit_score, status, platforms, urls,
    description_snippet, first_seen, last_seen, times_seen.
    """
    jobs = db.get_all_jobs(get_conn())
    return json.dumps([job.model_dump() for job in jobs], ensure_ascii=False, indent=2)


# ── RESOURCE TEMPLATE: jobs://listings/{fit_score} ───────────────────────
# Parameterized resource — the {fit_score} part is like a REST path param.
# The client fills in "great", "good", "stretch", or "discard".
# Direction: Server → Client
# Analogy: GET /jobs?fit_score=great
# Security: SELECT with WHERE clause. No writes.

@server.resource("jobs://listings/{fit_score}")
async def get_listings_by_score(fit_score: str) -> str:
    """Job listings filtered by fit score.

    Valid fit_score values: great, good, stretch, discard.
    Returns a JSON array of matching job objects.
    """
    try:
        cleaned = validate_fit_score(fit_score)
    except ValidationError as e:
        return json.dumps({"error": str(e)})

    jobs = db.get_jobs_by_score(get_conn(), cleaned)
    return json.dumps([job.model_dump() for job in jobs], ensure_ascii=False, indent=2)


# ── TOOL: update_status ─────────────────────────────────────────────────
# Write action — the client asks the server to change a job's status.
# Direction: Client → Server (client tells server to do something)
# Analogy: POST /jobs/:id/status
# Security: UPDATE one row + INSERT one log row. No deletes, no bulk ops.

@server.tool()
async def update_status(job_id: str, status: str, notes: str = "") -> str:
    """Update a job's application status.

    Changes the status of ONE job and logs the transition.
    Valid statuses: active, applied, interviewing, rejected, offer, closed.

    Args:
        job_id: The job identifier (e.g., "deloitte-gerente-auditoria-interna-sp")
        status: New status value
        notes: Optional notes about this status change (max 500 chars)
    """
    conn = get_conn()

    # Validate job exists
    job = db.get_job_by_id(conn, job_id)
    if job is None:
        return f"Error: Job '{job_id}' not found."

    # Validate status
    try:
        cleaned_status = validate_status(status)
    except ValidationError as e:
        return f"Error: {e}"

    # Validate notes
    try:
        cleaned_notes = validate_notes(notes)
    except ValidationError as e:
        return f"Error: {e}"

    # Perform the update
    update = db.update_job_status(conn, job_id, cleaned_status, cleaned_notes)

    return (
        f"Updated '{job.title}' at {job.company}: "
        f"{update.old_status} → {update.new_status}"
        + (f" (notes: {update.notes})" if update.notes else "")
    )


# ── PROMPT: summarize_application ────────────────────────────────────────
# Reusable prompt template — the server provides structured instructions
# that the client's LLM can use to generate an application brief.
# Direction: Server → Client (server offers a template)
# Analogy: a stored procedure that returns instructions, not data
# Security: Reads one job's data to build the prompt. No writes.

@server.prompt()
async def summarize_application(job_id: str) -> str:
    """Generate an application summary brief for a specific job.

    Produces a structured analysis in Portuguese: job details, why it fits
    the CV profile, key strengths to emphasize, and positioning strategy.

    Args:
        job_id: The job identifier to analyze
    """
    conn = get_conn()
    job = db.get_job_by_id(conn, job_id)

    if job is None:
        return f"Erro: vaga '{job_id}' não encontrada no banco de dados."

    platforms_str = ", ".join(job.platforms) if job.platforms else "não especificada"
    urls_str = "\n".join(f"  - {platform}: {url}" for platform, url in job.urls.items())

    return f"""Analise a vaga abaixo e gere um resumo estruturado para candidatura.

## Dados da Vaga

- **Cargo**: {job.title}
- **Empresa**: {job.company}
- **Local**: {job.location}
- **Score de compatibilidade**: {job.fit_score}
- **Status atual**: {job.status}
- **Plataformas**: {platforms_str}
- **Links**:
{urls_str}
- **Primeira vez vista**: {job.first_seen}
- **Última vez vista**: {job.last_seen}
- **Vezes encontrada**: {job.times_seen}
- **Descrição**: {job.description_snippet}

## O que gerar

Responda em **português brasileiro** com as seguintes seções:

### 1. Resumo da Vaga
Descreva em 2-3 frases o que a empresa busca nesta posição.

### 2. Por Que Combina
Liste os pontos de encaixe entre o perfil do candidato (Gerente Sênior de Auditoria, 13 anos BDO, IFRS/CPC, CRC ativo, CNAI/QTG/CVM, inglês avançado, gestão de até 60 pessoas) e os requisitos da vaga.

### 3. Pontos Fortes a Enfatizar
Liste 3-5 competências ou experiências específicas que devem ser destacadas na candidatura para esta vaga.

### 4. Estratégia de Posicionamento
Sugira como posicionar a transição de auditoria externa para {job.title.lower()}, conectando a experiência em auditoria com o que a vaga pede.

### 5. Pontos de Atenção
Identifique possíveis gaps ou requisitos da vaga que o perfil não cobre diretamente, e sugira como mitigá-los.
"""


# ── Server lifecycle ─────────────────────────────────────────────────────

def create_server(
    db_path: Path = db.DEFAULT_DB_PATH,
    json_path: Path = db.DEFAULT_JSON_PATH,
) -> FastMCP:
    """Initialize the database and return the configured server.

    Call this before running the server. It:
    1. Creates SQLite tables (if they don't exist)
    2. Imports jobs from JSON (INSERT OR REPLACE)
    3. Returns the FastMCP server ready to run
    """
    global _conn
    _conn = db.init_db(db_path, json_path)
    return server
