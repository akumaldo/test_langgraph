# Project 11: Job Search MCP Server — Design Spec

## Overview

An MCP (Model Context Protocol) server that exposes real job search data (from the linkedin-jobs skill's `job-history.json`) through the three MCP primitives: resources, tools, and prompts. The server runs locally via stdio, communicating with Claude Code through JSON-RPC messages over stdin/stdout.

This project teaches the **protocol layer** — how AI clients connect to external services. MCP is to AI what REST is to web apps: the standard interface.

## Data Source

Real job search data from the Obsidian vault:
- **Source file**: `/Users/brunolunardi/Downloads/obsidian_brain/Brain's Analyst/linkedin-jobs/job-history.json`
- **Current state**: 114 jobs, with fields: id, title, company, platforms, urls, location, fit_score, status, first_seen, last_seen, times_seen, description_snippet, full_description_available, reported_as_great_fit, first_reported_date
- **Import strategy**: JSON is imported into SQLite on startup. Re-import via `--sync` flag when new job searches are run.

## Architecture

```
job-history.json (Obsidian vault)
        |
        v  (import on startup / --sync)
   jobs.db (SQLite)
   +-- jobs table (listings)
   +-- status_updates table (status change log)
        |
        v
   FastMCP server (stdio transport)
   +-- Resource: jobs://listings
   +-- Resource: jobs://listings/{fit_score}
   +-- Tool: update_status
   +-- Prompt: summarize_application
```

**Transport**: stdio (default for Claude Code integration). No HTTP, no ports, no networking. Claude Code spawns the server as a subprocess and communicates via stdin/stdout using JSON-RPC messages.

## Data Model

### `jobs` table (imported from JSON)

| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT PK | Hash-id from JSON (e.g., `deloitte-gerente-auditoria-interna-sp`) |
| `title` | TEXT NOT NULL | Job title |
| `company` | TEXT NOT NULL | Company name |
| `location` | TEXT | Location (Sao Paulo, Remoto, etc.) |
| `fit_score` | TEXT | `great`, `good`, `stretch`, `discard` |
| `status` | TEXT | `active`, `applied`, `interviewing`, `rejected`, `offer`, `closed`, `possibly_filled` |
| `platforms` | TEXT | JSON array of platform names |
| `urls` | TEXT | JSON object mapping platform -> URL |
| `description_snippet` | TEXT | First ~200 chars of description |
| `first_seen` | TEXT | ISO date when first discovered |
| `last_seen` | TEXT | ISO date when last seen in search |
| `times_seen` | INTEGER | How many searches found this job |
| `reported_as_great_fit` | INTEGER | Boolean: was it reported as great fit |
| `first_reported_date` | TEXT | ISO date when first reported |

### `status_updates` table (write log)

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PK AUTOINCREMENT | Auto-incrementing ID |
| `job_id` | TEXT NOT NULL | FK -> jobs.id |
| `old_status` | TEXT | Status before update |
| `new_status` | TEXT NOT NULL | Status after update |
| `notes` | TEXT | Optional notes (max 500 chars) |
| `updated_at` | TEXT NOT NULL | ISO timestamp of update |

## MCP Primitives

### 1. Resource: `jobs://listings` (read-only)

Returns all job listings as JSON. Read-only — no writes, no access to `status_updates`.

```python
@server.resource("jobs://listings")
async def get_listings() -> str:
    """All job listings as JSON array."""
```

Returns: JSON array of job objects with all fields from the `jobs` table.

### 2. Resource Template: `jobs://listings/{fit_score}` (read-only, parameterized)

Returns job listings filtered by fit score. Teaches MCP resource templates (parameterized URIs).

```python
@server.resource("jobs://listings/{fit_score}")
async def get_listings_by_score(fit_score: str) -> str:
    """Job listings filtered by fit score (great/good/stretch/discard)."""
```

**Validation**: `fit_score` must be one of `great`, `good`, `stretch`, `discard`. Returns error message for invalid values.

### 3. Tool: `update_status` (write action)

Updates a single job's status. Narrowly-scoped — can only update one job at a time to a predefined status value. No deletes, no bulk operations.

```python
@server.tool()
async def update_status(job_id: str, status: str, notes: str = "") -> str:
    """Update a job's application status. One job at a time, predefined statuses only."""
```

**Valid statuses**: `active`, `applied`, `interviewing`, `rejected`, `offer`, `closed`

**Server-side validation**:
- `job_id`: must exist in the `jobs` table
- `status`: must be one of the valid status values
- `notes`: max 500 characters, whitespace-trimmed
- Logs the change in `status_updates` table (old_status -> new_status with timestamp)

**What it does**:
1. Validates all inputs
2. Reads current status from `jobs` table
3. Updates `jobs.status` to new value
4. Inserts a row into `status_updates` with old_status, new_status, notes, timestamp
5. Returns confirmation message with the change details

### 4. Prompt: `summarize_application` (reusable template)

Generates a structured application brief for a specific job. No direct database access — the server loads job data and injects it into the prompt template.

```python
@server.prompt()
async def summarize_application(job_id: str) -> list[Message]:
    """Application brief: job details, fit analysis, positioning strategy. In Portuguese."""
```

**Output structure** (injected into the prompt as context):
- Job details (title, company, location, platforms, dates)
- Why this job fits the CV profile (based on fit_score and description)
- Key strengths to emphasize
- Suggested positioning approach for audit-to-controller transition

**Validation**: `job_id` must exist in the `jobs` table.

## Security Boundaries

| Primitive | Can do | Cannot do |
|-----------|--------|-----------|
| Resource `jobs://listings` | `SELECT * FROM jobs` | No writes, no `status_updates` access |
| Resource `jobs://listings/{fit_score}` | `SELECT * FROM jobs WHERE fit_score = ?` | No writes, no `status_updates` access |
| Tool `update_status` | `UPDATE jobs SET status = ? WHERE id = ?` + `INSERT INTO status_updates` for ONE job | No deletes, no bulk updates, no arbitrary SQL |
| Prompt `summarize_application` | Read one job's data to build prompt | No database writes |

## File Structure

```
src/langgraph_portfolio/projects/project_11_mcp_server/
    __init__.py
    server.py          -- FastMCP server (resource + tool + prompt decorators)
    db.py              -- SQLite setup, JSON import, query functions
    models.py          -- Pydantic models (JobListing, StatusUpdate, valid statuses)
    validation.py      -- Input validation helpers with clear error messages

project_11_mcp_server/
    __init__.py
    main.py            -- thin wrapper entry point (--sync flag for re-import)
```

## New Concepts Taught

| Concept | How it shows up |
|---------|----------------|
| MCP protocol (stdio + JSON-RPC) | FastMCP server, spawned as subprocess by Claude Code |
| MCP resources (read-only) | `jobs://listings` — client pulls data |
| MCP resource templates | `jobs://listings/{fit_score}` — parameterized URI |
| MCP tools (write actions) | `update_status` — narrowly-scoped mutation |
| MCP prompts (templates) | `summarize_application` — reusable instructions |
| Least-privilege boundaries | Tool can only update status for one job, predefined enum |
| Server-side input validation | job_id exists, status in enum, notes length check |

## Integration

After building, register with Claude Code:

```bash
claude mcp add job-search -- python project_11_mcp_server/main.py
```

Then tools/resources/prompts are available natively in Claude Code conversations.

## Dependencies

- `mcp` (FastMCP Python SDK) — handles stdio transport and JSON-RPC protocol
- `pydantic` — already in the project
- `sqlite3` — Python standard library
