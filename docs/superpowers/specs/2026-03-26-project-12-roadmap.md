# Project 12: Investment Committee — Expansion Roadmap

## Overview

This document captures the planned expansions for Project 12 beyond the initial deep-dive engine. Each phase builds on the architecture established in the core deep-dive spec (`2026-03-26-project-12-investment-committee-design.md`) and reuses existing components.

**Build order:** Deep-Dive (initial) → Screening & Comparison → Monitoring & Alerts

---

## Phase 2 — Screening & Comparison

**Goal:** "Find me undervalued mid-caps in healthcare" — screen by financial criteria, compare candidates side-by-side, rank with reasoning.

### Design

- New CrewAI crew that uses FMP screener endpoints (`search-company-screener` with financial filters: sector, market cap range, P/E range, margin thresholds, etc.)
- Runs a lightweight version of the deep-dive (data gathering + valuation only) on each candidate that passes the screen
- Produces a ranked comparison table with per-company reasoning
- Reuses existing `phases/gathering.py` and `phases/valuation.py` — just orchestrated by a different LangGraph graph

### New Files

```
src/langgraph_portfolio/projects/project_12_analyst/
├── phases/
│   └── screening.py         # CrewAI crew for FMP screener + lightweight analysis
├── orchestrator_screen.py   # LangGraph graph for screening workflow
```

### New MCP Additions

- Tool: `screen_companies` — accepts filter parameters (sector, market_cap_min/max, pe_max, etc.), returns ranked list
- Resource: `screens://list` — list of saved screen results
- Resource: `screens://{screen_id}` — a specific screen result with comparison table

### New CLI Commands

- `screen {criteria}` — e.g., `screen healthcare market_cap>10B pe<20`
- `compare {TICKER1} {TICKER2} {TICKER3}` — head-to-head comparison using full deep-dive data

### What It Reuses

- `phases/gathering.py` — same data gathering crew, called per candidate
- `phases/valuation.py` — same valuation sub-graph for each candidate
- `tools/fmp.py` — same FMP wrappers, plus new screener endpoint wrapper
- `llm.py` — same LLM factory
- `server.py` — new tools/resources added to existing FastMCP server
- `main.py` — new commands in existing REPL

---

## Phase 3 — Monitoring & Alerts

**Goal:** "Watch my portfolio and tell me when something changes" — track holdings, flag material events, deliver periodic briefings.

### Design

- Watchlist stored locally as JSON file, exposed as MCP resource
- Monitoring check (manual trigger or scheduled) that scans each watchlist ticker for:
  - Earnings surprises (actual vs estimate from FMP `earnings-company` endpoint)
  - Insider trade spikes (`insider-trade-statistics` threshold breach)
  - Analyst rating changes (`grades` — new entries since last check)
  - Price movements beyond user-defined threshold (`quote` vs stored baseline)
  - New SEC filings (`latest-filings` since last check)
- Produces a briefing summary: what changed, why it matters, whether to investigate further
- Adds a new LangGraph graph (separate from the deep-dive graph) that orchestrates the monitoring pipeline
- Can trigger a full deep-dive re-run for any ticker with material changes

### New Files

```
src/langgraph_portfolio/projects/project_12_analyst/
├── phases/
│   └── monitoring.py        # Event detection + briefing generation
├── orchestrator_monitor.py  # LangGraph graph for monitoring workflow
├── watchlist.py             # Watchlist CRUD (JSON file storage)
├── reports/
│   └── briefings/           # Generated briefing reports
```

### New MCP Additions

- Tool: `add_to_watchlist` — add ticker with optional alert thresholds
- Tool: `remove_from_watchlist` — remove ticker
- Tool: `run_briefing` — trigger an on-demand portfolio scan
- Resource: `watchlist://` — current watchlist with last-checked timestamps
- Resource: `briefings://{date}` — a specific briefing report

### New CLI Commands

- `watch {TICKER}` — add to watchlist
- `unwatch {TICKER}` — remove from watchlist
- `briefing` — run on-demand portfolio scan
- `watchlist` — show current watchlist with status

### What It Reuses

- `tools/fmp.py` — same FMP wrappers for quotes, earnings, insider trades, grades, filings
- `llm.py` — same LLM factory
- `server.py` — new tools/resources added to existing FastMCP server
- `main.py` — new commands in existing REPL
- Can trigger `orchestrator.py` deep-dive graph for tickers with material changes

---

## Why the Architecture Supports This Roadmap

- **Screening** reuses `phases/gathering.py` and `phases/valuation.py` — same code, different orchestration
- **Monitoring** adds a new LangGraph graph (not nested inside the deep-dive) that calls the same FMP wrappers
- **MCP server** just gets new tools/resources added to `server.py` — same FastMCP pattern
- **CLI** gets new commands in the REPL loop — same pattern
- **No changes needed** to the core deep-dive workflow or existing phase implementations
