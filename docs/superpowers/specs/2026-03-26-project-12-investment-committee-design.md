# Project 12: Investment Committee — Multi-Framework Financial Analyst

## Overview

A co-pilot style investment research system where the user drives analysis phase by phase. Each phase uses the agentic framework best suited for it — tying together everything from P1–P11 into one real-world tool.

This is the capstone integration project for the portfolio. It proves that each framework learned individually can be composed into a single, production-useful system.

### Key Decisions

- **Co-pilot, not autopilot** — The user controls the pace, reviewing results and steering at every phase boundary.
- **Multi-framework by design** — LangGraph orchestrates, CrewAI gathers data, AG2 debates, LlamaIndex provides RAG, MCP exposes the system.
- **FMP Ultimate as data source** — All financial data comes from the FMP REST API via typed wrapper functions. User provides `FMP_API_KEY` env var.
- **Ollama (local or cloud)** — Env-configurable LLM. Same `langchain_ollama` integration, just `base_url` + API key for cloud.

---

## Workflow: Company Deep-Dive

```
User: "analyze AAPL"
    |
    v
+-----------------------------------------------------+
|  1. DATA GATHERING (CrewAI Crew)                     |
|     - Fetch financials, ratios, estimates via FMP     |
|     - Pull recent earnings transcripts                |
|     - Get insider trades, analyst grades              |
|     [interrupt() -> user reviews data summary]        |
+-----------------------------------------------------+
|  2. FINANCIAL ANALYSIS (LangGraph sub-graphs)         |
|     - Profitability analysis                          |
|     - Valuation modeling (DCF, comparables)           |
|     - Growth & momentum assessment                    |
|     [interrupt() -> user reviews findings]            |
+-----------------------------------------------------+
|  3. BULL/BEAR DEBATE (AG2 GroupChat)                  |
|     - Bull analyst argues upside                      |
|     - Bear analyst argues downside                    |
|     - Moderator synthesizes disagreements             |
|     [interrupt() -> user watches, can interject]      |
+-----------------------------------------------------+
|  4. THESIS SYNTHESIS (LangGraph + LlamaIndex RAG)     |
|     - RAG over transcripts/filings for evidence       |
|     - Draft investment thesis with citations           |
|     - Risk factors + catalyst timeline                |
|     [interrupt() -> user reviews, requests revisions] |
+-----------------------------------------------------+
|  5. REPORT GENERATION                                 |
|     - Structured markdown report                      |
|     - Saved locally + exposed via MCP resource        |
+-----------------------------------------------------+
```

### Framework Mapping

| Phase | Framework | Why | Portfolio Reference |
|-------|-----------|-----|---------------------|
| Orchestration + HITL | LangGraph | Sub-graphs, `interrupt()`, state management | P5 |
| Data gathering crew | CrewAI | Multi-agent task delegation, custom tools | P6 |
| Bull/Bear debate | AG2 | `GroupChat` with speaker selection, natural debate flow | P7 |
| RAG over filings | LlamaIndex | Document ingestion, retrieval strategies | P9 |
| MCP exposure | MCP/FastMCP | Tools + resources for external access | P11 |
| LLM provider | Ollama (local or cloud) | Shared factory, env-configurable | All projects |

---

## Architecture

Three layers:

```
+------------------------------------------------------+
|                    INTERFACE LAYER                     |
|                                                       |
|   CLI (co-pilot REPL)          MCP Server             |
|   - "analyze AAPL"             - tools: analyze,      |
|   - phase-by-phase control       get_report, compare  |
|   - rich terminal output       - resources: reports,   |
|                                  watchlist             |
+------------------+-------------------+---------------+
                   |                   |
                   v                   v
+------------------------------------------------------+
|                  ORCHESTRATION LAYER                   |
|                                                       |
|   LangGraph Parent Graph (StateGraph)                 |
|   +----------+   +-----------+   +---------+         |
|   | gather   |-->| analyze   |-->| debate  |-->...   |
|   | (CrewAI) |   | (sub-     |   | (AG2)   |         |
|   |          |   |  graphs)  |   |         |         |
|   +----------+   +-----------+   +---------+         |
|        ^              ^              ^                |
|     interrupt()    interrupt()    interrupt()         |
+----------------------+-------------------------------+
                       |
                       v
+------------------------------------------------------+
|                    SERVICE LAYER                       |
|                                                       |
|   LLM Factory          FMP Client         RAG Engine  |
|   - local/cloud Ollama  - wraps FMP MCP   - LlamaIndex|
|   - env-configured      - typed responses - transcripts|
|   - shared across all   - rate limiting   - filings    |
|     frameworks                                        |
+------------------------------------------------------+
```

### Key Design Decisions

1. **LangGraph as the spine** — The parent graph owns the overall workflow state and human checkpoints. CrewAI and AG2 run inside LangGraph nodes, not alongside it. LangGraph calls `crew.kickoff()` and `initiate_chat()` from within node functions, capturing output back into graph state.

2. **FMP via direct API calls** — The FMP MCP tools are available in Claude Code sessions, but within our Python code we call the FMP REST API directly using typed wrapper functions in `tools/fmp.py`. These wrappers use `httpx` with the user's FMP API key (`FMP_API_KEY` env var) and return typed dicts. The MCP server we build *exposes* our analysis tools — it doesn't *consume* the FMP MCP server at runtime.

3. **Shared state as TypedDict** — One `InvestmentState` flows through the whole graph. Each phase writes its section. State accumulates as the user progresses.

4. **Human checkpoints via `interrupt()`** — After each phase, the graph pauses and presents a summary. The user can: approve and continue, ask follow-up questions, request a re-run with different parameters, or skip to a specific phase.

5. **RAG is on-demand** — LlamaIndex ingests earnings transcripts and SEC filings only during the thesis phase. Documents are fetched from FMP, ingested into a temporary index, and queried during synthesis. No persistent vector store needed.

---

## State Model

```python
class InvestmentState(TypedDict):
    # Identity
    ticker: str
    company_name: str

    # Phase 1: Data Gathering (CrewAI output)
    financials: dict          # income, balance sheet, cash flow (3-5 years)
    ratios: dict              # key metrics + TTM
    estimates: dict           # analyst estimates + price targets
    insider_trades: list      # recent insider activity
    grades: list              # analyst grades + changes
    earnings_transcripts: list # raw transcript text
    data_summary: str         # human-readable summary for review

    # Phase 2: Financial Analysis (LangGraph sub-graphs output)
    profitability_analysis: str
    valuation_analysis: str   # DCF result, comps, fair value range
    growth_analysis: str
    analysis_summary: str     # consolidated for review

    # Phase 3: Debate (AG2 output)
    debate_transcript: str    # full bull/bear exchange
    key_disagreements: list   # moderator-extracted points of contention

    # Phase 4: Thesis (LangGraph + RAG output)
    investment_thesis: str    # final synthesized thesis
    risk_factors: list
    catalysts: list
    confidence_level: str     # high / medium / low

    # Phase 5: Report
    report_path: str          # saved file location

    # Control
    current_phase: str
    human_feedback: str       # captured at each interrupt
    messages: list            # conversation history
```

---

## Phase Details

### Phase 1 — Data Gathering (CrewAI Crew)

Three agents in a sequential crew:

- **Financial Data Agent** — Calls FMP tools: `income-statement`, `balance-sheet-statement`, `cashflow-statement`, `key-metrics-ttm`, `metrics-ratios-ttm`. Pulls 3-5 years of annual data plus TTM.
- **Market Intel Agent** — Calls FMP tools: `financial-estimates`, `price-target-consensus`, `grades`, `latest-insider-trade`, `insider-trade-statistics`. Builds a market sentiment picture.
- **Transcript Agent** — Calls `transcripts-dates-by-symbol` to find available dates, then fetches the 2 most recent earnings transcripts.

Output: structured data fields in state + a narrative `data_summary` for human review.

### Phase 2 — Financial Analysis (3 LangGraph Sub-graphs)

**Profitability sub-graph:**
```
analyze_margins -> analyze_returns -> assess_quality -> summarize
```
Gross/operating/net margins over time, ROE/ROA/ROIC trends, earnings quality (cash flow vs accruals).

**Valuation sub-graph:**
```
build_dcf -> run_comparables -> estimate_fair_value -> summarize
```
Simple DCF using FCF growth estimates from FMP, peer comparison on P/E, EV/EBITDA, P/FCF. Outputs a fair value range.

**Growth sub-graph:**
```
analyze_revenue_growth -> analyze_earnings_growth -> assess_momentum -> summarize
```
Revenue and EPS growth trends, estimate revisions direction.

All three sub-graphs run. Their summaries combine into `analysis_summary` for human review.

### Phase 3 — Bull/Bear Debate (AG2 GroupChat)

Three AG2 `ConversableAgent` instances:

- **Bull Analyst** — System prompt loaded with financial data + analysis. Argues FOR buying.
- **Bear Analyst** — Same data. Argues AGAINST. Finds weaknesses, risks, overvaluation.
- **Moderator** — Manages 3 rounds (bull -> bear -> bull -> bear -> bull -> bear), then synthesizes key disagreements.

Output: full `debate_transcript` + `key_disagreements` list for human review.

### Phase 4 — Thesis Synthesis (LangGraph + LlamaIndex RAG)

```
ingest_transcripts -> query_evidence -> draft_thesis -> review_thesis
```

- **Ingest**: Earnings transcripts from Phase 1 are loaded into a LlamaIndex temporary in-memory index.
- **Query**: Key claims from the debate are used as queries to find supporting/contradicting evidence in transcripts.
- **Draft**: An LLM synthesizes everything into a structured thesis: summary, bull case, bear case, risk factors, catalysts, confidence level, with citations from transcripts.
- **Review**: User reviews and can request revisions before finalizing.

### Phase 5 — Report Generation

Compiles all phases into a structured markdown report:

```markdown
# Investment Analysis: {TICKER} — {COMPANY_NAME}
## Date: {date}

## Executive Summary
{investment_thesis}

## Financial Data Overview
{data_summary}

## Profitability Analysis
{profitability_analysis}

## Valuation
{valuation_analysis}

## Growth Assessment
{growth_analysis}

## Bull/Bear Debate
{debate_transcript}
### Key Disagreements
{key_disagreements}

## Investment Thesis
{investment_thesis with citations}

## Risk Factors
{risk_factors}

## Catalysts
{catalysts}

## Confidence: {confidence_level}
```

Saved to `reports/{ticker}_{date}.md` within the project directory.

---

## MCP Server Design

### Tools (actions)

| Tool | Parameters | Description |
|------|------------|-------------|
| `analyze_company` | `ticker: str` | Start a full deep-dive. Returns Phase 1 data summary, waits for user to drive forward. |
| `continue_analysis` | `ticker: str, feedback: str?` | Advance to the next phase. Optionally pass steering instructions. |
| `get_phase_results` | `ticker: str, phase: str` | Retrieve results from a specific completed phase. |
| `rerun_phase` | `ticker: str, phase: str, instructions: str` | Re-run a phase with different parameters (e.g., "use 10% discount rate"). |
| `compare_companies` | `tickers: list[str]` | Run deep-dives on 2-3 tickers, produce side-by-side comparison. |

### Resources (read-only)

| Resource | Description |
|----------|-------------|
| `reports://{ticker}` | Full generated report for a ticker |
| `reports://list` | List of all completed reports |
| `analysis://{ticker}/{phase}` | Raw results from a specific phase |

### Prompts (templates)

| Prompt | Description |
|--------|-------------|
| `deep-dive` | Pre-built prompt: "Run a complete investment analysis on {ticker}" |
| `quick-valuation` | Shortcut: skip debate, just DCF + comps for {ticker} |

---

## CLI Interface

A REPL that consumes the same tools internally:

```
$ poetry run project-12-analyst

Investment Committee v0.1
> analyze AAPL

Gathering data for AAPL (Apple Inc.)...
[CrewAI crew runs]

-- Data Summary ------------------------------------
Revenue: $385B (TTM), +5.2% YoY
Net Margin: 25.3%, stable
FCF: $108B, FCF Yield: 3.1%
Insider Activity: 3 sells last 90 days
Analyst Consensus: 28 Buy, 11 Hold, 2 Sell
Target: $245 (current: $228)
----------------------------------------------------

What next? [analyze / debate / skip to thesis / ask a question]
> analyze

Running financial analysis...
```

Commands:
- `analyze {TICKER}` — Start deep-dive
- `continue` / `next` — Advance to next phase
- `rerun` — Re-run current phase with new instructions
- `skip to {phase}` — Jump ahead
- `report` — Generate final report
- `help` — Show available commands

---

## LLM Configuration

Shared factory in `llm.py`:

```python
import os
from langchain_ollama import ChatOllama

def get_llm(model: str = "qwen3.5:35b", temperature: float = 0.7) -> ChatOllama:
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    api_key = os.environ.get("OLLAMA_API_KEY")

    kwargs = {}
    if api_key:
        kwargs["client_kwargs"] = {
            "headers": {"Authorization": f"Bearer {api_key}"}
        }

    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=temperature,
        **kwargs,
    )
```

Environment variables:
- `OLLAMA_BASE_URL` — defaults to `http://localhost:11434` (local). Set to `https://ollama.com` for cloud.
- `OLLAMA_API_KEY` — only needed for cloud Ollama. Not set = local mode.
- `FMP_API_KEY` — **required**. Your FMP Ultimate API key for all financial data access.

---

## Project Structure

```
src/langgraph_portfolio/projects/project_12_analyst/
├── main.py                  # CLI REPL entry point
├── server.py                # MCP server (FastMCP)
├── orchestrator.py          # LangGraph parent graph + state definition
├── state.py                 # InvestmentState TypedDict + Pydantic models
├── llm.py                   # LLM factory (local/cloud Ollama)
├── phases/
│   ├── __init__.py
│   ├── gathering.py         # CrewAI crew (3 agents + FMP tool wrappers)
│   ├── profitability.py     # LangGraph sub-graph
│   ├── valuation.py         # LangGraph sub-graph (DCF + comps)
│   ├── growth.py            # LangGraph sub-graph
│   ├── debate.py            # AG2 GroupChat (bull/bear/moderator)
│   ├── thesis.py            # LangGraph + LlamaIndex RAG
│   └── report.py            # Report generation + file output
├── tools/
│   ├── __init__.py
│   └── fmp.py               # FMP REST API wrappers (typed functions)
└── reports/                  # Generated reports land here
```

Entry point in `pyproject.toml`:
```toml
project-12-analyst = "langgraph_portfolio.projects.project_12_analyst.main:main"
```

### Tests

```
tests/
├── test_project_12_analyst.py    # Smoke test (module imports, graph compiles)
├── test_p12_state.py             # State model validation
├── test_p12_phases.py            # Individual phase tests with mock data
└── test_p12_fmp_tools.py         # FMP wrapper tests
```

---

## New Concepts Learned

| Concept | Description |
|---------|-------------|
| Cross-framework composition | LangGraph orchestrating CrewAI + AG2 + LlamaIndex in one workflow |
| MCP server design | Exposing analysis capabilities as MCP tools/resources for any MCP client |
| Env-configurable LLM | Local/cloud Ollama switch via environment variables |
| Financial domain modeling | DCF, comparable analysis, financial ratios as structured state |
| Co-pilot HITL pattern | `interrupt()` between every phase, user steers the workflow |

---

## Future Roadmap

Screening & Comparison (Phase 2) and Monitoring & Alerts (Phase 3) are documented separately in `2026-03-26-project-12-roadmap.md`. The architecture above is designed to support both without changes to the core deep-dive workflow.
