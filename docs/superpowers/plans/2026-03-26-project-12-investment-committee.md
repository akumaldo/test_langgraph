# Project 12: Investment Committee — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a co-pilot investment research system that orchestrates CrewAI, AG2, LlamaIndex, and MCP inside a LangGraph parent graph with human checkpoints between every phase.

**Architecture:** LangGraph parent graph with 5 phase nodes (gather → analyze → debate → thesis → report), each calling a different framework. FMP REST API provides all financial data. Ollama (local or cloud) provides LLM. CLI REPL + MCP server as interfaces.

**Tech Stack:** LangGraph, CrewAI, AG2, LlamaIndex, FastMCP, httpx, langchain_ollama, Pydantic

---

## File Structure

```
src/langgraph_portfolio/projects/project_12_analyst/
├── __init__.py              # Package marker
├── llm.py                   # LLM factory (local/cloud Ollama)
├── state.py                 # InvestmentState TypedDict
├── tools/
│   ├── __init__.py
│   └── fmp.py               # FMP REST API wrappers (httpx, typed returns)
├── phases/
│   ├── __init__.py
│   ├── gathering.py         # CrewAI crew (3 agents for FMP data)
│   ├── profitability.py     # LangGraph sub-graph
│   ├── valuation.py         # LangGraph sub-graph (DCF + comps)
│   ├── growth.py            # LangGraph sub-graph
│   ├── debate.py            # AG2 GroupChat (bull/bear/moderator)
│   ├── thesis.py            # LangGraph + LlamaIndex RAG
│   └── report.py            # Markdown report generation
├── orchestrator.py          # LangGraph parent graph wiring all phases
├── server.py                # FastMCP server (tools + resources + prompts)
├── main.py                  # CLI REPL entry point
└── reports/                 # Generated reports (gitignored)

tests/
├── test_project_12_analyst.py   # Smoke tests (imports, graph compiles, routing)
├── test_p12_fmp_tools.py        # FMP wrapper tests (mocked HTTP)
└── test_p12_phases.py           # Phase tests with mock state
```

---

### Task 1: Project Skeleton + LLM Factory + State

**Files:**
- Create: `src/langgraph_portfolio/projects/project_12_analyst/__init__.py`
- Create: `src/langgraph_portfolio/projects/project_12_analyst/llm.py`
- Create: `src/langgraph_portfolio/projects/project_12_analyst/state.py`
- Create: `tests/test_project_12_analyst.py`

- [ ] **Step 1: Create the package directory and `__init__.py`**

```bash
mkdir -p src/langgraph_portfolio/projects/project_12_analyst/phases
mkdir -p src/langgraph_portfolio/projects/project_12_analyst/tools
mkdir -p src/langgraph_portfolio/projects/project_12_analyst/reports
```

Create `src/langgraph_portfolio/projects/project_12_analyst/__init__.py`:
```python
"""Project 12: Investment Committee — Multi-Framework Financial Analyst."""
```

Create `src/langgraph_portfolio/projects/project_12_analyst/phases/__init__.py`:
```python
"""Analysis phases — each uses a different agentic framework."""
```

Create `src/langgraph_portfolio/projects/project_12_analyst/tools/__init__.py`:
```python
"""FMP REST API tool wrappers."""
```

- [ ] **Step 2: Write the failing test for LLM factory**

Create `tests/test_project_12_analyst.py`:
```python
from __future__ import annotations
import unittest


class LLMFactoryTest(unittest.TestCase):
    """Test that the LLM factory produces a ChatOllama instance."""

    def test_get_llm_returns_chat_ollama(self) -> None:
        from langgraph_portfolio.projects.project_12_analyst.llm import get_llm

        llm = get_llm()
        self.assertEqual(llm.model, "qwen3.5:35b")
        self.assertIn("localhost", llm.base_url)

    def test_get_llm_respects_env_vars(self) -> None:
        import os
        from langgraph_portfolio.projects.project_12_analyst.llm import get_llm

        os.environ["OLLAMA_BASE_URL"] = "https://ollama.com"
        os.environ["OLLAMA_API_KEY"] = "test-key-123"
        try:
            llm = get_llm()
            self.assertIn("ollama.com", llm.base_url)
        finally:
            del os.environ["OLLAMA_BASE_URL"]
            del os.environ["OLLAMA_API_KEY"]


class StateTest(unittest.TestCase):
    """Test that InvestmentState can be constructed with all fields."""

    def test_state_construction(self) -> None:
        from langgraph_portfolio.projects.project_12_analyst.state import InvestmentState

        state: InvestmentState = {
            "ticker": "AAPL",
            "company_name": "Apple Inc.",
            "financials": {},
            "ratios": {},
            "estimates": {},
            "insider_trades": [],
            "grades": [],
            "earnings_transcripts": [],
            "data_summary": "",
            "profitability_analysis": "",
            "valuation_analysis": "",
            "growth_analysis": "",
            "analysis_summary": "",
            "debate_transcript": "",
            "key_disagreements": [],
            "investment_thesis": "",
            "risk_factors": [],
            "catalysts": [],
            "confidence_level": "",
            "report_path": "",
            "current_phase": "gathering",
            "human_feedback": "",
            "messages": [],
        }
        self.assertEqual(state["ticker"], "AAPL")
        self.assertEqual(state["current_phase"], "gathering")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `poetry run pytest tests/test_project_12_analyst.py -v`
Expected: FAIL — modules not found

- [ ] **Step 4: Implement LLM factory**

Create `src/langgraph_portfolio/projects/project_12_analyst/llm.py`:
```python
"""LLM factory — local or cloud Ollama, env-configurable.

CONCEPT: Shared LLM factory
All frameworks in this project (LangGraph, CrewAI, AG2, LlamaIndex) need an LLM.
Instead of configuring each separately, we have ONE factory that reads env vars.

Environment variables:
  OLLAMA_BASE_URL — defaults to http://localhost:11434 (local Ollama)
                    set to https://ollama.com for cloud
  OLLAMA_API_KEY  — only needed for cloud; adds Bearer token header

The factory returns a ChatOllama instance (langchain_ollama), which works
directly with LangGraph and LangChain. For CrewAI and AG2, we extract
the model name and base_url to build their own config formats.
"""

import os

from langchain_ollama import ChatOllama

DEFAULT_MODEL = "qwen3.5:35b"
DEFAULT_BASE_URL = "http://localhost:11434"


def get_llm(
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
) -> ChatOllama:
    """Create a ChatOllama instance configured from environment.

    Returns a ChatOllama that works with LangGraph nodes directly.
    For CrewAI, use get_crewai_llm_string() instead.
    For AG2, use get_ag2_llm_config() instead.
    """
    base_url = os.environ.get("OLLAMA_BASE_URL", DEFAULT_BASE_URL)
    api_key = os.environ.get("OLLAMA_API_KEY")

    kwargs: dict = {}
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


def get_crewai_llm_string(model: str = DEFAULT_MODEL) -> str:
    """Return the LiteLLM model string CrewAI expects.

    CrewAI uses LiteLLM under the hood. For Ollama, the format is
    'ollama/<model_name>'. CrewAI reads OLLAMA_HOST env var for the URL.
    """
    return f"ollama/{model}"


def get_ag2_llm_config(
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
) -> dict:
    """Return the LLM config dict AG2 expects.

    AG2 uses an OpenAI-compatible endpoint. Ollama exposes one at /v1.
    """
    base_url = os.environ.get("OLLAMA_BASE_URL", DEFAULT_BASE_URL)
    api_key = os.environ.get("OLLAMA_API_KEY", "ollama")

    return {
        "config_list": [
            {
                "model": model,
                "base_url": f"{base_url}/v1",
                "api_key": api_key,
            }
        ],
        "temperature": temperature,
    }
```

- [ ] **Step 5: Implement state model**

Create `src/langgraph_portfolio/projects/project_12_analyst/state.py`:
```python
"""Investment analysis state — flows through the entire LangGraph parent graph.

CONCEPT: Shared state as TypedDict
One InvestmentState accumulates results as the user progresses through phases.
Each phase writes its section. The 'current_phase' field tracks where we are,
and 'human_feedback' captures the user's steering input at each interrupt().

This is the PARENT graph state. Sub-graphs (profitability, valuation, growth)
define their own internal states with input/output schemas for isolation,
just like P5's department sub-graphs.
"""

from __future__ import annotations

from typing import TypedDict


class InvestmentState(TypedDict):
    """Full state for an investment deep-dive analysis."""

    # ── Identity ──
    ticker: str
    company_name: str

    # ── Phase 1: Data Gathering (CrewAI output) ──
    financials: dict          # income, balance sheet, cash flow (3-5 years)
    ratios: dict              # key metrics + TTM
    estimates: dict           # analyst estimates + price targets
    insider_trades: list      # recent insider activity
    grades: list              # analyst grades + changes
    earnings_transcripts: list  # raw transcript text
    data_summary: str         # human-readable summary for review

    # ── Phase 2: Financial Analysis (LangGraph sub-graphs output) ──
    profitability_analysis: str
    valuation_analysis: str   # DCF result, comps, fair value range
    growth_analysis: str
    analysis_summary: str     # consolidated for review

    # ── Phase 3: Debate (AG2 output) ──
    debate_transcript: str    # full bull/bear exchange
    key_disagreements: list   # moderator-extracted points of contention

    # ── Phase 4: Thesis (LangGraph + RAG output) ──
    investment_thesis: str    # final synthesized thesis
    risk_factors: list
    catalysts: list
    confidence_level: str     # high / medium / low

    # ── Phase 5: Report ──
    report_path: str          # saved file location

    # ── Control ──
    current_phase: str        # gathering | analysis | debate | thesis | report
    human_feedback: str       # captured at each interrupt()
    messages: list            # conversation history


def make_initial_state(ticker: str) -> InvestmentState:
    """Create a blank state for a new deep-dive analysis."""
    return InvestmentState(
        ticker=ticker.upper(),
        company_name="",
        financials={},
        ratios={},
        estimates={},
        insider_trades=[],
        grades=[],
        earnings_transcripts=[],
        data_summary="",
        profitability_analysis="",
        valuation_analysis="",
        growth_analysis="",
        analysis_summary="",
        debate_transcript="",
        key_disagreements=[],
        investment_thesis="",
        risk_factors=[],
        catalysts=[],
        confidence_level="",
        report_path="",
        current_phase="gathering",
        human_feedback="",
        messages=[],
    )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `poetry run pytest tests/test_project_12_analyst.py -v`
Expected: 3 tests PASS

- [ ] **Step 7: Commit**

```bash
git add src/langgraph_portfolio/projects/project_12_analyst/__init__.py \
        src/langgraph_portfolio/projects/project_12_analyst/llm.py \
        src/langgraph_portfolio/projects/project_12_analyst/state.py \
        src/langgraph_portfolio/projects/project_12_analyst/phases/__init__.py \
        src/langgraph_portfolio/projects/project_12_analyst/tools/__init__.py \
        tests/test_project_12_analyst.py
git commit -m "feat(p12): add project skeleton, LLM factory, and state model"
```

---

### Task 2: FMP REST API Wrappers

**Files:**
- Create: `src/langgraph_portfolio/projects/project_12_analyst/tools/fmp.py`
- Create: `tests/test_p12_fmp_tools.py`

- [ ] **Step 1: Write the failing test for FMP wrappers**

Create `tests/test_p12_fmp_tools.py`:
```python
from __future__ import annotations
import json
import unittest
from unittest.mock import AsyncMock, patch

import asyncio


def run(coro):
    """Helper to run async tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


class FMPClientTest(unittest.TestCase):
    """Test FMP wrappers with mocked HTTP responses."""

    @patch("langgraph_portfolio.projects.project_12_analyst.tools.fmp.httpx.AsyncClient")
    def test_get_income_statement(self, mock_client_cls) -> None:
        from langgraph_portfolio.projects.project_12_analyst.tools.fmp import FMPClient

        mock_response = AsyncMock()
        mock_response.json.return_value = [
            {"date": "2025-09-30", "revenue": 385000000000, "netIncome": 97000000000}
        ]
        mock_response.raise_for_status = lambda: None

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        client = FMPClient(api_key="test-key")
        result = run(client.get_income_statement("AAPL"))

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["revenue"], 385000000000)
        mock_client.get.assert_called_once()
        call_url = mock_client.get.call_args[0][0]
        self.assertIn("income-statement", call_url)
        self.assertIn("AAPL", call_url)

    @patch("langgraph_portfolio.projects.project_12_analyst.tools.fmp.httpx.AsyncClient")
    def test_get_key_metrics_ttm(self, mock_client_cls) -> None:
        from langgraph_portfolio.projects.project_12_analyst.tools.fmp import FMPClient

        mock_response = AsyncMock()
        mock_response.json.return_value = [
            {"peRatioTTM": 28.5, "roeTTM": 1.47, "dividendYielTTM": 0.005}
        ]
        mock_response.raise_for_status = lambda: None

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        client = FMPClient(api_key="test-key")
        result = run(client.get_key_metrics_ttm("AAPL"))

        self.assertEqual(result[0]["peRatioTTM"], 28.5)

    def test_client_requires_api_key(self) -> None:
        from langgraph_portfolio.projects.project_12_analyst.tools.fmp import FMPClient

        with self.assertRaises(ValueError):
            FMPClient(api_key="")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_p12_fmp_tools.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement FMP client**

Create `src/langgraph_portfolio/projects/project_12_analyst/tools/fmp.py`:
```python
"""FMP REST API wrappers — typed functions for Financial Modeling Prep.

CONCEPT: Service layer separation
Instead of having agents call HTTP endpoints directly, we wrap each FMP
endpoint in a typed async function. This gives us:
  1. Type safety — each function documents what it returns
  2. Testability — easy to mock at the function level
  3. Single source of truth — one place to handle auth, errors, rate limits

The FMP API uses a simple pattern: GET https://financialmodelingprep.com/api/v3/<endpoint>/<TICKER>?apikey=<key>

Environment variable: FMP_API_KEY (required)
"""

import os
from typing import Any

import httpx

BASE_URL = "https://financialmodelingprep.com/api"


class FMPClient:
    """Async client for FMP REST API."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("FMP_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "FMP API key required. Set FMP_API_KEY env var or pass api_key."
            )

    async def _get(self, path: str, params: dict | None = None) -> Any:
        """Make authenticated GET request to FMP API."""
        url = f"{BASE_URL}/{path}"
        request_params = {"apikey": self.api_key}
        if params:
            request_params.update(params)

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, params=request_params)
            response.raise_for_status()
            return response.json()

    # ── Financial Statements ──

    async def get_income_statement(
        self, ticker: str, period: str = "annual", limit: int = 5
    ) -> list[dict]:
        """Income statements (revenue, net income, EPS, margins)."""
        return await self._get(
            f"v3/income-statement/{ticker}",
            {"period": period, "limit": str(limit)},
        )

    async def get_balance_sheet(
        self, ticker: str, period: str = "annual", limit: int = 5
    ) -> list[dict]:
        """Balance sheet (assets, liabilities, equity)."""
        return await self._get(
            f"v3/balance-sheet-statement/{ticker}",
            {"period": period, "limit": str(limit)},
        )

    async def get_cash_flow(
        self, ticker: str, period: str = "annual", limit: int = 5
    ) -> list[dict]:
        """Cash flow statement (operating, investing, financing, FCF)."""
        return await self._get(
            f"v3/cash-flow-statement/{ticker}",
            {"period": period, "limit": str(limit)},
        )

    # ── Key Metrics & Ratios ──

    async def get_key_metrics_ttm(self, ticker: str) -> list[dict]:
        """Trailing twelve months key metrics (PE, ROE, FCF yield, etc.)."""
        return await self._get(f"v3/key-metrics-ttm/{ticker}")

    async def get_ratios_ttm(self, ticker: str) -> list[dict]:
        """Trailing twelve months financial ratios."""
        return await self._get(f"v3/ratios-ttm/{ticker}")

    async def get_key_metrics(
        self, ticker: str, period: str = "annual", limit: int = 5
    ) -> list[dict]:
        """Historical key metrics."""
        return await self._get(
            f"v3/key-metrics/{ticker}",
            {"period": period, "limit": str(limit)},
        )

    # ── Market Intelligence ──

    async def get_analyst_estimates(self, ticker: str, limit: int = 4) -> list[dict]:
        """Analyst consensus estimates (revenue, EPS, growth)."""
        return await self._get(
            f"v3/analyst-estimates/{ticker}",
            {"limit": str(limit)},
        )

    async def get_price_target_consensus(self, ticker: str) -> list[dict]:
        """Price target consensus (high, low, average, median)."""
        return await self._get(f"v4/price-target-consensus/{ticker}")

    async def get_grades(self, ticker: str, limit: int = 20) -> list[dict]:
        """Analyst grades and rating changes."""
        return await self._get(
            f"v3/grade/{ticker}",
            {"limit": str(limit)},
        )

    async def get_insider_trades(self, ticker: str, limit: int = 20) -> list[dict]:
        """Recent insider trades."""
        return await self._get(
            f"v4/insider-trading",
            {"symbol": ticker, "limit": str(limit)},
        )

    async def get_insider_trade_statistics(self, ticker: str) -> list[dict]:
        """Insider trading statistics summary."""
        return await self._get(
            f"v4/insider-trading-transaction-type",
            {"symbol": ticker},
        )

    # ── Company Profile ──

    async def get_profile(self, ticker: str) -> list[dict]:
        """Company profile (name, sector, market cap, description)."""
        return await self._get(f"v3/profile/{ticker}")

    async def get_peers(self, ticker: str) -> list[str]:
        """Peer companies for comparison."""
        result = await self._get(f"v4/stock_peers", {"symbol": ticker})
        if result and isinstance(result, list) and "peersList" in result[0]:
            return result[0]["peersList"]
        return []

    # ── Earnings Transcripts ──

    async def get_transcript_dates(self, ticker: str) -> list[list]:
        """Available earnings transcript dates."""
        return await self._get(f"v4/earning_call_transcript", {"symbol": ticker})

    async def get_transcript(self, ticker: str, year: int, quarter: int) -> list[dict]:
        """Single earnings call transcript."""
        return await self._get(
            f"v3/earning_call_transcript/{ticker}",
            {"year": str(year), "quarter": str(quarter)},
        )

    # ── Valuation ──

    async def get_dcf(self, ticker: str) -> list[dict]:
        """Discounted cash flow valuation."""
        return await self._get(f"v3/discounted-cash-flow/{ticker}")

    async def get_enterprise_values(
        self, ticker: str, limit: int = 5
    ) -> list[dict]:
        """Enterprise value data for comparables."""
        return await self._get(
            f"v3/enterprise-values/{ticker}",
            {"limit": str(limit)},
        )

    async def get_quote(self, ticker: str) -> list[dict]:
        """Current stock quote."""
        return await self._get(f"v3/quote/{ticker}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/test_p12_fmp_tools.py -v`
Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/langgraph_portfolio/projects/project_12_analyst/tools/fmp.py \
        tests/test_p12_fmp_tools.py
git commit -m "feat(p12): add FMP REST API client with typed wrappers"
```

---

### Task 3: Phase 1 — Data Gathering (CrewAI Crew)

**Files:**
- Create: `src/langgraph_portfolio/projects/project_12_analyst/phases/gathering.py`
- Create: `tests/test_p12_phases.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_p12_phases.py`:
```python
from __future__ import annotations
import unittest


class GatheringPhaseTest(unittest.TestCase):
    """Test the data gathering phase builds a valid CrewAI crew."""

    def test_build_gathering_crew_returns_crew(self) -> None:
        from langgraph_portfolio.projects.project_12_analyst.phases.gathering import (
            build_gathering_crew,
        )

        crew = build_gathering_crew("AAPL")
        self.assertIsNotNone(crew)
        self.assertEqual(len(crew.agents), 3)
        self.assertEqual(len(crew.tasks), 3)

    def test_gather_node_function_exists(self) -> None:
        from langgraph_portfolio.projects.project_12_analyst.phases.gathering import (
            gather_data,
        )

        self.assertTrue(callable(gather_data))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_p12_phases.py::GatheringPhaseTest -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement the gathering phase**

Create `src/langgraph_portfolio/projects/project_12_analyst/phases/gathering.py`:
```python
"""Phase 1: Data Gathering — CrewAI crew fetches financial data from FMP.

CONCEPT: CrewAI inside a LangGraph node
The parent LangGraph graph calls gather_data() as a regular node function.
Inside, we build a CrewAI crew with 3 agents that each handle a data domain.
The crew runs sequentially — each agent's output feeds the next's context.

KEY PATTERN (from P6): Separate I/O from reasoning.
We DON'T give CrewAI agents direct tool access (Ollama can't do agent tool use).
Instead, we pre-fetch ALL data from FMP using our typed wrappers, then pass
the raw data as text context to the crew. The agents' job is ANALYSIS and
SUMMARIZATION of the pre-fetched data, not fetching it themselves.

Flow:
  1. LangGraph node calls gather_data(state)
  2. gather_data() fetches all FMP data using FMPClient (async I/O)
  3. Raw data is formatted as text and passed to CrewAI crew
  4. Crew agents analyze and summarize the data
  5. Results are written back to state
"""

import asyncio
import json
from typing import Any

from crewai import Agent, Crew, Process, Task

from langgraph_portfolio.projects.project_12_analyst.llm import get_crewai_llm_string
from langgraph_portfolio.projects.project_12_analyst.state import InvestmentState
from langgraph_portfolio.projects.project_12_analyst.tools.fmp import FMPClient


LLM_MODEL = get_crewai_llm_string()


# ── CrewAI Agents ──

financial_data_agent = Agent(
    role="Financial Data Analyst",
    goal=(
        "Analyze raw financial statement data (income statement, balance sheet, "
        "cash flow) and key metrics to produce a clear financial overview."
    ),
    backstory=(
        "You are a senior financial analyst with 15 years of experience reading "
        "financial statements. You spot trends, flag anomalies, and distill "
        "complex financial data into concise summaries."
    ),
    llm=LLM_MODEL,
    verbose=True,
)

market_intel_agent = Agent(
    role="Market Intelligence Analyst",
    goal=(
        "Analyze analyst estimates, price targets, grades, and insider trading "
        "data to build a market sentiment picture."
    ),
    backstory=(
        "You are a sell-side equity research associate who tracks analyst "
        "consensus, insider activity, and institutional sentiment. You know "
        "what signals matter and what's noise."
    ),
    llm=LLM_MODEL,
    verbose=True,
)

transcript_agent = Agent(
    role="Earnings Call Analyst",
    goal=(
        "Read earnings call transcripts and extract key themes: management "
        "guidance, tone shifts, strategic priorities, and risk disclosures."
    ),
    backstory=(
        "You are a buy-side analyst who listens to every earnings call. You "
        "pick up on what management emphasizes, what they avoid, and how "
        "their tone compares to previous quarters."
    ),
    llm=LLM_MODEL,
    verbose=True,
)


def build_gathering_crew(ticker: str) -> Crew:
    """Build the 3-agent data gathering crew.

    NOTE: Task descriptions will be updated with actual data before kickoff.
    This function builds the crew structure; gather_data() fills in the data.
    """
    financial_task = Task(
        description=f"Analyze the financial statements and key metrics for {ticker}. "
        "Data will be provided in the context. Produce a summary covering: "
        "revenue trend, profitability margins, balance sheet health, cash flow quality.",
        expected_output=(
            "A structured financial overview with sections: Revenue & Growth, "
            "Profitability, Balance Sheet Health, Cash Flow Quality. Each section "
            "should highlight the 3-5 year trend and flag any concerns."
        ),
        agent=financial_data_agent,
    )

    market_task = Task(
        description=f"Analyze the market intelligence data for {ticker}. "
        "Data will be provided in the context. Produce a summary covering: "
        "analyst consensus, price target range, recent rating changes, insider activity.",
        expected_output=(
            "A structured market sentiment report with sections: Analyst Consensus, "
            "Price Targets, Recent Rating Changes, Insider Activity. Include specific "
            "numbers and flag any notable patterns."
        ),
        agent=market_intel_agent,
    )

    transcript_task = Task(
        description=f"Analyze the earnings call transcripts for {ticker}. "
        "Transcripts will be provided in the context. Extract key themes, "
        "management guidance, tone, and any risk disclosures.",
        expected_output=(
            "A summary of key themes from recent earnings calls: Management Guidance, "
            "Strategic Priorities, Tone & Confidence, Risk Disclosures. Compare across "
            "quarters if multiple transcripts are provided."
        ),
        agent=transcript_agent,
    )

    return Crew(
        agents=[financial_data_agent, market_intel_agent, transcript_agent],
        tasks=[financial_task, market_task, transcript_task],
        process=Process.sequential,
        memory=False,
        verbose=True,
    )


async def _fetch_fmp_data(ticker: str) -> dict[str, Any]:
    """Fetch all data from FMP for the gathering phase."""
    client = FMPClient()

    # Fetch everything concurrently
    (
        profile,
        income,
        balance,
        cashflow,
        metrics_ttm,
        ratios_ttm,
        estimates,
        price_target,
        grades,
        insider_trades,
        insider_stats,
        transcript_dates,
    ) = await asyncio.gather(
        client.get_profile(ticker),
        client.get_income_statement(ticker),
        client.get_balance_sheet(ticker),
        client.get_cash_flow(ticker),
        client.get_key_metrics_ttm(ticker),
        client.get_ratios_ttm(ticker),
        client.get_analyst_estimates(ticker),
        client.get_price_target_consensus(ticker),
        client.get_grades(ticker),
        client.get_insider_trades(ticker),
        client.get_insider_trade_statistics(ticker),
        client.get_transcript_dates(ticker),
    )

    # Fetch 2 most recent transcripts
    transcripts = []
    if transcript_dates and isinstance(transcript_dates, list):
        for entry in transcript_dates[:2]:
            if isinstance(entry, list) and len(entry) >= 2:
                year, quarter = entry[0], entry[1]
            elif isinstance(entry, dict):
                year = entry.get("year", entry.get("0"))
                quarter = entry.get("quarter", entry.get("1"))
            else:
                continue
            transcript = await client.get_transcript(ticker, int(year), int(quarter))
            if transcript:
                transcripts.append(transcript)

    return {
        "profile": profile,
        "income": income,
        "balance": balance,
        "cashflow": cashflow,
        "metrics_ttm": metrics_ttm,
        "ratios_ttm": ratios_ttm,
        "estimates": estimates,
        "price_target": price_target,
        "grades": grades,
        "insider_trades": insider_trades,
        "insider_stats": insider_stats,
        "transcripts": transcripts,
    }


def _truncate_json(data: Any, max_chars: int = 6000) -> str:
    """JSON-serialize data, truncating if too long for LLM context."""
    text = json.dumps(data, indent=2, default=str)
    if len(text) > max_chars:
        return text[:max_chars] + "\n\n[Data truncated for context limits]"
    return text


def gather_data(state: InvestmentState) -> dict:
    """LangGraph node: fetch FMP data and run CrewAI gathering crew.

    This is the function the parent graph calls as a node. It:
    1. Fetches all FMP data (async I/O)
    2. Builds a CrewAI crew with the data as context
    3. Runs the crew (LLM reasoning)
    4. Returns state updates
    """
    ticker = state["ticker"]

    # Step 1: Fetch all data from FMP
    raw_data = asyncio.run(_fetch_fmp_data(ticker))

    # Extract company name from profile
    company_name = ""
    if raw_data["profile"] and isinstance(raw_data["profile"], list):
        company_name = raw_data["profile"][0].get("companyName", ticker)

    # Step 2: Build crew with data injected into task descriptions
    crew = build_gathering_crew(ticker)

    # Inject actual data into task contexts
    crew.tasks[0].description += (
        f"\n\n--- FINANCIAL DATA ---\n"
        f"Income Statement:\n{_truncate_json(raw_data['income'])}\n\n"
        f"Balance Sheet:\n{_truncate_json(raw_data['balance'])}\n\n"
        f"Cash Flow:\n{_truncate_json(raw_data['cashflow'])}\n\n"
        f"Key Metrics TTM:\n{_truncate_json(raw_data['metrics_ttm'])}\n\n"
        f"Ratios TTM:\n{_truncate_json(raw_data['ratios_ttm'])}"
    )

    crew.tasks[1].description += (
        f"\n\n--- MARKET INTELLIGENCE ---\n"
        f"Analyst Estimates:\n{_truncate_json(raw_data['estimates'])}\n\n"
        f"Price Target Consensus:\n{_truncate_json(raw_data['price_target'])}\n\n"
        f"Analyst Grades:\n{_truncate_json(raw_data['grades'])}\n\n"
        f"Insider Trades:\n{_truncate_json(raw_data['insider_trades'])}\n\n"
        f"Insider Statistics:\n{_truncate_json(raw_data['insider_stats'])}"
    )

    transcript_text = ""
    for t in raw_data["transcripts"]:
        if isinstance(t, list) and t:
            transcript_text += _truncate_json(t[0]) + "\n\n"
        elif isinstance(t, dict):
            transcript_text += _truncate_json(t) + "\n\n"

    crew.tasks[2].description += (
        f"\n\n--- EARNINGS TRANSCRIPTS ---\n{transcript_text}"
        if transcript_text
        else "\n\n[No transcripts available for this ticker]"
    )

    # Step 3: Run the crew
    result = crew.kickoff()

    # Step 4: Build data summary from crew output
    data_summary = result.raw if hasattr(result, "raw") else str(result)

    # Step 5: Return state updates
    return {
        "company_name": company_name,
        "financials": {
            "income": raw_data["income"],
            "balance": raw_data["balance"],
            "cashflow": raw_data["cashflow"],
        },
        "ratios": {
            "metrics_ttm": raw_data["metrics_ttm"],
            "ratios_ttm": raw_data["ratios_ttm"],
        },
        "estimates": {
            "analyst_estimates": raw_data["estimates"],
            "price_target": raw_data["price_target"],
        },
        "insider_trades": raw_data["insider_trades"],
        "grades": raw_data["grades"],
        "earnings_transcripts": raw_data["transcripts"],
        "data_summary": data_summary,
        "current_phase": "analysis",
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/test_p12_phases.py::GatheringPhaseTest -v`
Expected: 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/langgraph_portfolio/projects/project_12_analyst/phases/gathering.py \
        tests/test_p12_phases.py
git commit -m "feat(p12): add Phase 1 data gathering with CrewAI crew"
```

---

### Task 4: Phase 2 — Financial Analysis (3 LangGraph Sub-graphs)

**Files:**
- Create: `src/langgraph_portfolio/projects/project_12_analyst/phases/profitability.py`
- Create: `src/langgraph_portfolio/projects/project_12_analyst/phases/valuation.py`
- Create: `src/langgraph_portfolio/projects/project_12_analyst/phases/growth.py`
- Modify: `tests/test_p12_phases.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_p12_phases.py`:
```python
class ProfitabilitySubgraphTest(unittest.TestCase):
    """Test the profitability analysis sub-graph compiles."""

    def test_subgraph_compiles(self) -> None:
        from langgraph_portfolio.projects.project_12_analyst.phases.profitability import (
            build_profitability_subgraph,
        )

        graph = build_profitability_subgraph()
        self.assertIsNotNone(graph)


class ValuationSubgraphTest(unittest.TestCase):
    """Test the valuation analysis sub-graph compiles."""

    def test_subgraph_compiles(self) -> None:
        from langgraph_portfolio.projects.project_12_analyst.phases.valuation import (
            build_valuation_subgraph,
        )

        graph = build_valuation_subgraph()
        self.assertIsNotNone(graph)


class GrowthSubgraphTest(unittest.TestCase):
    """Test the growth analysis sub-graph compiles."""

    def test_subgraph_compiles(self) -> None:
        from langgraph_portfolio.projects.project_12_analyst.phases.growth import (
            build_growth_subgraph,
        )

        graph = build_growth_subgraph()
        self.assertIsNotNone(graph)


class AnalyzeNodeTest(unittest.TestCase):
    """Test the analyze_financials node function that runs all 3 sub-graphs."""

    def test_analyze_node_exists_in_profitability(self) -> None:
        from langgraph_portfolio.projects.project_12_analyst.phases.profitability import (
            build_profitability_subgraph,
        )

        self.assertTrue(callable(build_profitability_subgraph))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/test_p12_phases.py -k "Subgraph or AnalyzeNode" -v`
Expected: FAIL — modules not found

- [ ] **Step 3: Implement profitability sub-graph**

Create `src/langgraph_portfolio/projects/project_12_analyst/phases/profitability.py`:
```python
"""Phase 2a: Profitability Analysis — LangGraph sub-graph.

CONCEPT: Sub-graph with input/output isolation (from P5)
This sub-graph receives financial data and produces a profitability summary.
It defines its own internal state with input/output schemas so the parent
graph only sees what it needs.

Flow: analyze_margins → analyze_returns → assess_quality → summarize
"""

from __future__ import annotations

import json
from typing import TypedDict

from langgraph.graph import END, StateGraph

from langgraph_portfolio.projects.project_12_analyst.llm import get_llm


# ── Sub-graph State ──

class ProfitabilityInput(TypedDict):
    """What the parent graph passes in."""
    financials: dict
    ratios: dict


class ProfitabilityOutput(TypedDict):
    """What the parent graph gets back."""
    profitability_analysis: str


class ProfitabilityState(TypedDict):
    """Internal state for profitability analysis."""
    financials: dict
    ratios: dict
    margin_analysis: str
    return_analysis: str
    quality_assessment: str
    profitability_analysis: str


# ── Node Functions ──

def analyze_margins(state: ProfitabilityState) -> dict:
    """Analyze gross, operating, and net margins over time."""
    llm = get_llm(temperature=0.3)
    financials_str = json.dumps(state["financials"], indent=2, default=str)[:8000]
    ratios_str = json.dumps(state["ratios"], indent=2, default=str)[:4000]

    prompt = (
        "You are a financial analyst. Analyze the profit margins from the data below.\n"
        "Focus on: gross margin, operating margin, net margin trends over 3-5 years.\n"
        "Flag any deterioration or improvement. Be specific with numbers.\n\n"
        f"Financial Statements:\n{financials_str}\n\n"
        f"Ratios:\n{ratios_str}\n\n"
        "Margin Analysis:"
    )
    response = llm.invoke(prompt)
    return {"margin_analysis": response.content}


def analyze_returns(state: ProfitabilityState) -> dict:
    """Analyze ROE, ROA, ROIC trends."""
    llm = get_llm(temperature=0.3)
    ratios_str = json.dumps(state["ratios"], indent=2, default=str)[:6000]

    prompt = (
        "You are a financial analyst. Analyze the return metrics from the data below.\n"
        "Focus on: ROE, ROA, ROIC trends. Compare to typical benchmarks.\n"
        "Flag any red flags (declining returns, leverage-driven ROE, etc.).\n\n"
        f"Ratios & Metrics:\n{ratios_str}\n\n"
        "Return Analysis:"
    )
    response = llm.invoke(prompt)
    return {"return_analysis": response.content}


def assess_quality(state: ProfitabilityState) -> dict:
    """Assess earnings quality — cash flow vs accruals."""
    llm = get_llm(temperature=0.3)
    financials_str = json.dumps(state["financials"], indent=2, default=str)[:8000]

    prompt = (
        "You are a financial analyst. Assess earnings quality from the data below.\n"
        "Compare operating cash flow to net income (accrual ratio).\n"
        "Check if FCF supports reported earnings. Flag any divergence.\n\n"
        f"Financial Statements:\n{financials_str}\n\n"
        "Earnings Quality Assessment:"
    )
    response = llm.invoke(prompt)
    return {"quality_assessment": response.content}


def summarize(state: ProfitabilityState) -> dict:
    """Combine margin, return, and quality analyses into a summary."""
    llm = get_llm(temperature=0.3)

    prompt = (
        "You are a senior financial analyst. Synthesize these three analyses into "
        "a concise profitability assessment (3-5 paragraphs).\n\n"
        f"Margin Analysis:\n{state['margin_analysis']}\n\n"
        f"Return Analysis:\n{state['return_analysis']}\n\n"
        f"Quality Assessment:\n{state['quality_assessment']}\n\n"
        "Profitability Summary:"
    )
    response = llm.invoke(prompt)
    return {"profitability_analysis": response.content}


# ── Build Sub-graph ──

def build_profitability_subgraph():
    """Compile the profitability analysis sub-graph.

    Returns a compiled graph that can be added as a node in the parent graph.
    Uses input/output schemas so only the right fields flow in and out.
    """
    graph = StateGraph(
        ProfitabilityState,
        input=ProfitabilityInput,
        output=ProfitabilityOutput,
    )

    graph.add_node("analyze_margins", analyze_margins)
    graph.add_node("analyze_returns", analyze_returns)
    graph.add_node("assess_quality", assess_quality)
    graph.add_node("summarize", summarize)

    graph.set_entry_point("analyze_margins")
    graph.add_edge("analyze_margins", "analyze_returns")
    graph.add_edge("analyze_returns", "assess_quality")
    graph.add_edge("assess_quality", "summarize")
    graph.add_edge("summarize", END)

    return graph.compile()
```

- [ ] **Step 4: Implement valuation sub-graph**

Create `src/langgraph_portfolio/projects/project_12_analyst/phases/valuation.py`:
```python
"""Phase 2b: Valuation Analysis — LangGraph sub-graph.

CONCEPT: Financial modeling as graph nodes
Each node performs one valuation approach (DCF, comparables, fair value).
The LLM acts as the analyst interpreting the numbers, not computing them —
FMP provides the raw calculations, the LLM provides the judgment.

Flow: build_dcf → run_comparables → estimate_fair_value → summarize
"""

from __future__ import annotations

import json
from typing import TypedDict

from langgraph.graph import END, StateGraph

from langgraph_portfolio.projects.project_12_analyst.llm import get_llm


# ── Sub-graph State ──

class ValuationInput(TypedDict):
    financials: dict
    ratios: dict
    estimates: dict


class ValuationOutput(TypedDict):
    valuation_analysis: str


class ValuationState(TypedDict):
    financials: dict
    ratios: dict
    estimates: dict
    dcf_analysis: str
    comparables_analysis: str
    fair_value_estimate: str
    valuation_analysis: str


# ── Node Functions ──

def build_dcf(state: ValuationState) -> dict:
    """Analyze DCF valuation using FMP data and estimates."""
    llm = get_llm(temperature=0.3)
    cashflow_str = json.dumps(
        state["financials"].get("cashflow", {}), indent=2, default=str
    )[:6000]
    estimates_str = json.dumps(state["estimates"], indent=2, default=str)[:4000]

    prompt = (
        "You are a valuation analyst. Build a DCF analysis from the data below.\n"
        "Use free cash flow trends and analyst growth estimates to project 5 years.\n"
        "Apply a reasonable discount rate (8-12% WACC). Show your assumptions.\n"
        "Calculate an implied share price range.\n\n"
        f"Cash Flow Statements:\n{cashflow_str}\n\n"
        f"Analyst Estimates:\n{estimates_str}\n\n"
        "DCF Analysis:"
    )
    response = llm.invoke(prompt)
    return {"dcf_analysis": response.content}


def run_comparables(state: ValuationState) -> dict:
    """Analyze relative valuation using peer multiples."""
    llm = get_llm(temperature=0.3)
    ratios_str = json.dumps(state["ratios"], indent=2, default=str)[:6000]
    estimates_str = json.dumps(state["estimates"], indent=2, default=str)[:4000]

    prompt = (
        "You are a valuation analyst. Perform a comparable company analysis.\n"
        "Using the metrics below, assess: P/E, EV/EBITDA, P/FCF, PEG ratio.\n"
        "Compare current multiples to historical averages and sector norms.\n"
        "Is the stock trading at a premium or discount? Why might that be?\n\n"
        f"Ratios & Metrics:\n{ratios_str}\n\n"
        f"Estimates:\n{estimates_str}\n\n"
        "Comparables Analysis:"
    )
    response = llm.invoke(prompt)
    return {"comparables_analysis": response.content}


def estimate_fair_value(state: ValuationState) -> dict:
    """Synthesize DCF and comparables into a fair value range."""
    llm = get_llm(temperature=0.3)

    prompt = (
        "You are a senior valuation analyst. Based on the DCF and comparables "
        "analyses below, estimate a fair value range for the stock.\n"
        "Weight both approaches. Explain your confidence in the range.\n"
        "Identify what would make you revise up or down.\n\n"
        f"DCF Analysis:\n{state['dcf_analysis']}\n\n"
        f"Comparables Analysis:\n{state['comparables_analysis']}\n\n"
        "Fair Value Estimate:"
    )
    response = llm.invoke(prompt)
    return {"fair_value_estimate": response.content}


def summarize(state: ValuationState) -> dict:
    """Combine all valuation work into a summary."""
    llm = get_llm(temperature=0.3)

    prompt = (
        "You are a senior analyst. Write a concise valuation summary (3-5 paragraphs) "
        "covering DCF, comparables, and your fair value estimate.\n\n"
        f"DCF:\n{state['dcf_analysis']}\n\n"
        f"Comparables:\n{state['comparables_analysis']}\n\n"
        f"Fair Value:\n{state['fair_value_estimate']}\n\n"
        "Valuation Summary:"
    )
    response = llm.invoke(prompt)
    return {"valuation_analysis": response.content}


def build_valuation_subgraph():
    """Compile the valuation analysis sub-graph."""
    graph = StateGraph(
        ValuationState,
        input=ValuationInput,
        output=ValuationOutput,
    )

    graph.add_node("build_dcf", build_dcf)
    graph.add_node("run_comparables", run_comparables)
    graph.add_node("estimate_fair_value", estimate_fair_value)
    graph.add_node("summarize", summarize)

    graph.set_entry_point("build_dcf")
    graph.add_edge("build_dcf", "run_comparables")
    graph.add_edge("run_comparables", "estimate_fair_value")
    graph.add_edge("estimate_fair_value", "summarize")
    graph.add_edge("summarize", END)

    return graph.compile()
```

- [ ] **Step 5: Implement growth sub-graph**

Create `src/langgraph_portfolio/projects/project_12_analyst/phases/growth.py`:
```python
"""Phase 2c: Growth Analysis — LangGraph sub-graph.

Flow: analyze_revenue_growth → analyze_earnings_growth → assess_momentum → summarize
"""

from __future__ import annotations

import json
from typing import TypedDict

from langgraph.graph import END, StateGraph

from langgraph_portfolio.projects.project_12_analyst.llm import get_llm


# ── Sub-graph State ──

class GrowthInput(TypedDict):
    financials: dict
    ratios: dict
    estimates: dict


class GrowthOutput(TypedDict):
    growth_analysis: str


class GrowthState(TypedDict):
    financials: dict
    ratios: dict
    estimates: dict
    revenue_growth: str
    earnings_growth: str
    momentum_assessment: str
    growth_analysis: str


# ── Node Functions ──

def analyze_revenue_growth(state: GrowthState) -> dict:
    """Analyze revenue growth trends and drivers."""
    llm = get_llm(temperature=0.3)
    income_str = json.dumps(
        state["financials"].get("income", {}), indent=2, default=str
    )[:6000]
    estimates_str = json.dumps(state["estimates"], indent=2, default=str)[:4000]

    prompt = (
        "You are a growth analyst. Analyze revenue growth from the data below.\n"
        "Focus on: YoY growth rate trend, organic vs inorganic, segment breakdown "
        "if visible, forward estimates vs historical growth.\n\n"
        f"Income Statements:\n{income_str}\n\n"
        f"Estimates:\n{estimates_str}\n\n"
        "Revenue Growth Analysis:"
    )
    response = llm.invoke(prompt)
    return {"revenue_growth": response.content}


def analyze_earnings_growth(state: GrowthState) -> dict:
    """Analyze EPS growth and operating leverage."""
    llm = get_llm(temperature=0.3)
    income_str = json.dumps(
        state["financials"].get("income", {}), indent=2, default=str
    )[:6000]
    ratios_str = json.dumps(state["ratios"], indent=2, default=str)[:4000]

    prompt = (
        "You are a growth analyst. Analyze earnings growth from the data below.\n"
        "Focus on: EPS growth trend, operating leverage (is earnings growing faster "
        "than revenue?), estimate revision direction.\n\n"
        f"Income Statements:\n{income_str}\n\n"
        f"Ratios:\n{ratios_str}\n\n"
        "Earnings Growth Analysis:"
    )
    response = llm.invoke(prompt)
    return {"earnings_growth": response.content}


def assess_momentum(state: GrowthState) -> dict:
    """Assess growth momentum — accelerating or decelerating?"""
    llm = get_llm(temperature=0.3)

    prompt = (
        "You are a growth analyst. Based on the revenue and earnings analyses below, "
        "assess the growth momentum. Is growth accelerating, stable, or decelerating? "
        "What are the leading indicators?\n\n"
        f"Revenue Growth:\n{state['revenue_growth']}\n\n"
        f"Earnings Growth:\n{state['earnings_growth']}\n\n"
        "Momentum Assessment:"
    )
    response = llm.invoke(prompt)
    return {"momentum_assessment": response.content}


def summarize(state: GrowthState) -> dict:
    """Combine growth analyses into a summary."""
    llm = get_llm(temperature=0.3)

    prompt = (
        "You are a senior analyst. Write a concise growth summary (3-5 paragraphs).\n\n"
        f"Revenue:\n{state['revenue_growth']}\n\n"
        f"Earnings:\n{state['earnings_growth']}\n\n"
        f"Momentum:\n{state['momentum_assessment']}\n\n"
        "Growth Summary:"
    )
    response = llm.invoke(prompt)
    return {"growth_analysis": response.content}


def build_growth_subgraph():
    """Compile the growth analysis sub-graph."""
    graph = StateGraph(
        GrowthState,
        input=GrowthInput,
        output=GrowthOutput,
    )

    graph.add_node("analyze_revenue_growth", analyze_revenue_growth)
    graph.add_node("analyze_earnings_growth", analyze_earnings_growth)
    graph.add_node("assess_momentum", assess_momentum)
    graph.add_node("summarize", summarize)

    graph.set_entry_point("analyze_revenue_growth")
    graph.add_edge("analyze_revenue_growth", "analyze_earnings_growth")
    graph.add_edge("analyze_earnings_growth", "assess_momentum")
    graph.add_edge("assess_momentum", "summarize")
    graph.add_edge("summarize", END)

    return graph.compile()
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `poetry run pytest tests/test_p12_phases.py -k "Subgraph or AnalyzeNode" -v`
Expected: 4 tests PASS

- [ ] **Step 7: Commit**

```bash
git add src/langgraph_portfolio/projects/project_12_analyst/phases/profitability.py \
        src/langgraph_portfolio/projects/project_12_analyst/phases/valuation.py \
        src/langgraph_portfolio/projects/project_12_analyst/phases/growth.py \
        tests/test_p12_phases.py
git commit -m "feat(p12): add Phase 2 financial analysis sub-graphs (profitability, valuation, growth)"
```

---

### Task 5: Phase 3 — Bull/Bear Debate (AG2 GroupChat)

**Files:**
- Create: `src/langgraph_portfolio/projects/project_12_analyst/phases/debate.py`
- Modify: `tests/test_p12_phases.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_p12_phases.py`:
```python
class DebatePhaseTest(unittest.TestCase):
    """Test the debate phase builds valid AG2 agents."""

    def test_build_debate_agents_returns_three(self) -> None:
        from langgraph_portfolio.projects.project_12_analyst.phases.debate import (
            build_debate_agents,
        )

        agents = build_debate_agents(
            data_summary="Revenue growing 5% YoY",
            analysis_summary="Fair value $245, trading at $228",
        )
        self.assertEqual(len(agents), 3)
        names = {a.name for a in agents}
        self.assertEqual(names, {"Bull_Analyst", "Bear_Analyst", "Moderator"})

    def test_run_debate_node_exists(self) -> None:
        from langgraph_portfolio.projects.project_12_analyst.phases.debate import (
            run_debate,
        )

        self.assertTrue(callable(run_debate))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_p12_phases.py::DebatePhaseTest -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement the debate phase**

Create `src/langgraph_portfolio/projects/project_12_analyst/phases/debate.py`:
```python
"""Phase 3: Bull/Bear Debate — AG2 GroupChat.

CONCEPT: AG2 inside a LangGraph node (from P7)
The parent graph calls run_debate(state) as a node. Inside, we build
3 AG2 ConversableAgents and run a GroupChat with custom speaker selection.

The debate runs for 3 rounds:
  Round 1: Bull opens → Bear responds
  Round 2: Bull rebuts → Bear rebuts
  Round 3: Bull closing → Bear closing
Then the Moderator synthesizes key disagreements.

All financial data and analysis results are injected into agent system
prompts so they argue with evidence, not generalities.
"""

from __future__ import annotations

import re

from ag2 import ConversableAgent, GroupChat, GroupChatManager

from langgraph_portfolio.projects.project_12_analyst.llm import get_ag2_llm_config
from langgraph_portfolio.projects.project_12_analyst.state import InvestmentState


MAX_ROUNDS = 10  # 3 rounds * 2 debaters + moderator open + moderator close + buffer


def build_debate_agents(
    data_summary: str,
    analysis_summary: str,
) -> list[ConversableAgent]:
    """Build the 3 debate agents with financial context in system prompts."""
    llm_config = get_ag2_llm_config()

    bull = ConversableAgent(
        name="Bull_Analyst",
        system_message=(
            "You are a senior equity analyst arguing IN FAVOR of buying this stock.\n\n"
            "You have access to the following research:\n"
            f"--- DATA SUMMARY ---\n{data_summary[:4000]}\n\n"
            f"--- ANALYSIS ---\n{analysis_summary[:4000]}\n\n"
            "Rules:\n"
            "- Present clear arguments backed by specific numbers from the data\n"
            "- Address and rebut the Bear's points directly\n"
            "- Acknowledge risks but explain why they're manageable\n"
            "- Keep each response to 2-3 focused paragraphs\n"
        ),
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    bear = ConversableAgent(
        name="Bear_Analyst",
        system_message=(
            "You are a senior equity analyst arguing AGAINST buying this stock.\n\n"
            "You have access to the following research:\n"
            f"--- DATA SUMMARY ---\n{data_summary[:4000]}\n\n"
            f"--- ANALYSIS ---\n{analysis_summary[:4000]}\n\n"
            "Rules:\n"
            "- Find weaknesses, risks, and overvaluation signals in the data\n"
            "- Challenge the Bull's assumptions with specific counter-evidence\n"
            "- Highlight what could go wrong and downside scenarios\n"
            "- Keep each response to 2-3 focused paragraphs\n"
        ),
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    moderator = ConversableAgent(
        name="Moderator",
        system_message=(
            "You are an investment committee moderator. Your job:\n"
            "1. Open the debate with a brief framing of the investment question\n"
            "2. After 3 rounds, synthesize the KEY DISAGREEMENTS between Bull and Bear\n"
            "3. List each disagreement as a bullet point with both sides' positions\n"
            "4. End your final summary with the word DEBATE_OVER on its own line\n\n"
            "Do NOT take sides. Be neutral and precise.\n"
        ),
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    return [bull, bear, moderator]


def _select_next_speaker(
    last_speaker: ConversableAgent,
    groupchat: GroupChat,
) -> ConversableAgent:
    """Custom speaker selection: Moderator opens, then Bull/Bear alternate,
    Moderator closes."""
    agent_map = {a.name: a for a in groupchat.agents}
    n = len(groupchat.messages)

    # Moderator opens
    if n <= 1:
        return agent_map["Moderator"]

    # After 7 messages (open + 3 rounds of 2), Moderator summarizes
    if n >= 7:
        return agent_map["Moderator"]

    # During debate: alternate Bull → Bear
    if last_speaker.name == "Moderator":
        return agent_map["Bull_Analyst"]
    elif last_speaker.name == "Bull_Analyst":
        return agent_map["Bear_Analyst"]
    elif last_speaker.name == "Bear_Analyst":
        return agent_map["Bull_Analyst"]

    return agent_map["Moderator"]


def _is_debate_over(msg: dict) -> bool:
    """Check if the Moderator has ended the debate."""
    content = msg.get("content", "") or ""
    return "DEBATE_OVER" in content


def _extract_key_disagreements(messages: list[dict]) -> list[str]:
    """Extract key disagreements from the Moderator's final summary."""
    for msg in reversed(messages):
        if msg.get("name") == "Moderator" and "DEBATE_OVER" in (msg.get("content") or ""):
            content = msg["content"]
            # Extract bullet points
            bullets = re.findall(r"[-•]\s*(.+)", content)
            if bullets:
                return bullets
            # Fallback: return the whole summary
            return [content.replace("DEBATE_OVER", "").strip()]
    return ["No disagreements extracted"]


def run_debate(state: InvestmentState) -> dict:
    """LangGraph node: run the AG2 bull/bear debate.

    Takes the data_summary and analysis_summary from previous phases
    and feeds them to the debate agents as context.
    """
    data_summary = state.get("data_summary", "")
    analysis_summary = state.get("analysis_summary", "")
    ticker = state["ticker"]

    # Build agents with financial context
    agents = build_debate_agents(data_summary, analysis_summary)
    bull, bear, moderator = agents

    # Apply termination check
    for agent in agents:
        agent._is_termination_msg = _is_debate_over

    # Build GroupChat
    group_chat = GroupChat(
        agents=[moderator, bull, bear],
        messages=[],
        max_round=MAX_ROUNDS,
        speaker_selection_method=_select_next_speaker,
        allow_repeat_speaker=False,
    )

    manager = GroupChatManager(
        groupchat=group_chat,
        llm_config=get_ag2_llm_config(),
    )

    # Run the debate
    moderator.initiate_chat(
        manager,
        message=(
            f"The investment committee is evaluating {ticker}. "
            f"Bull_Analyst, please open with your investment case."
        ),
    )

    # Collect results
    messages = group_chat.messages
    transcript_parts = []
    for msg in messages:
        name = msg.get("name", "Unknown")
        content = msg.get("content", "")
        transcript_parts.append(f"**{name}:**\n{content}\n")

    debate_transcript = "\n---\n\n".join(transcript_parts)
    key_disagreements = _extract_key_disagreements(messages)

    return {
        "debate_transcript": debate_transcript,
        "key_disagreements": key_disagreements,
        "current_phase": "thesis",
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/test_p12_phases.py::DebatePhaseTest -v`
Expected: 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/langgraph_portfolio/projects/project_12_analyst/phases/debate.py \
        tests/test_p12_phases.py
git commit -m "feat(p12): add Phase 3 bull/bear debate with AG2 GroupChat"
```

---

### Task 6: Phase 4 — Thesis Synthesis (LangGraph + LlamaIndex RAG)

**Files:**
- Create: `src/langgraph_portfolio/projects/project_12_analyst/phases/thesis.py`
- Modify: `tests/test_p12_phases.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_p12_phases.py`:
```python
class ThesisPhaseTest(unittest.TestCase):
    """Test the thesis synthesis phase."""

    def test_build_thesis_subgraph_compiles(self) -> None:
        from langgraph_portfolio.projects.project_12_analyst.phases.thesis import (
            build_thesis_subgraph,
        )

        graph = build_thesis_subgraph()
        self.assertIsNotNone(graph)

    def test_synthesize_thesis_node_exists(self) -> None:
        from langgraph_portfolio.projects.project_12_analyst.phases.thesis import (
            synthesize_thesis,
        )

        self.assertTrue(callable(synthesize_thesis))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_p12_phases.py::ThesisPhaseTest -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement thesis phase**

Create `src/langgraph_portfolio/projects/project_12_analyst/phases/thesis.py`:
```python
"""Phase 4: Thesis Synthesis — LangGraph + LlamaIndex RAG.

CONCEPT: On-demand RAG (from P9)
We only build the LlamaIndex index when the user reaches the thesis phase.
Earnings transcripts (fetched in Phase 1) are ingested into a temporary
in-memory vector index. Key claims from the debate are used as queries
to find supporting/contradicting evidence.

Flow: ingest_transcripts → query_evidence → draft_thesis

This is both a LangGraph sub-graph AND uses LlamaIndex internally.
The sub-graph nodes call LlamaIndex for retrieval and LangGraph's LLM
for synthesis.
"""

from __future__ import annotations

import asyncio
from typing import TypedDict

from langgraph.graph import END, StateGraph

from langgraph_portfolio.projects.project_12_analyst.llm import get_llm


# ── Sub-graph State ──

class ThesisInput(TypedDict):
    ticker: str
    company_name: str
    data_summary: str
    analysis_summary: str
    debate_transcript: str
    key_disagreements: list
    earnings_transcripts: list


class ThesisOutput(TypedDict):
    investment_thesis: str
    risk_factors: list
    catalysts: list
    confidence_level: str


class ThesisState(TypedDict):
    ticker: str
    company_name: str
    data_summary: str
    analysis_summary: str
    debate_transcript: str
    key_disagreements: list
    earnings_transcripts: list
    rag_evidence: str
    investment_thesis: str
    risk_factors: list
    catalysts: list
    confidence_level: str


# ── Node Functions ──

def ingest_and_query(state: ThesisState) -> dict:
    """Ingest transcripts into LlamaIndex index and query for evidence.

    Uses LlamaIndex's VectorStoreIndex for semantic search over
    earnings call transcripts. Queries key disagreements from the
    debate to find supporting/contradicting evidence.
    """
    transcripts = state.get("earnings_transcripts", [])
    key_disagreements = state.get("key_disagreements", [])

    if not transcripts:
        return {"rag_evidence": "No earnings transcripts available for RAG."}

    # Build transcript text for indexing
    transcript_texts = []
    for t in transcripts:
        if isinstance(t, list) and t:
            entry = t[0] if isinstance(t[0], dict) else t
        elif isinstance(t, dict):
            entry = t
        else:
            continue
        content = entry.get("content", str(entry))
        transcript_texts.append(str(content))

    if not transcript_texts:
        return {"rag_evidence": "No parseable transcript content found."}

    # Use LlamaIndex for RAG
    try:
        from llama_index.core import Document, Settings, VectorStoreIndex
        from llama_index.core.node_parser import SentenceSplitter
        from langchain_ollama import OllamaEmbeddings

        # Build documents
        documents = [
            Document(text=text, metadata={"source": f"transcript_{i}"})
            for i, text in enumerate(transcript_texts)
        ]

        # Chunk documents
        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        nodes = splitter.get_nodes_from_documents(documents)

        if not nodes:
            return {"rag_evidence": "Transcript chunking produced no nodes."}

        # Build in-memory index
        # Note: Uses default LlamaIndex embedding; for Ollama embeddings,
        # the user needs llama-index-embeddings-ollama installed (from P9)
        try:
            from llama_index.embeddings.ollama import OllamaEmbedding
            Settings.embed_model = OllamaEmbedding(model_name="qwen3-embedding")
        except ImportError:
            pass  # Fall back to LlamaIndex default

        index = VectorStoreIndex(nodes=nodes)
        query_engine = index.as_query_engine(similarity_top_k=5)

        # Query each key disagreement for evidence
        evidence_parts = []
        queries = key_disagreements[:5] if key_disagreements else [
            "What is management's outlook?",
            "What are the key risks mentioned?",
        ]

        for query_text in queries:
            response = query_engine.query(str(query_text))
            evidence_parts.append(
                f"**Query:** {query_text}\n**Evidence:** {response.response}\n"
            )

        return {"rag_evidence": "\n---\n".join(evidence_parts)}

    except ImportError:
        # LlamaIndex not available — fall back to direct transcript excerpts
        excerpt = "\n\n".join(t[:3000] for t in transcript_texts)
        return {
            "rag_evidence": (
                f"[LlamaIndex not available — raw transcript excerpts]\n\n{excerpt}"
            )
        }


def synthesize_thesis(state: ThesisState) -> dict:
    """Draft the investment thesis using all accumulated evidence."""
    llm = get_llm(temperature=0.5)

    prompt = (
        f"You are the chief investment strategist. Write a comprehensive investment "
        f"thesis for {state['ticker']} ({state['company_name']}).\n\n"
        f"You have:\n"
        f"1. Financial data summary:\n{state['data_summary'][:3000]}\n\n"
        f"2. Detailed analysis:\n{state['analysis_summary'][:3000]}\n\n"
        f"3. Bull/Bear debate key disagreements:\n"
        f"{chr(10).join('- ' + d for d in state['key_disagreements'])}\n\n"
        f"4. Evidence from earnings transcripts:\n{state['rag_evidence'][:3000]}\n\n"
        "Write your thesis with these exact sections:\n"
        "INVESTMENT THESIS: 2-3 paragraph summary of your conclusion\n"
        "RISK FACTORS: Bullet list of 3-5 key risks\n"
        "CATALYSTS: Bullet list of 3-5 potential positive catalysts\n"
        "CONFIDENCE: One of HIGH, MEDIUM, or LOW with a brief justification\n"
    )
    response = llm.invoke(prompt)
    content = response.content

    # Parse structured sections from LLM output
    risk_factors = _extract_bullets(content, "RISK FACTORS")
    catalysts = _extract_bullets(content, "CATALYSTS")
    confidence = _extract_confidence(content)

    return {
        "investment_thesis": content,
        "risk_factors": risk_factors,
        "catalysts": catalysts,
        "confidence_level": confidence,
    }


def _extract_bullets(text: str, section_name: str) -> list[str]:
    """Extract bullet points from a named section."""
    import re

    pattern = rf"{section_name}[:\s]*\n((?:[-•*]\s*.+\n?)+)"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        bullets = re.findall(r"[-•*]\s*(.+)", match.group(1))
        return [b.strip() for b in bullets if b.strip()]
    return [f"See thesis for {section_name.lower()}"]


def _extract_confidence(text: str) -> str:
    """Extract confidence level from the thesis."""
    import re

    match = re.search(r"CONFIDENCE[:\s]*(HIGH|MEDIUM|LOW)", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return "MEDIUM"


# ── Build Sub-graph ──

def build_thesis_subgraph():
    """Compile the thesis synthesis sub-graph."""
    graph = StateGraph(
        ThesisState,
        input=ThesisInput,
        output=ThesisOutput,
    )

    graph.add_node("ingest_and_query", ingest_and_query)
    graph.add_node("synthesize_thesis", synthesize_thesis)

    graph.set_entry_point("ingest_and_query")
    graph.add_edge("ingest_and_query", "synthesize_thesis")
    graph.add_edge("synthesize_thesis", END)

    return graph.compile()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/test_p12_phases.py::ThesisPhaseTest -v`
Expected: 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/langgraph_portfolio/projects/project_12_analyst/phases/thesis.py \
        tests/test_p12_phases.py
git commit -m "feat(p12): add Phase 4 thesis synthesis with LlamaIndex RAG"
```

---

### Task 7: Phase 5 — Report Generation

**Files:**
- Create: `src/langgraph_portfolio/projects/project_12_analyst/phases/report.py`
- Modify: `tests/test_p12_phases.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_p12_phases.py`:
```python
class ReportPhaseTest(unittest.TestCase):
    """Test report generation."""

    def test_generate_report_produces_markdown(self) -> None:
        from langgraph_portfolio.projects.project_12_analyst.phases.report import (
            generate_report,
        )
        from langgraph_portfolio.projects.project_12_analyst.state import InvestmentState

        state: InvestmentState = {
            "ticker": "AAPL",
            "company_name": "Apple Inc.",
            "financials": {},
            "ratios": {},
            "estimates": {},
            "insider_trades": [],
            "grades": [],
            "earnings_transcripts": [],
            "data_summary": "Revenue: $385B",
            "profitability_analysis": "Margins stable",
            "valuation_analysis": "Fair value $245",
            "growth_analysis": "Growing 5% YoY",
            "analysis_summary": "Solid fundamentals",
            "debate_transcript": "Bull said X, Bear said Y",
            "key_disagreements": ["Valuation premium justified?"],
            "investment_thesis": "Buy with medium confidence",
            "risk_factors": ["Slowing growth", "China risk"],
            "catalysts": ["AI spending", "Services growth"],
            "confidence_level": "MEDIUM",
            "report_path": "",
            "current_phase": "report",
            "human_feedback": "",
            "messages": [],
        }
        result = generate_report(state)
        self.assertIn("report_path", result)
        self.assertIn("AAPL", result["report_path"])
        # Verify the report content
        self.assertIn("# Investment Analysis: AAPL", result.get("_report_content", ""))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_p12_phases.py::ReportPhaseTest -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement report generation**

Create `src/langgraph_portfolio/projects/project_12_analyst/phases/report.py`:
```python
"""Phase 5: Report Generation — compile all phases into a markdown report.

This is a pure data assembly step — no LLM calls. It takes the accumulated
state from all previous phases and formats it into a structured report.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from langgraph_portfolio.projects.project_12_analyst.state import InvestmentState


REPORTS_DIR = Path(__file__).parent.parent / "reports"


def _format_list(items: list, prefix: str = "- ") -> str:
    """Format a list of items as markdown bullets."""
    if not items:
        return "_None identified_"
    return "\n".join(f"{prefix}{item}" for item in items)


def build_report_content(state: InvestmentState) -> str:
    """Build the full markdown report from state."""
    ticker = state["ticker"]
    company = state.get("company_name", ticker)
    today = date.today().isoformat()

    return f"""# Investment Analysis: {ticker} — {company}

**Date:** {today}

---

## Executive Summary

{state.get('investment_thesis', '_No thesis generated_')}

---

## Financial Data Overview

{state.get('data_summary', '_No data summary available_')}

---

## Profitability Analysis

{state.get('profitability_analysis', '_Not completed_')}

---

## Valuation

{state.get('valuation_analysis', '_Not completed_')}

---

## Growth Assessment

{state.get('growth_analysis', '_Not completed_')}

---

## Bull/Bear Debate

{state.get('debate_transcript', '_Debate not run_')}

### Key Disagreements

{_format_list(state.get('key_disagreements', []))}

---

## Risk Factors

{_format_list(state.get('risk_factors', []))}

## Catalysts

{_format_list(state.get('catalysts', []))}

## Confidence Level: {state.get('confidence_level', 'N/A')}

---

_Report generated by Investment Committee (Project 12)_
"""


def generate_report(state: InvestmentState) -> dict:
    """LangGraph node: generate and save the markdown report."""
    ticker = state["ticker"]
    today = date.today().isoformat()
    filename = f"{ticker}_{today}.md"

    content = build_report_content(state)

    # Ensure reports directory exists
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / filename
    report_path.write_text(content, encoding="utf-8")

    return {
        "report_path": str(report_path),
        "current_phase": "complete",
        "_report_content": content,  # For testing; not part of InvestmentState
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/test_p12_phases.py::ReportPhaseTest -v`
Expected: 1 test PASS

- [ ] **Step 5: Commit**

```bash
git add src/langgraph_portfolio/projects/project_12_analyst/phases/report.py \
        tests/test_p12_phases.py
git commit -m "feat(p12): add Phase 5 report generation"
```

---

### Task 8: Orchestrator — LangGraph Parent Graph

**Files:**
- Create: `src/langgraph_portfolio/projects/project_12_analyst/orchestrator.py`
- Modify: `tests/test_project_12_analyst.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_project_12_analyst.py`:
```python
class OrchestratorTest(unittest.TestCase):
    """Test that the parent graph compiles with all phases wired in."""

    def test_parent_graph_compiles(self) -> None:
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph_portfolio.projects.project_12_analyst.orchestrator import (
            build_analyst_graph,
        )

        checkpointer = MemorySaver()
        app = build_analyst_graph(checkpointer=checkpointer)
        self.assertIsNotNone(app)

    def test_route_after_interrupt_function(self) -> None:
        from langgraph_portfolio.projects.project_12_analyst.orchestrator import (
            route_after_review,
        )
        from langgraph_portfolio.projects.project_12_analyst.state import InvestmentState

        state: InvestmentState = {
            "ticker": "AAPL",
            "company_name": "Apple Inc.",
            "financials": {},
            "ratios": {},
            "estimates": {},
            "insider_trades": [],
            "grades": [],
            "earnings_transcripts": [],
            "data_summary": "",
            "profitability_analysis": "",
            "valuation_analysis": "",
            "growth_analysis": "",
            "analysis_summary": "",
            "debate_transcript": "",
            "key_disagreements": [],
            "investment_thesis": "",
            "risk_factors": [],
            "catalysts": [],
            "confidence_level": "",
            "report_path": "",
            "current_phase": "analysis",
            "human_feedback": "",
            "messages": [],
        }
        result = route_after_review(state)
        self.assertIn(result, ["analyze", "debate", "thesis", "report", "gather"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_project_12_analyst.py::OrchestratorTest -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement the orchestrator**

Create `src/langgraph_portfolio/projects/project_12_analyst/orchestrator.py`:
```python
"""Orchestrator — LangGraph parent graph wiring all 5 phases.

CONCEPT: Cross-framework composition
This is the heart of Project 12. The parent graph uses LangGraph to
orchestrate phases that internally use CrewAI (gathering), LangGraph
sub-graphs (analysis), AG2 (debate), and LlamaIndex (thesis).

Each phase is a node in the parent graph. Between phases, interrupt()
pauses execution so the user can review results and steer the next step.

The graph flow:
  gather → [interrupt] → analyze → [interrupt] → debate → [interrupt]
  → thesis → [interrupt] → report → END

The user can also skip phases or re-run them via human_feedback.
"""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from langgraph_portfolio.projects.project_12_analyst.state import InvestmentState
from langgraph_portfolio.projects.project_12_analyst.phases.gathering import gather_data
from langgraph_portfolio.projects.project_12_analyst.phases.profitability import (
    build_profitability_subgraph,
)
from langgraph_portfolio.projects.project_12_analyst.phases.valuation import (
    build_valuation_subgraph,
)
from langgraph_portfolio.projects.project_12_analyst.phases.growth import (
    build_growth_subgraph,
)
from langgraph_portfolio.projects.project_12_analyst.phases.debate import run_debate
from langgraph_portfolio.projects.project_12_analyst.phases.thesis import (
    build_thesis_subgraph,
)
from langgraph_portfolio.projects.project_12_analyst.phases.report import generate_report


# ── Phase-to-phase mapping ──

PHASE_ORDER = ["gathering", "analysis", "debate", "thesis", "report"]
PHASE_NEXT = {
    "gathering": "analyze",
    "analysis": "debate",
    "debate": "thesis",
    "thesis": "report",
}


# ── Review nodes (interrupt between phases) ──

def review_data(state: InvestmentState) -> dict:
    """Pause after data gathering for user review."""
    summary = state.get("data_summary", "No summary available")
    feedback = interrupt(
        f"── Data Gathering Complete ──\n\n{summary}\n\n"
        "What next? [continue / rerun / skip to <phase>]"
    )
    return {"human_feedback": feedback}


def review_analysis(state: InvestmentState) -> dict:
    """Pause after financial analysis for user review."""
    summary = state.get("analysis_summary", "No summary available")
    feedback = interrupt(
        f"── Financial Analysis Complete ──\n\n{summary}\n\n"
        "What next? [continue / rerun / skip to <phase>]"
    )
    return {"human_feedback": feedback}


def review_debate(state: InvestmentState) -> dict:
    """Pause after debate for user review."""
    disagreements = state.get("key_disagreements", [])
    disagreements_text = "\n".join(f"  - {d}" for d in disagreements)
    feedback = interrupt(
        f"── Bull/Bear Debate Complete ──\n\n"
        f"Key Disagreements:\n{disagreements_text}\n\n"
        "What next? [continue / rerun / skip to report]"
    )
    return {"human_feedback": feedback}


def review_thesis(state: InvestmentState) -> dict:
    """Pause after thesis for user review."""
    thesis = state.get("investment_thesis", "No thesis available")
    confidence = state.get("confidence_level", "?")
    feedback = interrupt(
        f"── Investment Thesis (Confidence: {confidence}) ──\n\n"
        f"{thesis[:2000]}\n\n"
        "What next? [continue to report / rerun / revise]"
    )
    return {"human_feedback": feedback}


# ── Analysis wrapper node ──

def analyze_financials(state: InvestmentState) -> dict:
    """Run all 3 analysis sub-graphs and combine results.

    Each sub-graph receives the data it needs from state and returns
    its analysis string. We combine them into analysis_summary.
    """
    # Build and run sub-graphs
    profitability_graph = build_profitability_subgraph()
    valuation_graph = build_valuation_subgraph()
    growth_graph = build_growth_subgraph()

    # Each sub-graph gets its input schema fields from state
    prof_input = {"financials": state["financials"], "ratios": state["ratios"]}
    val_input = {
        "financials": state["financials"],
        "ratios": state["ratios"],
        "estimates": state["estimates"],
    }
    growth_input = {
        "financials": state["financials"],
        "ratios": state["ratios"],
        "estimates": state["estimates"],
    }

    prof_result = profitability_graph.invoke(prof_input)
    val_result = valuation_graph.invoke(val_input)
    growth_result = growth_graph.invoke(growth_input)

    # Combine summaries
    profitability = prof_result.get("profitability_analysis", "")
    valuation = val_result.get("valuation_analysis", "")
    growth = growth_result.get("growth_analysis", "")

    analysis_summary = (
        f"## Profitability\n{profitability}\n\n"
        f"## Valuation\n{valuation}\n\n"
        f"## Growth\n{growth}"
    )

    return {
        "profitability_analysis": profitability,
        "valuation_analysis": valuation,
        "growth_analysis": growth,
        "analysis_summary": analysis_summary,
        "current_phase": "debate",
    }


# ── Routing ──

def route_after_review(state: InvestmentState) -> str:
    """Route to next phase based on current_phase and human_feedback.

    The user can say 'continue' (go to next phase), 'rerun' (repeat current),
    or 'skip to <phase>' (jump ahead).
    """
    feedback = (state.get("human_feedback") or "continue").strip().lower()
    current = state.get("current_phase", "gathering")

    # Handle skip commands
    if feedback.startswith("skip to"):
        target = feedback.replace("skip to", "").strip()
        phase_map = {
            "analysis": "analyze",
            "analyze": "analyze",
            "debate": "debate",
            "thesis": "thesis",
            "report": "report",
        }
        return phase_map.get(target, PHASE_NEXT.get(current, "report"))

    # Handle rerun
    if feedback == "rerun":
        phase_node_map = {
            "gathering": "gather",
            "analysis": "analyze",
            "debate": "debate",
            "thesis": "thesis",
        }
        return phase_node_map.get(current, "report")

    # Default: continue to next phase
    return PHASE_NEXT.get(current, "report")


# ── Build Parent Graph ──

def build_analyst_graph(
    checkpointer: BaseCheckpointSaver | None = None,
):
    """Build and compile the parent orchestration graph.

    Returns a compiled LangGraph app ready to invoke with InvestmentState.
    """
    graph = StateGraph(InvestmentState)

    # Phase nodes
    graph.add_node("gather", gather_data)
    graph.add_node("review_data", review_data)
    graph.add_node("analyze", analyze_financials)
    graph.add_node("review_analysis", review_analysis)
    graph.add_node("debate", run_debate)
    graph.add_node("review_debate", review_debate)
    graph.add_node("thesis", build_thesis_subgraph())
    graph.add_node("review_thesis", review_thesis)
    graph.add_node("report", generate_report)

    # Flow: gather → review → analyze → review → debate → review → thesis → review → report
    graph.set_entry_point("gather")
    graph.add_edge("gather", "review_data")
    graph.add_conditional_edges("review_data", route_after_review)
    graph.add_edge("analyze", "review_analysis")
    graph.add_conditional_edges("review_analysis", route_after_review)
    graph.add_edge("debate", "review_debate")
    graph.add_conditional_edges("review_debate", route_after_review)
    graph.add_edge("thesis", "review_thesis")
    graph.add_conditional_edges("review_thesis", route_after_review)
    graph.add_edge("report", END)

    return graph.compile(checkpointer=checkpointer)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/test_project_12_analyst.py -v`
Expected: All tests PASS (LLMFactory + State + Orchestrator)

- [ ] **Step 5: Commit**

```bash
git add src/langgraph_portfolio/projects/project_12_analyst/orchestrator.py \
        tests/test_project_12_analyst.py
git commit -m "feat(p12): add orchestrator wiring all 5 phases with interrupt() checkpoints"
```

---

### Task 9: CLI REPL

**Files:**
- Create: `src/langgraph_portfolio/projects/project_12_analyst/main.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Implement the CLI REPL**

Create `src/langgraph_portfolio/projects/project_12_analyst/main.py`:
```python
"""CLI REPL — co-pilot interface for the Investment Committee.

CONCEPT: REPL driving a LangGraph graph with interrupt()
This is the same pattern as P5's main.py. The REPL:
  1. Takes a ticker from the user
  2. Starts the LangGraph parent graph
  3. Streams until an interrupt() pauses execution
  4. Shows the interrupt message and prompts for feedback
  5. Resumes the graph with the user's feedback
  6. Repeats until the graph reaches END

The user can also type commands at any pause point:
  - 'continue' / 'next' → advance to next phase
  - 'rerun' → repeat the current phase
  - 'skip to <phase>' → jump ahead
  - 'report' → skip to report generation
  - 'quit' → exit
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from langgraph_portfolio.projects.project_12_analyst.orchestrator import (
    build_analyst_graph,
)
from langgraph_portfolio.projects.project_12_analyst.state import make_initial_state


BANNER = """
╔══════════════════════════════════════════════╗
║       Investment Committee v0.1              ║
║   Multi-Framework Financial Analyst          ║
║                                              ║
║   Commands:                                  ║
║     analyze <TICKER>  — start deep-dive      ║
║     continue / next   — advance phase        ║
║     rerun             — repeat current phase  ║
║     skip to <phase>   — jump ahead           ║
║     report            — generate report       ║
║     help              — show commands         ║
║     quit              — exit                  ║
╚══════════════════════════════════════════════╝
"""

HELP_TEXT = """
Available commands:
  analyze <TICKER>   Start a new deep-dive analysis
  continue / next    Advance to the next phase
  rerun              Re-run the current phase with new instructions
  skip to <phase>    Jump to: analysis, debate, thesis, report
  report             Skip directly to report generation
  help               Show this help text
  quit               Exit the program

Phases: gathering → analysis → debate → thesis → report
"""


def _stream_with_interrupts(app, input_value, config) -> dict | None:
    """Stream the graph, handling interrupt() pauses.

    Same pattern as P5's _stream_with_interrupts.
    """
    current_input = input_value
    final_state = None

    while True:
        for chunk in app.stream(current_input, config=config, stream_mode="updates"):
            for node_name, updates in chunk.items():
                phase = updates.get("current_phase", "")
                if phase:
                    print(f"\n  [Phase: {phase}]")

        state_snapshot = app.get_state(config)

        if state_snapshot.next:
            # Graph is paused at an interrupt
            for task in state_snapshot.tasks:
                if hasattr(task, "interrupts") and task.interrupts:
                    interrupt_value = task.interrupts[0].value
                    print(f"\n{interrupt_value}")
                    user_input = input("\n> ").strip()

                    if user_input.lower() == "quit":
                        print("Exiting analysis.")
                        return state_snapshot.values

                    current_input = Command(resume=user_input)
                    break
        else:
            final_state = state_snapshot.values
            break

    return final_state


def run_analysis(ticker: str) -> None:
    """Run a full deep-dive analysis for a ticker."""
    print(f"\nStarting analysis for {ticker.upper()}...\n")

    checkpointer = MemorySaver()
    app = build_analyst_graph(checkpointer=checkpointer)

    initial_state = make_initial_state(ticker)
    config = {"configurable": {"thread_id": f"analysis-{ticker.upper()}"}}

    final_state = _stream_with_interrupts(app, initial_state, config)

    if final_state:
        report_path = final_state.get("report_path", "")
        if report_path:
            print(f"\nReport saved to: {report_path}")
        else:
            print("\nAnalysis complete (no report generated).")


def main() -> None:
    """Entry point for the CLI REPL."""
    print(BANNER)

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        parts = user_input.lower().split()
        command = parts[0]

        if command == "quit":
            print("Goodbye!")
            break
        elif command == "help":
            print(HELP_TEXT)
        elif command == "analyze" and len(parts) > 1:
            ticker = parts[1].upper()
            run_analysis(ticker)
        elif command == "analyze":
            print("Usage: analyze <TICKER>  (e.g., analyze AAPL)")
        else:
            print(f"Unknown command: '{user_input}'. Type 'help' for available commands.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add entry point to pyproject.toml**

Add to `[tool.poetry.scripts]` section in `pyproject.toml`:
```toml
project-12-analyst = "langgraph_portfolio.projects.project_12_analyst.main:main"
```

- [ ] **Step 3: Run a quick import test**

Run: `poetry run python -c "from langgraph_portfolio.projects.project_12_analyst.main import main; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/langgraph_portfolio/projects/project_12_analyst/main.py pyproject.toml
git commit -m "feat(p12): add CLI REPL entry point"
```

---

### Task 10: MCP Server

**Files:**
- Create: `src/langgraph_portfolio/projects/project_12_analyst/server.py`
- Modify: `tests/test_project_12_analyst.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_project_12_analyst.py`:
```python
class MCPServerTest(unittest.TestCase):
    """Test that the MCP server initializes and has expected tools."""

    def test_server_creates(self) -> None:
        from langgraph_portfolio.projects.project_12_analyst.server import server

        self.assertIsNotNone(server)
        self.assertEqual(server.name, "investment-committee")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_project_12_analyst.py::MCPServerTest -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement the MCP server**

Create `src/langgraph_portfolio/projects/project_12_analyst/server.py`:
```python
"""MCP Server — expose Investment Committee as tools/resources/prompts.

CONCEPT: FastMCP server (from P11)
This server exposes the analysis system for use from Claude Code,
Claude Desktop, or any MCP client. It provides:
  - Tools: analyze_company, continue_analysis, get_phase_results
  - Resources: reports://{ticker}, reports://list
  - Prompts: deep-dive, quick-valuation

The server manages analysis sessions internally using the same
LangGraph orchestrator + MemorySaver checkpointer.
"""

from __future__ import annotations

import json
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from langgraph_portfolio.projects.project_12_analyst.orchestrator import (
    build_analyst_graph,
)
from langgraph_portfolio.projects.project_12_analyst.state import make_initial_state


server = FastMCP("investment-committee")

# In-memory session management
_checkpointer = MemorySaver()
_app = build_analyst_graph(checkpointer=_checkpointer)

REPORTS_DIR = Path(__file__).parent / "reports"


# ── Tools ──

@server.tool()
async def analyze_company(ticker: str) -> str:
    """Start a full deep-dive analysis for a company.

    Begins Phase 1 (data gathering) and returns the data summary.
    Use continue_analysis to advance to the next phase.
    """
    ticker = ticker.upper()
    initial_state = make_initial_state(ticker)
    config = {"configurable": {"thread_id": f"mcp-{ticker}"}}

    # Run until first interrupt
    for chunk in _app.stream(initial_state, config=config, stream_mode="updates"):
        pass

    state = _app.get_state(config)

    # Check for interrupt
    for task in state.tasks:
        if hasattr(task, "interrupts") and task.interrupts:
            return task.interrupts[0].value

    return f"Analysis of {ticker} started. Use continue_analysis to proceed."


@server.tool()
async def continue_analysis(ticker: str, feedback: str = "continue") -> str:
    """Advance the analysis to the next phase.

    Optionally pass feedback to steer the analysis (e.g., 'skip to thesis',
    'rerun', or specific instructions).
    """
    ticker = ticker.upper()
    config = {"configurable": {"thread_id": f"mcp-{ticker}"}}

    state = _app.get_state(config)
    if not state.next:
        return f"No active analysis for {ticker}. Use analyze_company first."

    # Resume with feedback
    for chunk in _app.stream(
        Command(resume=feedback), config=config, stream_mode="updates"
    ):
        pass

    # Check for next interrupt or completion
    state = _app.get_state(config)
    for task in state.tasks:
        if hasattr(task, "interrupts") and task.interrupts:
            return task.interrupts[0].value

    # Analysis complete
    report_path = state.values.get("report_path", "")
    if report_path:
        return f"Analysis complete. Report saved to: {report_path}"
    return "Analysis phase complete."


@server.tool()
async def get_phase_results(ticker: str, phase: str) -> str:
    """Retrieve results from a specific completed phase.

    Valid phases: gathering, analysis, debate, thesis, report.
    """
    ticker = ticker.upper()
    config = {"configurable": {"thread_id": f"mcp-{ticker}"}}

    state = _app.get_state(config)
    if not state.values:
        return f"No analysis found for {ticker}."

    values = state.values
    phase_data = {
        "gathering": values.get("data_summary", "Phase not completed"),
        "analysis": values.get("analysis_summary", "Phase not completed"),
        "debate": (
            f"Transcript:\n{values.get('debate_transcript', 'N/A')}\n\n"
            f"Key Disagreements:\n"
            + "\n".join(f"- {d}" for d in values.get("key_disagreements", []))
        ),
        "thesis": values.get("investment_thesis", "Phase not completed"),
        "report": values.get("report_path", "No report generated"),
    }

    return phase_data.get(phase, f"Unknown phase: {phase}")


# ── Resources ──

@server.resource("reports://list")
async def list_reports() -> str:
    """List all completed analysis reports."""
    if not REPORTS_DIR.exists():
        return json.dumps([])

    reports = []
    for f in sorted(REPORTS_DIR.glob("*.md")):
        reports.append({"ticker": f.stem.split("_")[0], "file": f.name})

    return json.dumps(reports, indent=2)


@server.resource("reports://{ticker}")
async def get_report(ticker: str) -> str:
    """Get the full analysis report for a ticker."""
    ticker = ticker.upper()
    if not REPORTS_DIR.exists():
        return f"No reports directory found."

    matches = list(REPORTS_DIR.glob(f"{ticker}_*.md"))
    if not matches:
        return f"No report found for {ticker}."

    # Return most recent
    latest = sorted(matches)[-1]
    return latest.read_text(encoding="utf-8")


# ── Prompts ──

@server.prompt()
async def deep_dive(ticker: str) -> str:
    """Run a complete investment analysis on a ticker.

    Walks through all 5 phases: data gathering, financial analysis,
    bull/bear debate, thesis synthesis, and report generation.
    """
    return (
        f"Please run a complete investment analysis on {ticker.upper()}.\n\n"
        "Steps:\n"
        f"1. Call analyze_company with ticker='{ticker.upper()}'\n"
        "2. Review the data summary\n"
        "3. Call continue_analysis to advance through each phase\n"
        "4. Review results at each checkpoint\n"
        "5. The final report will be saved automatically\n"
    )


@server.prompt()
async def quick_valuation(ticker: str) -> str:
    """Get a quick valuation estimate — skip debate, just DCF + comps."""
    return (
        f"Please do a quick valuation of {ticker.upper()}.\n\n"
        "Steps:\n"
        f"1. Call analyze_company with ticker='{ticker.upper()}'\n"
        "2. At the data review, say 'continue'\n"
        "3. At the analysis review, say 'skip to report'\n"
        "4. This skips the debate and thesis for a faster result\n"
    )


def main() -> None:
    """Run the MCP server (stdio transport)."""
    server.run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/test_project_12_analyst.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/langgraph_portfolio/projects/project_12_analyst/server.py \
        tests/test_project_12_analyst.py
git commit -m "feat(p12): add MCP server with tools, resources, and prompts"
```

---

### Task 11: Gitignore Reports + Update CLAUDE.md + Final Smoke Test

**Files:**
- Create: `src/langgraph_portfolio/projects/project_12_analyst/reports/.gitkeep`
- Modify: `.gitignore`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add .gitkeep and update .gitignore**

Create `src/langgraph_portfolio/projects/project_12_analyst/reports/.gitkeep` (empty file).

Add to `.gitignore`:
```
# P12 generated reports
src/langgraph_portfolio/projects/project_12_analyst/reports/*.md
```

- [ ] **Step 2: Update CLAUDE.md progress table**

Add Project 12 to the progress table in `CLAUDE.md`:
```
| 12 | Investment Committee | All 5 | DONE | `orchestrator.py`, `phases/`, `server.py` | cross-framework composition, MCP server, co-pilot HITL |
```

Add to the Commands section:
```bash
poetry run project-12-analyst
```

Update the Tech Stack section to mention httpx:
```
- httpx for FMP REST API calls
```

- [ ] **Step 3: Run the full test suite**

Run: `poetry run pytest tests/test_project_12_analyst.py tests/test_p12_fmp_tools.py tests/test_p12_phases.py -v`
Expected: All tests PASS

- [ ] **Step 4: Run existing test suite to verify no regressions**

Run: `poetry run pytest -v`
Expected: All existing tests still PASS

- [ ] **Step 5: Commit**

```bash
git add .gitignore CLAUDE.md \
        src/langgraph_portfolio/projects/project_12_analyst/reports/.gitkeep
git commit -m "feat(p12): finalize project — update docs, gitignore reports"
```

---

## Summary

| Task | Component | Framework | Files |
|------|-----------|-----------|-------|
| 1 | Skeleton + LLM + State | — | `__init__.py`, `llm.py`, `state.py` |
| 2 | FMP API Client | httpx | `tools/fmp.py` |
| 3 | Phase 1: Data Gathering | CrewAI | `phases/gathering.py` |
| 4 | Phase 2: Financial Analysis | LangGraph | `phases/profitability.py`, `valuation.py`, `growth.py` |
| 5 | Phase 3: Bull/Bear Debate | AG2 | `phases/debate.py` |
| 6 | Phase 4: Thesis Synthesis | LangGraph + LlamaIndex | `phases/thesis.py` |
| 7 | Phase 5: Report Generation | — | `phases/report.py` |
| 8 | Orchestrator | LangGraph | `orchestrator.py` |
| 9 | CLI REPL | — | `main.py` |
| 10 | MCP Server | FastMCP | `server.py` |
| 11 | Finalize | — | `.gitignore`, `CLAUDE.md` |
