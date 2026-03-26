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


# -- CrewAI Agents --
# CONCEPT: Agent specialization
# Each agent has a distinct domain of expertise. This mirrors how real
# investment teams work: a financial analyst reads 10-Ks, a market intel
# analyst tracks consensus, and an earnings analyst dissects management tone.
# CrewAI's sequential process means each agent sees the previous agent's output,
# building progressively richer context — like a relay race of analysis.

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

    CONCEPT: Crew as a reusable unit
    build_gathering_crew() returns a Crew object that can be inspected (for tests)
    or kicked off (for real runs). The task descriptions are generic placeholders
    here — gather_data() injects actual FMP data before calling crew.kickoff().

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

    # CONCEPT: Sequential process
    # Process.sequential runs tasks in order: financial -> market -> transcript.
    # Each agent sees the previous agent's output as context. This means the
    # market intel agent can reference financial findings, and the transcript
    # analyst can connect management's words to both financial and market data.
    # memory=False because we don't need cross-session memory for a one-shot analysis.
    return Crew(
        agents=[financial_data_agent, market_intel_agent, transcript_agent],
        tasks=[financial_task, market_task, transcript_task],
        process=Process.sequential,
        memory=False,
        verbose=True,
    )


async def _fetch_fmp_data(ticker: str) -> dict[str, Any]:
    """Fetch all data from FMP for the gathering phase.

    CONCEPT: Concurrent I/O with asyncio.gather
    We fire all 12 FMP requests at once using asyncio.gather. This is MUCH
    faster than sequential requests — if each takes ~200ms, sequential would
    be ~2.4s while concurrent is ~200ms (limited by the slowest request).
    """
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
    # CONCEPT: Conditional async calls
    # transcript_dates returns a list of [year, quarter] pairs. We grab the
    # 2 most recent so the earnings analyst can compare quarter-over-quarter.
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
    """JSON-serialize data, truncating if too long for LLM context.

    CONCEPT: Context window management
    Local LLMs have limited context windows. We cap each data section at
    6000 chars (~1500 tokens) to avoid blowing the context budget. The
    truncation marker tells the agent that data was cut, so it can note
    that caveat in its analysis.
    """
    text = json.dumps(data, indent=2, default=str)
    if len(text) > max_chars:
        return text[:max_chars] + "\n\n[Data truncated for context limits]"
    return text


def gather_data(state: InvestmentState) -> dict:
    """LangGraph node: fetch FMP data and run CrewAI gathering crew.

    CONCEPT: Node function pattern
    This is a standard LangGraph node — it takes state, does work, returns
    a dict of state updates. The twist is that internally it orchestrates
    both async I/O (FMP fetching) and a full CrewAI crew (LLM reasoning).

    This is the function the parent graph calls as a node. It:
    1. Fetches all FMP data (async I/O)
    2. Builds a CrewAI crew with the data as context
    3. Runs the crew (LLM reasoning)
    4. Returns state updates
    """
    ticker = state["ticker"]

    # Step 1: Fetch all data from FMP
    # CONCEPT: asyncio.run inside a sync node
    # LangGraph nodes are sync functions, but FMPClient is async.
    # asyncio.run() bridges the gap — it creates a new event loop, runs our
    # coroutine, then closes the loop. This is safe because LangGraph nodes
    # run in their own thread.
    raw_data = asyncio.run(_fetch_fmp_data(ticker))

    # Extract company name from profile
    company_name = ""
    if raw_data["profile"] and isinstance(raw_data["profile"], list):
        company_name = raw_data["profile"][0].get("companyName", ticker)

    # Step 2: Build crew with data injected into task descriptions
    crew = build_gathering_crew(ticker)

    # CONCEPT: Data injection into task descriptions
    # Instead of giving agents tools (which Ollama can't use via CrewAI),
    # we append the raw data directly to each task's description field.
    # The agent sees it as part of its instructions — "here's the data,
    # now analyze it." This is the same pattern we used in P6.
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
    # CONCEPT: crew.kickoff() is blocking
    # Unlike LangGraph (which can be async), CrewAI's kickoff() is synchronous.
    # It runs all tasks in sequence, each agent thinking and producing output.
    # The result object contains .raw (full text) and per-task outputs.
    result = crew.kickoff()

    # Step 4: Build data summary from crew output
    data_summary = result.raw if hasattr(result, "raw") else str(result)

    # Step 5: Return state updates
    # CONCEPT: Partial state updates
    # We only return the fields we want to update. LangGraph merges these
    # into the existing state — fields we don't mention stay unchanged.
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
