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

    # -- Identity --
    ticker: str
    company_name: str

    # -- Phase 1: Data Gathering (CrewAI output) --
    financials: dict          # income, balance sheet, cash flow (3-5 years)
    ratios: dict              # key metrics + TTM
    estimates: dict           # analyst estimates + price targets
    insider_trades: list      # recent insider activity
    grades: list              # analyst grades + changes
    earnings_transcripts: list  # raw transcript text
    data_summary: str         # human-readable summary for review

    # -- Phase 2: Financial Analysis (LangGraph sub-graphs output) --
    profitability_analysis: str
    valuation_analysis: str   # DCF result, comps, fair value range
    growth_analysis: str
    analysis_summary: str     # consolidated for review

    # -- Phase 3: Debate (AG2 output) --
    debate_transcript: str    # full bull/bear exchange
    key_disagreements: list   # moderator-extracted points of contention

    # -- Phase 4: Thesis (LangGraph + RAG output) --
    investment_thesis: str    # final synthesized thesis
    risk_factors: list
    catalysts: list
    confidence_level: str     # high / medium / low

    # -- Phase 5: Report --
    report_path: str          # saved file location

    # -- Control --
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
