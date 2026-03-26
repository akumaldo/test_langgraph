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


# -- Sub-graph State --

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


# -- Node Functions --

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


# -- Build Sub-graph --

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
