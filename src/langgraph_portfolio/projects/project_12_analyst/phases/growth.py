"""Phase 2c: Growth Analysis — LangGraph sub-graph.

Flow: analyze_revenue_growth → analyze_earnings_growth → assess_momentum → summarize
"""

from __future__ import annotations

import json
from typing import TypedDict

from langgraph.graph import END, StateGraph

from langgraph_portfolio.projects.project_12_analyst.llm import get_llm


# -- Sub-graph State --

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


# -- Node Functions --

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
