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


# -- Sub-graph State --

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


# -- Node Functions --

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
