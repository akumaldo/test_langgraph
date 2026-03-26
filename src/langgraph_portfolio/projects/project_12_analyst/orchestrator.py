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


# -- Phase-to-phase mapping --
# CONCEPT: Phase routing table
# Each phase has a name and a corresponding next-node in the graph.
# This keeps routing logic centralized instead of scattered across nodes.

PHASE_ORDER = ["gathering", "analysis", "debate", "thesis", "report"]
PHASE_NEXT = {
    "gathering": "analyze",
    "analysis": "debate",
    "debate": "thesis",
    "thesis": "report",
}


# -- Review nodes (interrupt between phases) --
# CONCEPT: interrupt() for human-in-the-loop (from P5)
# Each review node calls interrupt() which pauses graph execution and
# returns a prompt to the user. When the user resumes with Command(resume=...),
# the interrupt() call returns their feedback string. This lets users
# inspect results at each stage and decide what to do next.


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


# -- Analysis wrapper node --
# CONCEPT: Parallel sub-graphs inside a single node
# The analyze_financials node runs 3 independent LangGraph sub-graphs
# (profitability, valuation, growth). Each sub-graph was compiled in its
# own module with input/output schemas for state isolation (from P5).
# We invoke them sequentially here, but they could be parallelized with
# asyncio in a production system.


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


# -- Routing --
# CONCEPT: Conditional edges with human steering (from P1, P5)
# After each interrupt(), the user's feedback determines the next node.
# This single routing function handles all review nodes because the
# logic is the same: parse feedback → map to node name.


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


# -- Build Parent Graph --
# CONCEPT: StateGraph as orchestration backbone
# The parent graph ties everything together. Each framework runs inside
# its own node — CrewAI in gather, LangGraph sub-graphs in analyze,
# AG2 in debate, LlamaIndex in thesis, plain Python in report.
# LangGraph handles state flow, checkpointing, and human-in-the-loop
# across all of them.


def build_analyst_graph(
    checkpointer: BaseCheckpointSaver | None = None,
):
    """Build and compile the parent orchestration graph.

    CONCEPT: Graph compilation with checkpointer
    The checkpointer (e.g. MemorySaver) enables interrupt/resume.
    Without it, interrupt() would raise an error because there's nowhere
    to persist state between pauses.

    Returns a compiled LangGraph app ready to invoke with InvestmentState.
    """
    graph = StateGraph(InvestmentState)

    # Phase nodes — each wraps a different framework
    graph.add_node("gather", gather_data)           # CrewAI crew
    graph.add_node("review_data", review_data)      # interrupt()
    graph.add_node("analyze", analyze_financials)    # 3 LangGraph sub-graphs
    graph.add_node("review_analysis", review_analysis)
    graph.add_node("debate", run_debate)             # AG2 GroupChat
    graph.add_node("review_debate", review_debate)
    graph.add_node("thesis", build_thesis_subgraph())  # LlamaIndex + LangGraph
    graph.add_node("review_thesis", review_thesis)
    graph.add_node("report", generate_report)        # Plain Python

    # Flow: gather → review → analyze → review → debate → review
    #       → thesis → review → report → END
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
