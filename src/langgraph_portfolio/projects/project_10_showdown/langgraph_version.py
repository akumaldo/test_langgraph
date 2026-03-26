"""
PROJECT 10 — FRAMEWORK SHOWDOWN: LANGGRAPH VERSION
====================================================

This is the LangGraph implementation of the research assistant.
It reuses patterns from P2 (research agent):
  - @tool for tool definition
  - bind_tools() + ToolNode for the agent loop
  - Conditional edges for loop control
  - with_structured_output() for Pydantic output

NEW IN P10:
  - Two-phase graph: agent loop (research) + report node (structured output)
  - Why two phases? bind_tools() and with_structured_output() can't be on
    the same LLM call. The research phase uses tools, the report phase
    uses structured output. The graph separates these concerns cleanly.

GRAPH FLOW:
  START → research → should_continue?
           ↑              |
           └── tools ←── YES (has tool_calls)
                          |
                      NO (done searching)
                          ↓
                   report (structured output) → END
"""

import time

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from typing import Annotated
from typing_extensions import TypedDict

from langgraph_portfolio.projects.project_10_showdown.problem import (
    RESEARCH_QUESTION,
    ResearchReport,
    search_knowledge_base,
    save_result,
)


# ── MODEL CONFIG ─────────────────────────────────────────────────────────
# qwen3.5:2b for all frameworks — fair comparison, no timeout issues.
LLM_MODEL = "qwen3.5:2b"


# ── TOOL DEFINITION ─────────────────────────────────────────────────────
# Wrap our shared search function as a LangGraph @tool.
#
# HOW @tool WORKS (recap from P2):
#   - Function name → tool name
#   - Docstring → tool description (the LLM reads this to decide when to call it)
#   - Type hints → parameter schema (auto-inferred by LangChain)
#   - Returns a string (tool results are always strings in LangChain)

@tool
def search_kb(query: str) -> str:
    """Search the knowledge base about AI orchestration in industry.
    Use this to find information about how AI orchestration is used
    in different industries like finance, healthcare, manufacturing,
    and customer service.
    """
    results = search_knowledge_base(query)
    # Format results as text for the LLM to read
    formatted = []
    for doc in results:
        formatted.append(
            f"[{doc['title']}] (Source: {doc['source']})\n{doc['content']}"
        )
    return "\n\n---\n\n".join(formatted)


tools = [search_kb]


# ── STATE ────────────────────────────────────────────────────────────────
# TypedDict with messages — LangGraph's standard pattern.
# add_messages reducer appends new messages (doesn't replace).

class ShowdownState(TypedDict):
    messages: Annotated[list, add_messages]


# ── NODES ────────────────────────────────────────────────────────────────
# Two LLM instances:
#   - research_llm: has tools bound (for searching)
#   - report_llm: has structured output (for the final report)

research_llm = ChatOllama(model=LLM_MODEL, temperature=0).bind_tools(tools)
report_llm = ChatOllama(model=LLM_MODEL, temperature=0).with_structured_output(
    ResearchReport
)


def research_node(state: ShowdownState) -> dict:
    """The research agent — calls the LLM with tools available.

    The LLM decides whether to search the KB or produce a final answer.
    If it returns tool_calls, the graph loops to the ToolNode.
    If it doesn't, the graph moves to the report node.
    """
    response = research_llm.invoke(state["messages"])
    return {"messages": [response]}


def report_node(state: ShowdownState) -> dict:
    """Produce the final structured report.

    Takes all the research gathered so far (in messages) and produces
    a ResearchReport via with_structured_output(). This is a separate
    node because structured output and tool binding can't coexist on
    the same LLM call.
    """
    # Build a focused prompt for the report
    report_prompt = SystemMessage(content=(
        "You are a research report writer. Based on the research conversation above, "
        "produce a structured research report. Include:\n"
        "- summary: 2-3 sentence overview of the main uses of AI orchestration\n"
        "- findings: list of key findings as bullet points\n"
        "- citations: list of document titles/sources that were used\n"
        "- framework: 'langgraph'\n"
        "Be concise and factual."
    ))
    messages = state["messages"] + [report_prompt]
    report = report_llm.invoke(messages)
    return {"messages": [HumanMessage(content=report.model_dump_json())]}


# ── CONDITIONAL EDGE ─────────────────────────────────────────────────────
# Decides whether to loop back to tools or move to the report.
#
# HOW IT WORKS (recap from P2):
#   The LLM's response contains tool_calls if it wants to use a tool.
#   If tool_calls is empty, the LLM is done researching.

def should_continue(state: ShowdownState) -> str:
    """Check if the LLM wants to call more tools."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "report"


# ── GRAPH ASSEMBLY ───────────────────────────────────────────────────────
# Wire up: START → research → conditional → tools (loop) or report → END

def build_graph():
    graph = StateGraph(ShowdownState)

    # Add nodes
    graph.add_node("research", research_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("report", report_node)

    # Edges
    graph.set_entry_point("research")
    graph.add_conditional_edges("research", should_continue, {
        "tools": "tools",
        "report": "report",
    })
    graph.add_edge("tools", "research")  # loop back after tool execution
    graph.add_edge("report", END)

    return graph.compile()


# ── RUN ──────────────────────────────────────────────────────────────────

def run():
    """Run the LangGraph research assistant and save results."""
    print("=" * 60)
    print("FRAMEWORK SHOWDOWN — LangGraph Version")
    print("=" * 60)
    print(f"\nQuestion: {RESEARCH_QUESTION}\n")

    graph = build_graph()

    # System message sets the research assistant role
    system = SystemMessage(content=(
        "You are a research assistant. Use the search_kb tool to find information "
        "about AI orchestration in industry. Search for relevant information, then "
        "provide your findings. You MUST use the search tool at least once before "
        "answering."
    ))
    human = HumanMessage(content=RESEARCH_QUESTION)

    start = time.time()
    result = graph.invoke({"messages": [system, human]})
    elapsed = time.time() - start

    # The last message contains the structured report as JSON
    last_msg = result["messages"][-1]
    print(f"\n{'─' * 40}")
    print(f"Result (elapsed: {elapsed:.1f}s):")
    print(last_msg.content)

    # Parse and save
    import json
    try:
        output = json.loads(last_msg.content)
        output["structured_output_success"] = True
    except (json.JSONDecodeError, Exception):
        output = {
            "summary": last_msg.content,
            "findings": [],
            "citations": [],
            "framework": "langgraph",
            "structured_output_success": False,
            "raw_text": last_msg.content,
        }

    path = save_result("langgraph", output, elapsed)
    print(f"\nResult saved to: {path}")


if __name__ == "__main__":
    run()
