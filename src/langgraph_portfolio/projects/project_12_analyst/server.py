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


# -- Tools --

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


# -- Resources --

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


# -- Prompts --

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
