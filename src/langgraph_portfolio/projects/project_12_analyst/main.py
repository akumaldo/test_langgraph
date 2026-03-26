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

from dotenv import load_dotenv

load_dotenv()

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
        # -----------------------------------------------------------
        # CONCEPT: stream_mode="updates"
        # Each chunk is {node_name: state_updates} — we print the
        # current_phase so the user sees progress as nodes execute.
        # -----------------------------------------------------------
        for chunk in app.stream(current_input, config=config, stream_mode="updates"):
            for node_name, updates in chunk.items():
                if isinstance(updates, dict):
                    phase = updates.get("current_phase", "")
                    if phase:
                        print(f"\n  [Phase: {phase}]")

        # -----------------------------------------------------------
        # CONCEPT: interrupt-based human-in-the-loop
        # After streaming stops, we check if the graph paused at an
        # interrupt(). If so, state_snapshot.next will be non-empty
        # and each task will have an .interrupts list with the value
        # the node passed to interrupt().
        # -----------------------------------------------------------
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

                    # ---------------------------------------------------
                    # CONCEPT: Command(resume=...)
                    # To resume a graph paused at interrupt(), we pass a
                    # Command with the user's response. The node that
                    # called interrupt() receives this as the return
                    # value of interrupt().
                    # ---------------------------------------------------
                    current_input = Command(resume=user_input)
                    break
        else:
            # No more nodes to run — graph reached END
            final_state = state_snapshot.values
            break

    return final_state


def run_analysis(ticker: str) -> None:
    """Run a full deep-dive analysis for a ticker."""
    print(f"\nStarting analysis for {ticker.upper()}...\n")

    # -----------------------------------------------------------
    # CONCEPT: MemorySaver checkpointer
    # Same as P5 — in-memory checkpointer lets us pause/resume
    # the graph across interrupt() calls within a single session.
    # For persistence across sessions, swap to SqliteSaver.
    # -----------------------------------------------------------
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
