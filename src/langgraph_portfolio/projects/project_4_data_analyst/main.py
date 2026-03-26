from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    """Run the data analyst.

    Interactive mode (in a terminal):
        python -m langgraph_portfolio.projects.project_4_data_analyst.main

    Single question mode (non-interactive):
        python -m langgraph_portfolio.projects.project_4_data_analyst.main --question "What affects grades?"
    """
    from .langgraph_data_analyst import interactive_loop, single_question

    csv_path = str(Path(__file__).parent / "student-mat.csv")

    if "--question" in sys.argv:
        idx = sys.argv.index("--question")
        question = sys.argv[idx + 1]
        single_question(csv_path, question)
    else:
        interactive_loop(csv_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
