"""
PROJECT 10 — FRAMEWORK SHOWDOWN: ENTRY POINT
==============================================

Thin wrapper that imports and runs framework implementations from
src/langgraph_portfolio/projects/project_10_showdown/

Usage:
  # Run a specific framework
  poetry run python project_10_showdown/main.py --framework langgraph
  poetry run python project_10_showdown/main.py --framework crewai
  poetry run python project_10_showdown/main.py --framework ag2
  poetry run python project_10_showdown/main.py --framework beeai
  poetry run python project_10_showdown/main.py --framework llamaindex

  # Generate comparison report (after running frameworks)
  poetry run python project_10_showdown/main.py --compare
"""

import argparse
import sys
import time


FRAMEWORKS = ["langgraph", "crewai", "ag2", "beeai", "llamaindex"]


def run_framework(name: str):
    """Import and run a specific framework implementation."""
    if name == "langgraph":
        from langgraph_portfolio.projects.project_10_showdown.langgraph_version import run
    elif name == "crewai":
        from langgraph_portfolio.projects.project_10_showdown.crewai_version import run
    elif name == "ag2":
        from langgraph_portfolio.projects.project_10_showdown.ag2_version import run
    elif name == "beeai":
        from langgraph_portfolio.projects.project_10_showdown.beeai_version import run
    elif name == "llamaindex":
        from langgraph_portfolio.projects.project_10_showdown.llamaindex_version import run
    else:
        print(f"Unknown framework: {name}")
        print(f"Available: {', '.join(FRAMEWORKS)}")
        sys.exit(1)

    try:
        run()
    except Exception as e:
        # Save error result so compare.py can report it
        from langgraph_portfolio.projects.project_10_showdown.problem import save_result
        error_output = {
            "summary": "",
            "findings": [],
            "citations": [],
            "framework": name,
            "structured_output_success": False,
            "error": str(e),
        }
        save_result(name, error_output, 0.0)
        print(f"\nERROR running {name}: {e}")
        raise


def run_compare():
    """Generate the comparison report."""
    from langgraph_portfolio.projects.project_10_showdown.compare import run
    run()


def main():
    parser = argparse.ArgumentParser(description="Project 10 — Framework Showdown")
    parser.add_argument(
        "--framework", "-f",
        choices=FRAMEWORKS,
        help="Which framework to run",
    )
    parser.add_argument(
        "--compare", "-c",
        action="store_true",
        help="Generate comparison report from saved results",
    )
    args = parser.parse_args()

    if not args.framework and not args.compare:
        parser.print_help()
        sys.exit(1)

    if args.compare:
        run_compare()
    elif args.framework:
        run_framework(args.framework)


if __name__ == "__main__":
    main()
