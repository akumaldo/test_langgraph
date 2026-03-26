"""
Project 8 — BeeAI Research Agent entry point.

Run with: poetry run python -m langgraph_portfolio.projects.project_8_beeai_research.main
"""

import asyncio
from .agent import run_research


def main():
    """Run the BeeAI research agent with a sample question."""
    asyncio.run(run_research("How do LangGraph, CrewAI, and LlamaIndex differ?"))


if __name__ == "__main__":
    main()
