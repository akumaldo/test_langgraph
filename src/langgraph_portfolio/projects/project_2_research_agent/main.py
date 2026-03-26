from __future__ import annotations

from json import dumps

from .graph import run_research_agent


def main(query: str | None = None) -> int:
    result = run_research_agent(query or "Explain how LangGraph, LlamaIndex, and CrewAI differ.")
    print(dumps(result.model_dump(), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

