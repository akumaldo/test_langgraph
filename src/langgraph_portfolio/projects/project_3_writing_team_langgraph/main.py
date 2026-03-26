from __future__ import annotations

from json import dumps

from .graph import run_writing_team


def main(topic: str | None = None) -> int:
    result = run_writing_team(topic or "LangGraph versus CrewAI for portfolio agents")
    print(dumps(result.model_dump(), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

