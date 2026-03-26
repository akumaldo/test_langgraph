from __future__ import annotations

from json import dumps
from typing import Iterable

from .graph import run_chatbot


def main(messages: Iterable[str] | None = None) -> int:
    demo_messages = list(messages or ["Can you compare LangGraph and CrewAI?"])
    result = run_chatbot(demo_messages)
    print(dumps(result.model_dump(), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

