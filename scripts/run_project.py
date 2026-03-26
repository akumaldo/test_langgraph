from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any, Callable


DEFAULT_APP = "langgraph_portfolio.projects.project_1_chatbot.main:main"
ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_entrypoint(spec: str) -> Callable[[], Any]:
    module_name, _, attr = spec.partition(":")
    module = importlib.import_module(module_name)
    entrypoint_name = attr or "main"
    entrypoint = getattr(module, entrypoint_name)
    if not callable(entrypoint):  # pragma: no cover - defensive guard
        raise TypeError(f"{spec} does not resolve to a callable entrypoint")
    return entrypoint


def main() -> Any:
    spec = os.getenv("APP_MODULE", DEFAULT_APP)
    return load_entrypoint(spec)()


if __name__ == "__main__":
    raise SystemExit(main())
