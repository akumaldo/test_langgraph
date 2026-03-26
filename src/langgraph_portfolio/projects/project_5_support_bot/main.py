from __future__ import annotations

import sys

from .langgraph_support_bot import interactive_loop, single_issue


def main() -> int:
    """Entry point for the customer support bot.

    Supports two modes:
      poetry run project-5-capstone                              → interactive
      poetry run project-5-capstone --issue "I was charged twice" → single issue
    """
    if "--issue" in sys.argv:
        idx = sys.argv.index("--issue")
        if idx + 1 < len(sys.argv):
            single_issue(sys.argv[idx + 1])
        else:
            print("Error: --issue requires an argument")
            return 1
    else:
        interactive_loop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
