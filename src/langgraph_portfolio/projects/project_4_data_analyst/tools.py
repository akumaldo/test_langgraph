"""
Project 4 — Tools for the data analyst agent.

In P2 (Research Agent), we created a search_documents tool:
    @tool
    def search_documents(query: str) -> str:
        ...

That tool was READ-ONLY — it looked up text and returned it.

Here we have tools that EXECUTE CODE. This is a bigger deal:
  - search_documents: "find me info" (safe, read-only)
  - execute_python: "run this code" (powerful, can fail, can modify state)

These aren't LangChain @tool decorated functions because we're NOT using
the agent loop pattern from P2 (where the LLM decides when to call tools).
Instead, our GRAPH decides when to execute code — the nodes call these
functions directly. The LLM produces the code (via structured output),
and the node runs it using these helpers.

Why not use @tool + ToolNode like P2?
  In P2, the LLM was in a loop: think → call tool → see result → think again.
  Here, the flow is more structured: plan → execute step 1 → execute step 2 → ...
  The graph controls the sequence, not the LLM. So we don't need the
  agent loop pattern — we just need helper functions the nodes can call.
"""

from __future__ import annotations

import io
import contextlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no GUI window, just files
import matplotlib.pyplot as plt
import pandas as pd


def execute_python(code: str, df: pd.DataFrame) -> dict[str, object]:
    """Execute Python/pandas code against a DataFrame and return the result.

    This is the most powerful tool in our portfolio so far. The LLM writes
    pandas code, and this function RUNS it.

    How it works:
      1. We create a namespace dict with 'df' (the data) and common imports
      2. We exec() the code string in that namespace
      3. We capture anything printed to stdout (via contextlib.redirect_stdout)
      4. We look for a 'result' variable in the namespace (convention)
      5. We return both the printed output and the result

    Safety note:
      exec() runs arbitrary Python — in production, you'd sandbox this
      (e.g., Docker container, RestrictedPython, or a code interpreter API).
      For learning, we trust the LLM's output since we're running locally.

    Args:
        code: Python code to execute. Should use 'df' for the DataFrame
              and store output in 'result'.
        df: The pandas DataFrame to analyze.

    Returns:
        dict with:
          - "success": bool
          - "output": printed stdout (str)
          - "result": the 'result' variable if set, else None
          - "error": error message if failed, else None
    """
    # The namespace is the "environment" the code runs in.
    # We pre-load it with the DataFrame and common imports so the
    # LLM's code can use them directly.
    namespace: dict[str, object] = {
        "df": df,
        "pd": pd,
    }

    # Capture stdout — anything the code print()s goes here
    stdout_capture = io.StringIO()

    try:
        with contextlib.redirect_stdout(stdout_capture):
            exec(code, namespace)  # noqa: S102 — exec is intentional here

        return {
            "success": True,
            "output": stdout_capture.getvalue(),
            "result": namespace.get("result"),
            "error": None,
        }
    except Exception as e:
        return {
            "success": False,
            "output": stdout_capture.getvalue(),
            "result": None,
            "error": f"{type(e).__name__}: {e}",
        }


def save_chart(code: str, df: pd.DataFrame, chart_path: Path) -> dict[str, object]:
    """Execute matplotlib code and save the resulting chart to a file.

    Similar to execute_python, but specialized for chart generation:
      - Pre-loads matplotlib.pyplot as plt
      - Provides chart_path so the code knows where to save
      - Always closes the figure to free memory

    The LLM writes full matplotlib code (create figure, plot, label, save).
    We just provide the environment and handle cleanup.

    Args:
        code: Matplotlib code to execute. Should use 'df', 'plt', and
              'chart_path'. Must call plt.savefig(chart_path).
        df: The pandas DataFrame to plot from.
        chart_path: Where to save the chart image.

    Returns:
        dict with:
          - "success": bool
          - "chart_path": str path if saved, else None
          - "error": error message if failed, else None
    """
    namespace: dict[str, object] = {
        "df": df,
        "pd": pd,
        "plt": plt,
        "chart_path": str(chart_path),
    }

    try:
        exec(code, namespace)  # noqa: S102

        if chart_path.exists():
            return {
                "success": True,
                "chart_path": str(chart_path),
                "error": None,
            }
        else:
            return {
                "success": False,
                "chart_path": None,
                "error": "Code ran but no chart file was saved. Did you call plt.savefig(chart_path)?",
            }
    except Exception as e:
        return {
            "success": False,
            "chart_path": None,
            "error": f"{type(e).__name__}: {e}",
        }
    finally:
        # Always close all figures to prevent memory leaks.
        # Even if the code failed, there might be a half-built figure.
        plt.close("all")
