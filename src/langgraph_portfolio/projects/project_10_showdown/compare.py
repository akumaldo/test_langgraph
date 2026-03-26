"""
PROJECT 10 — FRAMEWORK SHOWDOWN: COMPARISON REPORT GENERATOR
==============================================================

This file loads the saved results from all 5 framework runs and
generates a markdown comparison report.

IT DOES NOT RUN THE FRAMEWORKS — each runs independently via its
own entry point. This only reads the JSON results and source files.

AUTOMATED METRICS:
  - Execution time (from saved JSON)
  - Lines of code (counted from source files)
  - Output completeness (did it produce all fields?)
  - Structured output success (Pydantic parsed or raw text?)

QUALITATIVE SECTIONS:
  Generated as templates for you to fill in after running all
  frameworks. These capture subjective experience: how did it feel
  to build and debug each version?
"""

from pathlib import Path

from langgraph_portfolio.projects.project_10_showdown.problem import load_results


# ── LINE COUNTING ────────────────────────────────────────────────────────
# Count non-blank, non-comment lines in each framework's source file.
# This measures implementation complexity — how much code did each
# framework need for the same task?

PROJECT_DIR = Path(__file__).parent

FRAMEWORK_FILES = {
    "langgraph": "langgraph_version.py",
    "crewai": "crewai_version.py",
    "ag2": "ag2_version.py",
    "beeai": "beeai_version.py",
    "llamaindex": "llamaindex_version.py",
}


def count_lines(filename: str) -> int:
    """Count non-blank, non-comment lines in a source file."""
    path = PROJECT_DIR / filename
    if not path.exists():
        return 0

    count = 0
    in_docstring = False
    for line in path.read_text().splitlines():
        stripped = line.strip()

        # Track docstrings (triple quotes)
        if '"""' in stripped or "'''" in stripped:
            # Single-line docstring
            if stripped.count('"""') == 2 or stripped.count("'''") == 2:
                continue
            in_docstring = not in_docstring
            continue
        if in_docstring:
            continue

        # Skip blank lines and comments
        if not stripped or stripped.startswith("#"):
            continue

        count += 1

    return count


# ── OUTPUT COMPLETENESS ──────────────────────────────────────────────────
# Check if the result has all expected fields with non-empty values.

def check_completeness(result: dict) -> dict:
    """Check which output fields are present and non-empty."""
    return {
        "summary": bool(result.get("summary")),
        "findings": bool(result.get("findings") and len(result["findings"]) > 0),
        "citations": bool(result.get("citations") and len(result["citations"]) > 0),
        "all_complete": all([
            bool(result.get("summary")),
            bool(result.get("findings") and len(result["findings"]) > 0),
            bool(result.get("citations") and len(result["citations"]) > 0),
        ]),
    }


# ── REPORT GENERATION ────────────────────────────────────────────────────

def generate_report() -> str:
    """Generate the comparison report as markdown."""
    results = load_results()

    if not results:
        return "# No results found\n\nRun the framework implementations first."

    lines = []
    lines.append("# Project 10 — Framework Showdown: Comparison Report\n")
    lines.append(f"**Question**: {results[next(iter(results))].get('question', 'N/A')}\n")
    lines.append(f"**Frameworks tested**: {', '.join(sorted(results.keys()))}\n")

    # ── Automated Metrics Table ──────────────────────────────────────
    lines.append("## Automated Metrics\n")
    lines.append("| Framework | Time (s) | Lines of Code | Output Complete | Structured Output |")
    lines.append("|-----------|----------|---------------|-----------------|-------------------|")

    for fw_name in ["langgraph", "crewai", "ag2", "beeai", "llamaindex"]:
        if fw_name not in results:
            lines.append(f"| {fw_name} | — | — | — | — |")
            continue

        result = results[fw_name]
        elapsed = result.get("elapsed_seconds", "—")
        loc = count_lines(FRAMEWORK_FILES.get(fw_name, ""))
        completeness = check_completeness(result)
        complete_str = "Yes" if completeness["all_complete"] else "Partial"
        structured = "Yes" if result.get("structured_output_success") else "No"

        lines.append(f"| {fw_name} | {elapsed} | {loc} | {complete_str} | {structured} |")

    # ── Output Summaries ─────────────────────────────────────────────
    lines.append("\n## Output Summaries\n")

    for fw_name in ["langgraph", "crewai", "ag2", "beeai", "llamaindex"]:
        if fw_name not in results:
            continue

        result = results[fw_name]
        lines.append(f"### {fw_name}\n")

        if result.get("structured_output_success"):
            lines.append(f"**Summary**: {result.get('summary', 'N/A')}\n")
            findings = result.get("findings", [])
            if findings:
                lines.append("**Findings**:")
                for f in findings:
                    lines.append(f"- {f}")
                lines.append("")
            citations = result.get("citations", [])
            if citations:
                lines.append("**Citations**:")
                for c in citations:
                    lines.append(f"- {c}")
                lines.append("")
        else:
            raw = result.get("raw_text", result.get("summary", "N/A"))
            lines.append(f"**Raw output** (structured parsing failed):\n")
            lines.append(f"```\n{raw[:1000]}\n```\n")

    # ── Qualitative Notes (template) ─────────────────────────────────
    lines.append("## Qualitative Notes\n")
    lines.append("*Fill in these sections after running all frameworks.*\n")

    for fw_name in ["langgraph", "crewai", "ag2", "beeai", "llamaindex"]:
        lines.append(f"### {fw_name}\n")
        lines.append("- **Control level**: <!-- how much did you design the flow vs framework decided? -->")
        lines.append("- **Debuggability**: <!-- how easy was it to trace what happened? -->")
        lines.append("- **Boilerplate**: <!-- how much setup code vs actual logic? -->")
        lines.append("- **Constraints**: <!-- what didn't work or required workarounds? -->")
        lines.append("")

    # ── Limitations ──────────────────────────────────────────────────
    lines.append("## Known Limitations\n")
    lines.append("- **Model**: Using `qwen3.5:2b` for all frameworks. Production would use larger models or API-based LLMs. Chosen for fair comparison and to avoid timeout issues experienced in earlier projects.")
    lines.append("- **CrewAI tool use**: Broken with Ollama due to Instructor incompatibility. Tools called from Python instead of by agents. Would work fine with OpenAI API.")
    lines.append("- **CrewAI structured output**: `output_pydantic` disabled for same reason. Returns plain text.")
    lines.append("- **BeeAI/CrewAI dep conflict**: Can't coexist in same Poetry venv (json-repair version clash). Must install separately via pip.")
    lines.append("- **Knowledge base**: Static keyword search, not vector/semantic retrieval. Chosen for simplicity and equal footing across frameworks.")
    lines.append("- **Single question**: One research question for speed. More questions would give stronger comparison signal.")
    lines.append("- **Structured output**: Only native in LangGraph (`with_structured_output`). Others rely on prompt-based JSON — a real framework difference worth noting.")
    lines.append("- **Local LLM context limits**: Prompts kept under ~15k tokens to avoid empty responses from Ollama.")
    lines.append("")

    return "\n".join(lines)


# ── RUN ──────────────────────────────────────────────────────────────────

def run():
    """Generate and save the comparison report."""
    print("=" * 60)
    print("FRAMEWORK SHOWDOWN — Generating Comparison Report")
    print("=" * 60)

    report = generate_report()

    output_path = PROJECT_DIR / "comparison_report.md"
    output_path.write_text(report)

    print(f"\nReport saved to: {output_path}")
    print(f"\n{report}")


if __name__ == "__main__":
    run()
