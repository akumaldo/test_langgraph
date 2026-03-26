"""
Project 6 — Report Crew (Sequential Process).

The Report Crew is the THIRD and final stage of our intelligence pipeline.
It takes the competitive analysis and produces a polished executive
briefing — both as structured data and as a markdown file.

Like the Research Crew, this uses Process.sequential:
  Task 1 (Writer) → Task 2 (Editor)

The pattern is the same as P3's writer → editor flow, but with two key
differences:
  1. In P3, writer and editor were in the SAME crew with 3 other agents.
     Here, they're in their OWN crew — isolated, focused, independent.
  2. In P3, the editor could send work back to the writer (revision loop).
     Here, we keep it simple — one pass, write then edit. The editor
     polishes but doesn't loop back.

Why no revision loop?
  We COULD add one (CrewAI supports it in hierarchical mode), but:
    - The data has already been validated by the Analysis Crew
    - The Writer is transforming structured data into prose (less room for error)
    - Adding a loop here would mean using hierarchical mode (for the manager
      to decide when revisions are needed), defeating the purpose of showing
      sequential mode in contrast

  In a production system, you'd probably use hierarchical for the Report Crew
  too. But for learning, the contrast matters more.
"""

from crewai import Agent, Crew, Process, Task

# DISABLED: output_pydantic import — Instructor + Ollama incompatibility.
# See research_crew.py for the full explanation.
# from .models import IntelligenceReport


# ---------------------------------------------------------------------------
# LLM CONFIGURATION
# ---------------------------------------------------------------------------

LLM_MODEL = "ollama/qwen3.5:35b"


# ---------------------------------------------------------------------------
# AGENTS
#
# Two agents, mirroring P3's writer and editor but specialized for
# intelligence reports instead of articles.
#
# No tools needed — the Report Crew works entirely with text from the
# Analysis Crew. Its job is TRANSFORMATION (analysis → report), not
# data gathering.
# ---------------------------------------------------------------------------

report_writer = Agent(
    role="Intelligence Report Writer",
    goal=(
        "Transform competitive analysis data into a clear, actionable "
        "executive briefing. Your reports should be concise yet comprehensive "
        "— an executive should be able to understand the competitive landscape "
        "in 5 minutes."
    ),
    backstory=(
        "You are a senior business intelligence writer. You specialize in "
        "turning raw analysis into polished executive reports. You know that "
        "executives want the bottom line first (executive summary), details "
        "second (analysis), and clear next steps (recommendations). "
        "You write with precision — no fluff, no jargon, every sentence "
        "earns its place."
    ),
    llm=LLM_MODEL,
    verbose=True,
)

report_editor = Agent(
    role="Report Editor",
    goal=(
        "Review and polish intelligence reports for clarity, accuracy, "
        "consistency, and actionability. Ensure the report is ready for "
        "executive consumption."
    ),
    backstory=(
        "You are an editor specializing in business intelligence reports. "
        "You check for logical consistency (do the recommendations follow "
        "from the analysis?), factual accuracy (does the summary match the "
        "details?), and readability (can a busy executive scan this quickly?). "
        "You improve structure and clarity without changing the substance."
    ),
    llm=LLM_MODEL,
    verbose=True,
)


# ---------------------------------------------------------------------------
# BUILDING THE CREW
# ---------------------------------------------------------------------------


def build_report_crew(
    target_company: str,
    analysis_text: str,
) -> Crew:
    """Build and return the Report Crew.

    Args:
        target_company: The company being analyzed.
        analysis_text: Raw text output from the Analysis Crew containing
            the full competitive analysis.
    """

    write_task = Task(
        description=(
            f"Write a comprehensive intelligence report about {target_company}'s "
            f"competitive landscape based on the analysis below.\n\n"
            f"COMPETITIVE ANALYSIS DATA:\n"
            f"{analysis_text}\n\n"
            f"The report must include:\n"
            f"  1. Title — clear and descriptive\n"
            f"  2. Executive Summary — 2-3 paragraphs covering the key findings. "
            f"An executive who reads ONLY this section should understand the "
            f"competitive landscape.\n"
            f"  3. Detailed Analysis — the full comparison with market context. "
            f"Reference specific data points from the research.\n"
            f"  4. Competitor Summaries — a brief breakdown for each competitor "
            f"(overview, strengths, weaknesses, competitive position).\n"
            f"  5. Recommendations — 3-5 specific, actionable recommendations. "
            f"Each should reference the analysis that supports it.\n\n"
            f"Write for a senior executive audience. Be direct and specific."
        ),
        expected_output=(
            "A complete intelligence report with executive summary, "
            "detailed analysis, competitor summaries, and recommendations."
        ),
        agent=report_writer,
    )

    edit_task = Task(
        description=(
            f"Review and polish the intelligence report about {target_company}.\n\n"
            f"Check for:\n"
            f"  - Logical consistency: do recommendations follow from the analysis?\n"
            f"  - Factual accuracy: does the executive summary match the details?\n"
            f"  - Completeness: are all competitors covered in the summaries?\n"
            f"  - Clarity: can a busy executive scan this in 5 minutes?\n"
            f"  - Actionability: are recommendations specific and prioritized?\n\n"
            f"Improve the report where needed, then produce the final version. "
            f"The output must include all sections: title, executive summary, "
            f"detailed analysis, competitor summaries, and recommendations."
        ),
        expected_output=(
            "A polished, publication-ready intelligence report with all "
            "sections complete and consistent."
        ),
        agent=report_editor,
        # DISABLED: output_pydantic=IntelligenceReport
        #
        # The IntelligenceReport model has typed fields for every section,
        # ensuring the final output is structured and predictable.
        # This is the end of the entire pipeline:
        #   Research Crew → Analysis Crew → Report Crew → IntelligenceReport
        #
        # Compare to P3:
        #   Planner → Researcher → Writer → Editor → Publisher → PublishedArticle
        # Same idea (pipeline ending in structured output), but here the
        # pipeline spans THREE crews instead of one.
        #
        # Instructor + Ollama incompatibility prevents output_pydantic.
        # See research_crew.py for the full explanation.
        # With OpenAI/Anthropic, you'd re-enable:
        #   output_pydantic=IntelligenceReport,
    )

    crew = Crew(
        agents=[report_writer, report_editor],
        tasks=[write_task, edit_task],
        process=Process.sequential,
        # DISABLED: memory=True requires an embedding model (defaults to OpenAI).
        # To enable with Ollama:
        #   memory=True,
        #   embedder={"provider": "ollama", "config": {"model": "nomic-embed-text"}},
        memory=False,
        verbose=True,
    )

    return crew
