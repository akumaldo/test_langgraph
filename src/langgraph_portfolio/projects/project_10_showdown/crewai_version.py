"""
PROJECT 10 — FRAMEWORK SHOWDOWN: CREWAI VERSION
=================================================

This is the CrewAI implementation of the research assistant.
It reuses patterns from P6 (intel crew):
  - Agent + Task + Crew for orchestration
  - Sequential process (fixed task order)
  - Tools called from Python, not by agents

KEY CONSTRAINTS (from P6 experience):
  1. CrewAI tool use is BROKEN with Ollama (Instructor incompatibility).
     Agents cannot call BaseTool classes — the error is:
     "Instructor does not support multiple tool calls"
     WORKAROUND: call search_knowledge_base from Python, pass results
     as text context to the crew.

  2. output_pydantic is BROKEN with Ollama (same Instructor issue).
     WORKAROUND: ask the agent to produce JSON in its text output,
     then parse with parse_report().

  These are CrewAI + Ollama + Instructor version compatibility issues,
  NOT bugs in our code. They would likely work fine with OpenAI API.

FLOW:
  Python calls search_knowledge_base() → results as text
      ↓
  Crew (sequential):
    Task 1: researcher analyzes the search results
    Task 2: analyst writes the final report as JSON
      ↓
  parse_report() → save_result()

COMPARE WITH LANGGRAPH:
  - LangGraph: YOU build the graph, the edges, the loop. Maximum control.
  - CrewAI: YOU define agents and tasks. The framework runs them.
    Less control, less code, but also less visibility into what happens.
"""

import time

from crewai import Agent, Crew, Task, Process, LLM

from langgraph_portfolio.projects.project_10_showdown.problem import (
    RESEARCH_QUESTION,
    search_knowledge_base,
    parse_report,
    save_result,
)


# ── MODEL CONFIG ─────────────────────────────────────────────────────────
# CrewAI uses LiteLLM format for Ollama: "ollama/model_name"
LLM_MODEL = "ollama/qwen3.5:2b"

llm = LLM(model=LLM_MODEL)


# ── PRE-SEARCH ───────────────────────────────────────────────────────────
# We call the search function from Python BEFORE the crew runs.
# This is the workaround for the Ollama tool-use incompatibility.
#
# In a real CrewAI project with OpenAI, you'd assign a BaseTool to the
# agent and let it decide when to search. Here we can't, so we do it
# ourselves and pass the results as context.

def get_search_context() -> str:
    """Search the KB and format results as text context for the crew."""
    results = search_knowledge_base(RESEARCH_QUESTION)
    formatted = []
    for doc in results:
        formatted.append(
            f"[{doc['title']}] (Source: {doc['source']})\n{doc['content']}"
        )
    return "\n\n---\n\n".join(formatted)


# ── AGENTS ───────────────────────────────────────────────────────────────
# Two agents with different roles:
#   - researcher: reads and analyzes the search results
#   - analyst: synthesizes findings into a structured report
#
# HOW AGENTS WORK (recap from P6):
#   An Agent has a role, goal, and backstory. These become the system
#   prompt that shapes how the LLM behaves. The agent doesn't "know"
#   it's an agent — it's just an LLM with a specific persona.

researcher = Agent(
    role="Research Analyst",
    goal="Analyze documents about AI orchestration in industry and extract key findings",
    backstory=(
        "You are a research analyst specializing in AI technology adoption. "
        "You read source documents carefully and extract the most important "
        "facts, statistics, and trends."
    ),
    llm=llm,
    verbose=False,
)

analyst = Agent(
    role="Report Writer",
    goal="Write a structured research report as JSON based on research findings",
    backstory=(
        "You are a report writer who produces structured output. "
        "You take research findings and organize them into a clear, "
        "concise report format."
    ),
    llm=llm,
    verbose=False,
)


# ── TASKS ────────────────────────────────────────────────────────────────
# Tasks define WHAT each agent does. Sequential process means Task 1
# runs first, its output becomes available to Task 2.
#
# KEY DIFFERENCE FROM LANGGRAPH:
#   In LangGraph, you wire nodes with edges and the graph controls flow.
#   In CrewAI, you list tasks in order and the framework handles it.
#   Less explicit, but also less boilerplate.

def build_tasks(search_context: str) -> list[Task]:
    """Create the crew's tasks with the pre-fetched search context."""

    research_task = Task(
        description=(
            f"Analyze the following research documents about AI orchestration "
            f"in industry. Extract the key findings, important statistics, and "
            f"main use cases mentioned.\n\n"
            f"RESEARCH QUESTION: {RESEARCH_QUESTION}\n\n"
            f"DOCUMENTS:\n{search_context}"
        ),
        expected_output=(
            "A detailed analysis listing the main uses of AI orchestration "
            "across industries, with specific examples and statistics."
        ),
        agent=researcher,
    )

    report_task = Task(
        description=(
            "Based on the research analysis, produce a JSON report with this "
            "exact structure:\n"
            "{\n"
            '  "summary": "2-3 sentence overview",\n'
            '  "findings": ["finding 1", "finding 2", ...],\n'
            '  "citations": ["document title - source", ...],\n'
            '  "framework": "crewai"\n'
            "}\n\n"
            "Include all key findings from the research. Cite the source "
            "documents by title and source. Output ONLY the JSON, no other text."
        ),
        expected_output="A JSON object with summary, findings, citations, and framework fields.",
        agent=analyst,
    )

    return [research_task, report_task]


# ── RUN ──────────────────────────────────────────────────────────────────

def run():
    """Run the CrewAI research assistant and save results."""
    print("=" * 60)
    print("FRAMEWORK SHOWDOWN — CrewAI Version")
    print("=" * 60)
    print(f"\nQuestion: {RESEARCH_QUESTION}\n")

    # Step 1: Pre-search (Python, not agent)
    print("Searching knowledge base...")
    search_context = get_search_context()
    print(f"Found {search_context.count('---') + 1} documents\n")

    # Step 2: Run the crew
    tasks = build_tasks(search_context)
    crew = Crew(
        agents=[researcher, analyst],
        tasks=tasks,
        process=Process.sequential,
        verbose=False,
    )

    start = time.time()
    result = crew.kickoff()
    elapsed = time.time() - start

    # Step 3: Parse and save
    raw_output = result.raw
    print(f"\n{'─' * 40}")
    print(f"Result (elapsed: {elapsed:.1f}s):")
    print(raw_output)

    output = parse_report(raw_output, "crewai")
    path = save_result("crewai", output, elapsed)
    print(f"\nResult saved to: {path}")


if __name__ == "__main__":
    run()
