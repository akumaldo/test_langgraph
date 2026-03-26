"""
PROJECT 10 — FRAMEWORK SHOWDOWN: LLAMAINDEX WORKFLOWS VERSION
===============================================================

This is the LlamaIndex Workflows implementation of the research assistant.
It reuses patterns from P9 (RAG pipeline):
  - @step decorator for workflow methods
  - Event subclasses for inter-step communication
  - Async execution (await workflow.run())

KEY CONCEPT — EVENT-DRIVEN ORCHESTRATION:
  In LangGraph, you wire steps with EXPLICIT EDGES (add_edge, add_conditional_edges).
  In CrewAI, you list tasks in ORDER (Process.sequential).
  In AG2, agents take TURNS in a conversation.
  In BeeAI, the agent runs a BUILT-IN loop.
  In LlamaIndex Workflows, steps are connected by EVENTS.

  A step declares what event it RECEIVES (type hint on parameter) and
  what event it EMITS (return type). The framework routes events
  automatically — no explicit wiring needed.

  This is like a pub/sub system: steps subscribe to event types.
  If step A emits EventX and step B takes EventX, they're connected.
  You never write add_edge("A", "B").

EVENT FLOW:
  StartEvent(question)
      ↓
  search_step() → SearchDone(documents)
      ↓
  analyze_step() → AnalysisDone(analysis)
      ↓
  report_step() → StopEvent(result)

COMPARE WITH P9:
  P9 used vector stores, embeddings, and retrieval strategies.
  Here we use the shared keyword search — simpler, but same workflow
  pattern. The interesting comparison is the orchestration, not the
  retrieval.
"""

import asyncio
import time

from llama_index.core.workflow import Workflow, step, Event, StartEvent, StopEvent
from llama_index.llms.ollama import Ollama

from langgraph_portfolio.projects.project_10_showdown.problem import (
    RESEARCH_QUESTION,
    search_knowledge_base,
    parse_report,
    save_result,
)


# ── MODEL CONFIG ─────────────────────────────────────────────────────────
# LlamaIndex uses its own Ollama integration.
LLM_MODEL = "qwen3.5:2b"


# ── EVENTS ───────────────────────────────────────────────────────────────
# Typed events define what data flows between steps.
#
# HOW EVENTS WORK (recap from P9):
#   Each Event is a Pydantic model. Steps receive and emit events.
#   The framework matches event types automatically:
#     - search_step emits SearchDone
#     - analyze_step receives SearchDone (matched by type)
#   No explicit wiring needed — the type IS the connection.
#
# COMPARE WITH LANGGRAPH:
#   LangGraph: add_edge("search", "analyze")  ← explicit
#   LlamaIndex: search emits SearchDone, analyze takes SearchDone  ← implicit

class SearchDone(Event):
    """Emitted when KB search is complete."""
    documents: str  # formatted search results


class AnalysisDone(Event):
    """Emitted when analysis is complete."""
    analysis: str  # the researcher's analysis text


# ── WORKFLOW ─────────────────────────────────────────────────────────────
# Three steps: search → analyze → report
#
# Each @step method:
#   - Receives a specific event type (parameter type hint)
#   - Does its work
#   - Returns a new event (or StopEvent to end)

class ResearchWorkflow(Workflow):

    @step
    async def search_step(self, ev: StartEvent) -> SearchDone:
        """Search the knowledge base.

        Receives: StartEvent with the research question
        Emits: SearchDone with formatted document text

        This is the equivalent of LangGraph's tool node, but there's
        no LLM deciding whether to search — we always search.
        The intelligence comes in the analyze step.
        """
        question = ev.get("question", RESEARCH_QUESTION)
        results = search_knowledge_base(question)

        formatted = []
        for doc in results:
            formatted.append(
                f"[{doc['title']}] (Source: {doc['source']})\n{doc['content']}"
            )
        documents = "\n\n---\n\n".join(formatted)

        print(f"  📚 Found {len(results)} documents")
        return SearchDone(documents=documents)

    @step
    async def analyze_step(self, ev: SearchDone) -> AnalysisDone:
        """Analyze the search results using the LLM.

        Receives: SearchDone with document text
        Emits: AnalysisDone with the analysis

        This is where the LLM reads the documents and extracts
        key findings. Equivalent to LangGraph's research node or
        CrewAI's researcher agent.
        """
        llm = Ollama(model=LLM_MODEL, request_timeout=600)

        prompt = (
            f"Analyze the following documents about AI orchestration in industry. "
            f"Extract the key findings, important statistics, and main use cases.\n\n"
            f"QUESTION: {RESEARCH_QUESTION}\n\n"
            f"DOCUMENTS:\n{ev.documents}"
        )

        response = await llm.acomplete(prompt)
        analysis = response.text.strip()

        print(f"  🔍 Analysis complete ({len(analysis)} chars)")
        return AnalysisDone(analysis=analysis)

    @step
    async def report_step(self, ev: AnalysisDone) -> StopEvent:
        """Produce the final JSON report.

        Receives: AnalysisDone with analysis text
        Emits: StopEvent with the JSON report string

        Asks the LLM to format the analysis as a JSON report
        matching the ResearchReport schema. This is prompt-based
        structured output (unlike LangGraph's native with_structured_output).
        """
        llm = Ollama(model=LLM_MODEL, request_timeout=600)

        prompt = (
            f"Based on the following analysis, produce a JSON report with this "
            f"exact structure:\n"
            f'{{"summary": "2-3 sentence overview", '
            f'"findings": ["finding 1", "finding 2", ...], '
            f'"citations": ["document title - source", ...], '
            f'"framework": "llamaindex"}}\n\n'
            f"Output ONLY the JSON, no other text.\n\n"
            f"ANALYSIS:\n{ev.analysis}"
        )

        response = await llm.acomplete(prompt)
        report = response.text.strip()

        print(f"  📝 Report generated")
        return StopEvent(result=report)


# ── RUN ──────────────────────────────────────────────────────────────────

async def run_async():
    """Run the LlamaIndex research assistant and save results."""
    print("=" * 60)
    print("FRAMEWORK SHOWDOWN — LlamaIndex Workflows Version")
    print("=" * 60)
    print(f"\nQuestion: {RESEARCH_QUESTION}\n")

    workflow = ResearchWorkflow(timeout=300)

    start = time.time()
    result = await workflow.run(question=RESEARCH_QUESTION)
    elapsed = time.time() - start

    raw_output = str(result)

    print(f"\n{'─' * 40}")
    print(f"Result (elapsed: {elapsed:.1f}s):")
    print(raw_output)

    output = parse_report(raw_output, "llamaindex")
    path = save_result("llamaindex", output, elapsed)
    print(f"\nResult saved to: {path}")


def run():
    """Sync wrapper for the async run."""
    asyncio.run(run_async())


if __name__ == "__main__":
    run()
