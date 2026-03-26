"""
PROJECT 10 — FRAMEWORK SHOWDOWN: BEEAI VERSION
================================================

This is the BeeAI implementation of the research assistant.
It reuses patterns from P8 (BeeAI research):
  - ReActAgent with built-in Think→Act→Observe loop
  - Tool subclass with explicit Pydantic input schema
  - Event-driven observability (emitter.on)
  - Async execution (await agent.run())

KEY CONCEPT — BUILT-IN REASONING LOOP:
  In LangGraph, YOU build the agent loop (node → conditional edge → tool → loop).
  In CrewAI, the framework runs tasks sequentially (no loop).
  In AG2, agents take turns in a conversation (implicit loop).
  In BeeAI, the agent HAS a ReAct loop BUILT IN. You give it tools and
  a question, and it autonomously:
    1. Thinks about what to do
    2. Acts (calls a tool)
    3. Observes the result
    4. Repeats until it has an answer

  You can't customize the loop structure — it's fixed. But you CAN
  observe every step via the event system, which gives the finest-grained
  debugging of any framework.

TOOL DEFINITION COMPARISON:
  - LangGraph: @tool decorator, schema inferred from type hints
  - CrewAI: BaseTool subclass, _run() method
  - AG2: no tools in P7 (conversational)
  - BeeAI: Tool subclass with EXPLICIT Pydantic input schema
    (you define a BaseModel for inputs — more verbose but more precise)

FLOW:
  agent.run(question) → ReAct loop:
    Think → Act (search_kb) → Observe → Think → final answer
      ↓
  parse_report() → save_result()
"""

import asyncio
import time

from beeai_framework.agents.react import ReActAgent
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.backend import ChatModel
from beeai_framework.tools import Tool, StringToolOutput
from beeai_framework.context import RunContext
from pydantic import BaseModel, Field

from langgraph_portfolio.projects.project_10_showdown.problem import (
    RESEARCH_QUESTION,
    search_knowledge_base,
    parse_report,
    save_result,
)


# ── MODEL CONFIG ─────────────────────────────────────────────────────────
# BeeAI uses provider-agnostic format: "ollama:model_name"
LLM_MODEL = "ollama:qwen3.5:2b"


# ── TOOL DEFINITION ─────────────────────────────────────────────────────
# BeeAI tools require an EXPLICIT Pydantic input schema.
#
# COMPARE WITH LANGGRAPH:
#   LangGraph's @tool infers the schema from type hints:
#     @tool
#     def search_kb(query: str) -> str:  # schema inferred from `query: str`
#
#   BeeAI requires a full BaseModel:
#     class SearchInput(BaseModel):
#         query: str = Field(description="...")
#
#   More verbose, but you get explicit validation and documentation.
#   In production, this precision matters for complex tool inputs.

class SearchInput(BaseModel):
    query: str = Field(description="Search query about AI orchestration in industry")


class SearchKBTool(Tool[SearchInput, None, StringToolOutput]):
    """Search the knowledge base about AI orchestration in industry."""

    name = "search_kb"
    description = (
        "Search the knowledge base for information about AI orchestration "
        "in industry. Use this to find documents about finance, healthcare, "
        "manufacturing, and customer service."
    )
    input_schema = SearchInput

    def _create_emitter(self, *args, **kwargs):
        """Create the event emitter for this tool (required by BeeAI)."""
        from beeai_framework.emitter import Emitter
        return Emitter.root().child(
            namespace=["tool", "search_kb"],
            creator=self,
        )

    async def _run(self, input: SearchInput, options: None, context: RunContext) -> StringToolOutput:
        results = search_knowledge_base(input.query)
        formatted = []
        for doc in results:
            formatted.append(
                f"[{doc['title']}] (Source: {doc['source']})\n{doc['content']}"
            )
        return StringToolOutput(result="\n\n---\n\n".join(formatted))


# ── EVENT HANDLER ────────────────────────────────────────────────────────
# BeeAI's event system is the finest-grained of all frameworks.
# You can observe every Think, Act, and Observe step.
#
# COMPARE:
#   - LangGraph: manual (print in your node functions)
#   - CrewAI: verbose=True flag (all or nothing)
#   - AG2: message callbacks (per-message)
#   - BeeAI: emitter.on("update") — per-thought, per-tool-call, per-observation

def log_event(data, event):
    """Log agent events for observability."""
    if hasattr(data, "update") and hasattr(data.update, "key"):
        key = data.update.key
        value = str(data.update.value)[:200]  # truncate for readability
        if key == "thought":
            print(f"  🧠 Think: {value}")
        elif key == "tool_name":
            print(f"  🔧 Tool: {value}")
        elif key == "final_answer":
            print(f"  ✅ Answer ready")


# ── RUN ──────────────────────────────────────────────────────────────────

async def run_async():
    """Run the BeeAI research assistant and save results."""
    print("=" * 60)
    print("FRAMEWORK SHOWDOWN — BeeAI Version")
    print("=" * 60)
    print(f"\nQuestion: {RESEARCH_QUESTION}\n")

    # Create the agent
    #
    # HOW ReActAgent WORKS (recap from P8):
    #   - memory: stores conversation history (UnconstrainedMemory = keep all)
    #   - tools: list of Tool instances the agent can use
    #   - The agent AUTONOMOUSLY decides when to use tools and when to stop
    #   - max_iterations is a safety cap (like LangGraph's graph structure
    #     or AG2's max_round)
    agent = ReActAgent(
        llm=ChatModel.from_name(LLM_MODEL),
        memory=UnconstrainedMemory(),
        tools=[SearchKBTool()],
    )

    # NOTE: Event handler disabled. The emitter API in this version of
    # beeai-framework raises EmitterError with our log_event handler.
    # P8 worked because it matched the exact event signature of its version.
    # This is a real framework pain point: API instability across versions.
    # agent.emitter.on("update", log_event)

    # System prompt asks for JSON output matching ResearchReport
    prompt = (
        f"{RESEARCH_QUESTION}\n\n"
        f"Use the search_kb tool to find information, then produce a JSON report "
        f"with this exact structure:\n"
        f'{{"summary": "2-3 sentence overview", '
        f'"findings": ["finding 1", "finding 2", ...], '
        f'"citations": ["document title - source", ...], '
        f'"framework": "beeai"}}\n\n'
        f"Output ONLY the JSON in your final answer, no other text."
    )

    start = time.time()
    try:
        result = await agent.run(prompt)
        elapsed = time.time() - start
        # Extract the final answer from the agent's output
        raw_output = result.output.text if result.output else ""
    except Exception as e:
        elapsed = time.time() - start
        # BeeAI's ReAct parser fails with qwen3.5:2b because the model's
        # thinking mode outputs duplicate "Thought:" lines that violate
        # the expected Thought→Action→Observation sequence.
        # This worked in P8 with qwen3.5:35b which follows ReAct format better.
        # This incompatibility is itself a comparison data point.
        print(f"\n  ⚠ BeeAI ReAct parser error: {e}")
        raw_output = ""

    print(f"\n{'─' * 40}")
    print(f"Result (elapsed: {elapsed:.1f}s):")
    print(raw_output)

    output = parse_report(raw_output, "beeai")
    path = save_result("beeai", output, elapsed)
    print(f"\nResult saved to: {path}")


def run():
    """Sync wrapper for the async run."""
    asyncio.run(run_async())


if __name__ == "__main__":
    run()
