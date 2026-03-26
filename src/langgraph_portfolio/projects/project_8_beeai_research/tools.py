"""
Project 8 — Custom Tools for the BeeAI Research Agent.

---------------------------------------------------------------------------
HOW BEEAI TOOLS WORK (vs LangGraph)
---------------------------------------------------------------------------

In LangGraph (P2), we defined a tool like this:

    @tool
    def search_documents(query: str) -> str:
        '''Search a knowledge base...'''
        hits = kb.search(query)
        return format(hits)

LangGraph reads the function name, docstring, and type hints to build the
tool's schema automatically. Very concise, but the schema is implicit.

In BeeAI, tools are CLASSES, not decorated functions:

    class SearchDocumentsTool(Tool[SearchInput, None, StringToolOutput]):
        name = "search_documents"
        description = "Search a knowledge base..."
        input_schema = SearchInput      # a Pydantic model

        async def _run(self, input, options, context):
            ...
            return StringToolOutput(result)

Why the difference?

BeeAI treats tools as first-class objects with:
  - An explicit Pydantic schema (you define exactly what the tool accepts)
  - An async _run() method (all BeeAI tools are async)
  - A built-in emitter (the tool itself emits events when it runs!)
  - Type parameters: Tool[TInput, TRunOptions, TOutput]

The tradeoff:
  - LangGraph: less code, schema inferred → faster to write
  - BeeAI: more code, schema explicit → clearer contract, better validation

Both approaches tell the LLM: "You have a tool called X that takes Y."
The LLM doesn't care HOW the tool was defined — it just sees the name,
description, and input schema.
---------------------------------------------------------------------------
"""

from pydantic import BaseModel, Field

from beeai_framework.tools import StringToolOutput, Tool
from beeai_framework.context import RunContext

# Reuse the same knowledge base from our core module — same data as P2.
# This makes the comparison fair: same tool, same data, different framework.
from ...core.knowledge import Document, KnowledgeBase


# ---------------------------------------------------------------------------
# 1. INPUT SCHEMAS
#
# In BeeAI, every tool needs a Pydantic model for its input.
# This is what gets sent to the LLM as the tool's parameter schema.
#
# Compare to P2: there, the @tool decorator read `query: str` from the
# function signature. Here, we define it as a proper Pydantic model.
# The result is the same — the LLM sees "this tool takes a query string" —
# but BeeAI makes it explicit.
# ---------------------------------------------------------------------------

class SearchInput(BaseModel):
    """Input schema for the search tool."""
    query: str = Field(description="Search query to find relevant documents")


class CalculatorInput(BaseModel):
    """Input schema for the calculator tool."""
    expression: str = Field(description="A mathematical expression to evaluate, e.g. '2 + 3 * 4'")


# ---------------------------------------------------------------------------
# 2. KNOWLEDGE BASE SETUP
#
# Same documents as P2 — we're building the same research agent, just with
# a different framework. Using identical data lets us compare the frameworks
# on equal footing.
# ---------------------------------------------------------------------------

def create_knowledge_base() -> KnowledgeBase:
    """Create and populate the knowledge base (same data as P2)."""
    kb = KnowledgeBase()
    kb.index([
        Document(
            title="LangGraph Overview",
            content="LangGraph provides stateful orchestration, routing, and persistence for agentic workflows.",
            source="docs/langgraph.md",
        ),
        Document(
            title="CrewAI Overview",
            content="CrewAI focuses on role-based multi-agent coordination and task-oriented collaboration.",
            source="docs/crewai.md",
        ),
        Document(
            title="LlamaIndex Connectors",
            content="LlamaIndex provides data connectors, document loading, and retrieval pipelines for RAG systems.",
            source="docs/llamaindex.md",
        ),
    ])
    return kb


# ---------------------------------------------------------------------------
# 3. SEARCH TOOL
#
# This is the BeeAI equivalent of P2's @tool search_documents.
#
# The class inherits from Tool[SearchInput, None, StringToolOutput]:
#   - SearchInput:       the Pydantic model defining what the tool accepts
#   - None:              no custom run options (we use the default)
#   - StringToolOutput:  the tool returns plain text
#
# Three required properties (all abstract in the base class):
#   - name:         what the LLM sees as the tool name
#   - description:  what the LLM reads to decide when to use the tool
#   - input_schema: the Pydantic model class (not an instance!)
#
# One required method:
#   - async _run(): the actual logic (note: it's ASYNC, unlike P2's sync @tool)
#
# The _run() method receives:
#   - input:   a validated SearchInput instance (BeeAI validates for you!)
#   - options: run options (None in our case)
#   - context: RunContext with metadata about the current execution
# ---------------------------------------------------------------------------

class SearchDocumentsTool(Tool[SearchInput, None, StringToolOutput]):
    """Search the knowledge base for documents about AI frameworks."""

    name = "search_documents"
    description = (
        "Search a knowledge base of documents about AI frameworks. "
        "Use this tool to find information about LangGraph, CrewAI, LlamaIndex, "
        "and other agentic AI frameworks. Pass a search query and get back "
        "matching document excerpts with their sources."
    )
    input_schema = SearchInput

    def __init__(self, knowledge_base: KnowledgeBase) -> None:
        super().__init__()
        self._kb = knowledge_base

    def _create_emitter(self, *args, **kwargs):
        """Create the event emitter for this tool.

        Every BeeAI tool has its own emitter — this is part of the
        observability system. When the tool runs, it emits events that
        event handlers can listen to. We get this for free by inheriting
        from Tool.
        """
        from beeai_framework.emitter import Emitter
        return Emitter.root().child(
            namespace=["tool", "search_documents"],
            creator=self,
        )

    async def _run(
        self,
        input: SearchInput,
        options: None,
        context: RunContext,
    ) -> StringToolOutput:
        """Execute the search (called by the agent automatically).

        Compare to P2's search_documents():
          - P2: receives raw `query: str`, returns raw `str`
          - BeeAI: receives validated `SearchInput`, returns `StringToolOutput`

        The validation is automatic — if someone passes bad input, BeeAI
        raises a ToolInputValidationError before _run() is even called.
        """
        hits = self._kb.search(input.query, limit=3)

        if not hits:
            return StringToolOutput("No documents found matching your query.")

        results = []
        for hit in hits:
            results.append(
                f"[Source: {hit.document.title} ({hit.document.source})] "
                f"{hit.document.content}"
            )
        return StringToolOutput("\n\n".join(results))


# ---------------------------------------------------------------------------
# 4. CALCULATOR TOOL
#
# A simple second tool to show that adding tools is just defining another
# class. The agent can use multiple tools in a single research session.
#
# In P2, we only had the search tool. Here we add a calculator to show
# how BeeAI handles multiple tools — the agent decides which to call
# based on the question.
# ---------------------------------------------------------------------------

class CalculatorTool(Tool[CalculatorInput, None, StringToolOutput]):
    """Evaluate mathematical expressions."""

    name = "calculator"
    description = "Evaluate a mathematical expression. Use for any arithmetic or calculations."
    input_schema = CalculatorInput

    def _create_emitter(self, *args, **kwargs):
        from beeai_framework.emitter import Emitter
        return Emitter.root().child(
            namespace=["tool", "calculator"],
            creator=self,
        )

    async def _run(
        self,
        input: CalculatorInput,
        options: None,
        context: RunContext,
    ) -> StringToolOutput:
        """Safely evaluate a math expression.

        We use a restricted eval for safety — only math operations allowed.
        In production you'd use a proper math parser, but this is fine
        for learning.
        """
        allowed_names = {"__builtins__": {}}
        try:
            result = eval(input.expression, allowed_names)  # noqa: S307
            return StringToolOutput(f"Result: {result}")
        except Exception as e:
            return StringToolOutput(f"Error evaluating '{input.expression}': {e}")
