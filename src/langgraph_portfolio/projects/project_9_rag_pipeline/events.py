"""
PROJECT 9 — RAG Pipeline (LlamaIndex Workflows)
================================================
FILE: events.py — Custom Event Types

CONCEPT: EVENT-DRIVEN ARCHITECTURE
In LangGraph, steps connect via EDGES that you explicitly draw:
    node_A --add_edge--> node_B --add_edge--> node_C

In LlamaIndex Workflows, steps connect via EVENTS:
    step_A emits EventX → step_B listens for EventX → step_B emits EventY

The key difference: steps DON'T KNOW about each other. They only know about
events. step_B doesn't import or call step_A — it just says "I fire when
EventX arrives." This is DECOUPLED design.

WHY A SEPARATE FILE?
Events are the CONTRACT between steps. By putting them in their own file:
1. Any step can import the events it needs without importing other steps
2. You can see the entire "API" of the system in one place
3. Adding a new step just means: create a new event, have the step listen for it

Think of events like TypedDict states in LangGraph — they define the data
that flows between steps. But instead of ONE shared state dict, each event
carries exactly the data its consumer needs.

Each event inherits from llama_index's Event base class, which itself is
a Pydantic BaseModel. So you get type validation for free.
"""

from llama_index.core.workflow import Event
from llama_index.core.schema import TextNode


# ── INGESTION WORKFLOW EVENTS ────────────────────────────────────────────────
# These events flow through the document ingestion pipeline:
# StartEvent → load_document → DocumentLoaded → chunk_and_embed → StopEvent


class DocumentLoaded(Event):
    """Fired after the PDF has been read and raw pages extracted.

    This is the first processing event — the raw material before any
    chunking or embedding happens.
    """

    raw_docs: list  # list of LlamaIndex Document objects


# Note: we don't need an IndexReady event because the IngestionWorkflow
# returns the index via StopEvent. The index and nodes are stored on the
# QueryWorkflow instance, not passed through events.


# ── QUERY WORKFLOW EVENTS ────────────────────────────────────────────────────
# These events flow through the query pipeline. Notice the BRANCHING:
#
# For keyword/semantic/hybrid strategies:
#   StartEvent → validate → QueryValidated → retrieve → DocumentsReady → synthesize → ...
#
# For reranking strategy:
#   StartEvent → validate → QueryValidated → retrieve → RetrievalDone → rerank → DocumentsReady → synthesize → ...
#
# Both paths converge at DocumentsReady — the synthesize step doesn't care
# HOW the documents were prepared, it just needs them ready.
# This is the power of event-driven design: you can insert steps (like reranking)
# without changing downstream consumers.


class QueryValidated(Event):
    """Fired after the user's query passes validation checks.

    Validation rejects empty queries, too-short queries, etc.
    The strategy field tells the retrieve step which approach to use.
    """

    query: str
    strategy: str


class RetrievalDone(Event):
    """Fired by the retrieve step ONLY for the reranking strategy.

    These are the initial candidates before LLM-based reranking.
    For other strategies, retrieve emits DocumentsReady directly,
    skipping the rerank step entirely.

    This conditional emission is how LlamaIndex Workflows handle
    branching — the step decides which event to emit based on logic,
    and different downstream steps listen for different events.
    """

    query: str
    documents: list  # list of NodeWithScore from initial retrieval


class DocumentsReady(Event):
    """The CONVERGENCE POINT — all strategies eventually emit this.

    - Keyword, semantic, hybrid: retrieve step emits this directly
    - Reranking: rerank step emits this after LLM scoring

    The synthesize step ONLY listens for DocumentsReady. It doesn't
    know or care whether reranking happened. This is event-driven
    decoupling in action.
    """

    query: str
    documents: list  # list of NodeWithScore, final ranked order


class SynthesisDone(Event):
    """Fired after the LLM generates an answer from the retrieved chunks.

    The evaluate_answer step checks quality. If the answer is good,
    it returns StopEvent. If not, it emits ResynthesizeRequest.
    """

    query: str
    answer: str
    documents: list  # the chunks used to generate the answer


class ResynthesizeRequest(Event):
    """Fired when evaluate_answer judges the answer as low quality.

    CONCEPT: QUALITY-CONTROL LOOP (same pattern as P3's editor→writer)
    In P3, the editor could send work back to the writer with feedback.
    Here, evaluate_answer can send work back to synthesize with feedback.

    The retry_count prevents infinite loops — after MAX_RETRIES, the
    system returns whatever it has (same safety cap as P3's MAX_REVISIONS).

    The synthesize step listens for BOTH DocumentsReady (first attempt)
    and ResynthesizeRequest (retry). This dual-listen pattern is how
    LlamaIndex Workflows handle loops — the step fires for either event.
    """

    query: str
    documents: list
    feedback: str  # what to improve
    retry_count: int  # how many retries so far
