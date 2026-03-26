"""
PROJECT 9 — RAG Pipeline (LlamaIndex Workflows)
================================================
FILE: models.py — Pydantic Models for Results

These models structure the output that the CLI displays.
They're separate from events (which carry data between workflow steps)
— these are for presenting results to the user.

In P4 we used with_structured_output() to make the LLM return Pydantic
models directly. Here we construct the models ourselves from workflow output,
since the LLM generates free-text answers and we wrap them with metadata.
"""

from pydantic import BaseModel


class ChunkInfo(BaseModel):
    """A single retrieved chunk, formatted for display."""

    rank: int  # position in the result list (1-indexed)
    text: str  # the chunk content (truncated for display)
    page: int | None = None  # source page number, if available
    score: float | None = None  # relevance score from the retriever


class RAGResponse(BaseModel):
    """The complete response for a single query + strategy combination."""

    query: str
    strategy: str  # keyword, semantic, hybrid, reranking
    answer: str  # the synthesized answer from the LLM
    chunks: list[ChunkInfo]  # the retrieved chunks used
    elapsed_seconds: float  # how long this strategy took
    was_resynthesized: bool = False  # True if quality loop triggered
    low_confidence: bool = False  # True if max retries hit
