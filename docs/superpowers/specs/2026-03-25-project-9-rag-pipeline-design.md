# Project 9: RAG Pipeline (LlamaIndex Workflows) — Design Spec

## Overview

A Retrieval-Augmented Generation pipeline built with LlamaIndex Workflows. Ingests a user-provided PDF, builds a searchable vector index, and answers questions using four different retrieval strategies. The user can compare strategies side-by-side in an interactive CLI.

This project introduces **event-driven architecture** — steps communicate through typed events rather than explicit graph edges. It also teaches document ingestion, embedding models, and retrieval strategy trade-offs.

## Architecture: Two Composed Workflows

### IngestionWorkflow (runs once at startup)

Loads a PDF, chunks it, embeds the chunks, and builds a vector index.

```
StartEvent(pdf_path)
  → load_document → DocumentLoaded(raw_docs)
    → chunk_and_embed → IndexReady(index)
      → StopEvent(index)
```

### QueryWorkflow (runs per question)

Takes a query and strategy, retrieves relevant chunks, synthesizes an answer, and evaluates quality.

```
StartEvent(query, strategy)
  → validate_query → QueryValidated(query, strategy)
    → retrieve → DocumentsReady(query, documents)
    │               ↑
    │  (reranking strategy only)
    │  → retrieve → RetrievalDone(query, documents)
    │      → rerank (LLM scores relevance) → DocumentsReady(query, documents)
    │
    └→ synthesize (listens for DocumentsReady) → SynthesisDone(query, answer, documents)
        → evaluate_answer
            ├→ StopEvent(result)                          [quality OK]
            └→ ResynthesizeRequest(query, docs, feedback, retry_count)  [quality low]
                  → synthesize (also listens for ResynthesizeRequest)
                      (max 2 retries, then returns best answer with low-confidence note)
```

The index is stored on the `QueryWorkflow` instance (`self.index`, `self.nodes`) rather than threaded through events — keeps event payloads clean.

**Conditional routing**: The `retrieve` step checks the strategy. For keyword/semantic/hybrid, it emits `DocumentsReady` directly. For reranking, it emits `RetrievalDone`, which triggers the `rerank` step, which then emits `DocumentsReady`. The `synthesize` step only listens for `DocumentsReady` — it doesn't care how the documents got there.

**Quality loop**: `evaluate_answer` emits either `StopEvent` (pass) or `ResynthesizeRequest` (fail). `synthesize` listens for both `DocumentsReady` and `ResynthesizeRequest`. The retry count travels in the event.

### Why Two Workflows

Ingestion and querying have different lifecycles. Ingestion runs once; querying runs many times. Separating them shows that workflows are composable building blocks — the output of IngestionWorkflow feeds into QueryWorkflow.

## Custom Events

Each event is a typed class inheriting from LlamaIndex's `Event`:

| Event | Fields | Purpose |
|-------|--------|---------|
| `DocumentLoaded` | `raw_docs: list[Document]` | Raw pages extracted from PDF |
| `IndexReady` | `index: VectorStoreIndex, nodes: list[TextNode]` | Vector index + parsed nodes (nodes needed for BM25) |
| `QueryValidated` | `query: str, strategy: str` | Query passed validation |
| `RetrievalDone` | `query: str, documents: list` | Initial retrieval done (reranking strategy only) |
| `DocumentsReady` | `query: str, documents: list` | Final documents ready for synthesis (all strategies emit this) |
| `SynthesisDone` | `query: str, answer: str, documents: list` | Answer generated, pending evaluation |
| `ResynthesizeRequest` | `query: str, documents: list, feedback: str, retry_count: int` | Quality too low, retry synthesis |

Events are the "contract" between steps. They live in a dedicated `events.py` file to make the decoupling visible.

## Retrieval Strategies

Four strategies. Keyword and hybrid need both the `VectorStoreIndex` and the raw `nodes` (for BM25). The ingestion workflow returns both.

| Strategy | How It Works | What It Teaches |
|----------|-------------|-----------------|
| **Keyword** | BM25 text matching | Baseline traditional search |
| **Semantic** | Embedding similarity via `qwen3-embedding` | Default vector search |
| **Hybrid** | Combines keyword + semantic scores | Real-world standard |
| **Reranking** | Semantic retrieval → LLM reranks top results | Two-pass retrieval, quality vs cost |

Reranking uses the LLM (`qwen3.5:35b`) to score chunk relevance rather than a dedicated reranker model — teaches the concept without extra dependencies.

## Quality-Control Loop (DISABLED — local LLM performance constraint)

**Status**: Code is written but commented out in `query.py`. The evaluate step adds an extra LLM call per query, which causes timeouts with local Ollama models. The core pipeline works as: validate → retrieve → [rerank] → synthesize → done.

**Design (for re-enabling later)**: The `evaluate_answer` step checks if the answer addresses the query and uses the retrieved context. If quality is too low, it emits a `ResynthesizeRequest` event with feedback and a retry count. The `synthesize` step listens for both `DocumentsReady` (first attempt) and `ResynthesizeRequest` (retry). Safety cap: MAX_RETRIES = 2 (same pattern as P3's editor→writer loop). After max retries, returns the best answer with a low-confidence note.

**To re-enable**: Change synthesize to return `SynthesisDone` instead of `StopEvent`, and uncomment the `evaluate_answer` step.

## File Structure

```
src/langgraph_portfolio/projects/project_9_rag_pipeline/
  __init__.py
  main.py            — entry point: init workflows, CLI loop
  events.py          — all custom Event classes
  ingestion.py       — IngestionWorkflow: load PDF → chunk → embed → index
  query.py           — QueryWorkflow: validate → retrieve → rerank → synthesize → evaluate
  strategies.py      — retrieval strategy implementations
  models.py          — Pydantic models for results (RetrievalResult, RAGResponse, etc.)

project_9_rag_pipeline/
  __init__.py
  main.py            — thin wrapper (imports from src)
  data/              — where the user drops the PDF
```

**Why this split**:
- `events.py` isolated — events are the contract between steps
- `strategies.py` isolated — swapping/adding strategies doesn't touch the workflow
- `query.py` imports from both but neither knows about the other — event-driven decoupling in action

## LLM and Embedding Configuration

- **LLM**: Ollama `qwen3.5:35b` via LlamaIndex's Ollama integration (not LangChain's ChatOllama)
- **Embeddings**: Ollama `qwen3-embedding` via LlamaIndex's Ollama embedding class
- **Chunk size/overlap**: Configurable constants at top of `ingestion.py`

## CLI Interaction

Interactive loop:

```
=== P9: RAG Pipeline (LlamaIndex Workflows) ===
Ingesting PDF: data/your_file.pdf
Loading... chunking... embedding... done (42 chunks indexed)

Strategies: [1] Keyword  [2] Semantic  [3] Hybrid  [4] Reranking  [5] All

> Your question: What is X?
> Strategy [1-5]: 5

── Keyword ─────────────────────────────
Retrieved 3 chunks (0.8s)
  [1] p.12: "X is defined as..."
  [2] p.5:  "The concept of X..."
Answer: X is...

── Semantic ────────────────────────────
Retrieved 3 chunks (1.2s)
  [1] p.5:  "The concept of X..."
  [2] p.14: "Related to X, we find..."
Answer: X is...

(... Hybrid, Reranking ...)

Type 'quit' to exit.
```

Each response shows: retrieved chunks with page reference, synthesized answer, and timing. Evaluation step runs silently — if it triggers re-synthesis, shows a note.

## Error Handling

**Ingestion**: PDF not found/unreadable, empty PDF, zero chunks → clear error messages.

**Query validation**: Rejects empty queries, too-short queries (< 3 words) → asks user to rephrase.

**Retrieval**: No chunks found → tells user the document doesn't cover the topic (no hallucination).

**Quality loop**: MAX_RETRIES = 2, then returns best answer with low-confidence note.

**Infrastructure**: Ollama connection errors → "Is Ollama running?" message. Timeout on large chunks → configurable timeout.

## Tutor-Style Learning Concepts

Code comments will cover these P9 concepts:

1. **Event-driven vs graph-based** — `@step` + events vs `add_node` + `add_edge`
2. **Document ingestion pipeline** — load → chunk → embed → index, why each step
3. **Embedding models** — what embeddings are, how similarity search works
4. **Retrieval strategies compared** — keyword vs semantic vs hybrid vs reranking
5. **Quality-control loop** — same pattern as P3 but in event-driven form
6. **Workflow composition** — workflows as building blocks, one feeds into another

## Dependencies

In pyproject.toml (may not be installed yet — verify with `poetry install`):
- `llama-index = "^0.12.0"`

Packages to add:
- `llama-index-llms-ollama` — LLM integration for Ollama
- `llama-index-embeddings-ollama` — embedding integration for `qwen3-embedding`
- `llama-index-readers-file` — PDF reader
- `llama-index-retrievers-bm25` — BM25 retriever for keyword/hybrid strategies

If Poetry dep resolution fails (Python version constraint `<=3.13` vs installed 3.13.x), install via pip as done for BeeAI.

Entry point to register in pyproject.toml:
```toml
project-9-rag-pipeline = "langgraph_portfolio.projects.project_9_rag_pipeline.main:main"
```

## Configuration Constants

```python
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K = 5
MAX_RETRIES = 2
LLM_MODEL = "qwen3.5:35b"
EMBEDDING_MODEL = "qwen3-embedding"
```

## Scope Notes

- **PDF only** — the expansion plan mentions PDFs, text files, and web pages. This project focuses on PDF ingestion. Adding other formats later is straightforward (swap the reader).
- **Sequential strategy execution** — when "All" is selected, strategies run one at a time (Ollama serves one request at a time locally).
- **Async-native** — all workflow steps are `async def`. The CLI uses `asyncio.run()`.
