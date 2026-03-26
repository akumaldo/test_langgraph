"""
PROJECT 9 — RAG Pipeline (LlamaIndex Workflows)
================================================
FILE: main.py — Entry Point and Interactive CLI

CONCEPT: WORKFLOW COMPOSITION
This file demonstrates how two workflows work together:
1. IngestionWorkflow runs ONCE → produces index + nodes
2. QueryWorkflow runs PER QUESTION → uses the index to answer

The output of the first feeds into the second. This is like
function composition (f(g(x))) but at the workflow level.

RUNNING:
    poetry run project-9-rag-pipeline

Place a PDF file in src/langgraph_portfolio/projects/project_9_rag_pipeline/data/ before running.
"""

import asyncio
import sys
import time
from pathlib import Path

from .ingestion import IngestionWorkflow
from .models import ChunkInfo, RAGResponse
from .query import QueryWorkflow

# ── STRATEGY MAP ─────────────────────────────────────────────────────────────
# Maps CLI menu numbers to strategy names

STRATEGIES = {
    "1": "keyword",
    "2": "semantic",
    "3": "hybrid",
    "4": "reranking",
    "5": "all",
}


def find_pdf() -> Path:
    """Find a PDF file in the data/ directory.

    Looks in src/langgraph_portfolio/projects/project_9_rag_pipeline/data/ first,
    then asks the user.
    """
    data_dirs = [
        Path(__file__).resolve().parent / "data",
        Path.cwd() / "data",
    ]

    for data_dir in data_dirs:
        if data_dir.exists():
            pdfs = list(data_dir.glob("*.pdf"))
            if pdfs:
                if len(pdfs) == 1:
                    return pdfs[0]
                # Multiple PDFs — let the user pick
                print(f"\nFound {len(pdfs)} PDFs in {data_dir}:")
                for i, pdf in enumerate(pdfs, 1):
                    print(f"  [{i}] {pdf.name}")
                choice = input("Pick a PDF [1]: ").strip() or "1"
                try:
                    return pdfs[int(choice) - 1]
                except (ValueError, IndexError):
                    return pdfs[0]

    print("\nNo PDF found in src/langgraph_portfolio/projects/project_9_rag_pipeline/data/")
    print("Please place a PDF file there and try again.")
    sys.exit(1)


def format_response(response: RAGResponse) -> str:
    """Format a RAGResponse for CLI display."""
    lines = [f"── {response.strategy.upper()} {'─' * (40 - len(response.strategy))}"]
    lines.append(f"Retrieved {len(response.chunks)} chunks ({response.elapsed_seconds:.1f}s)")

    for chunk in response.chunks:
        page_str = f"p.{chunk.page}" if chunk.page else "p.?"
        # Truncate chunk text for display
        text_preview = chunk.text[:120].replace("\n", " ")
        if len(chunk.text) > 120:
            text_preview += "..."
        score_str = f" (score: {chunk.score:.2f})" if chunk.score is not None else ""
        lines.append(f"  [{chunk.rank}] {page_str}: \"{text_preview}\"{score_str}")

    lines.append(f"\nAnswer: {response.answer}")

    if response.was_resynthesized:
        lines.append("  (re-synthesized: initial answer didn't pass quality check)")
    if response.low_confidence:
        lines.append("  ⚠ Low confidence: max retries reached")

    return "\n".join(lines)


async def run_query(
    workflow: QueryWorkflow, query: str, strategy: str
) -> RAGResponse:
    """Run a single query with a single strategy, return structured result."""
    start_time = time.time()

    result = await workflow.run(query=query, strategy=strategy)

    elapsed = time.time() - start_time

    # Extract chunk info for display
    chunks = []
    for i, doc in enumerate(result.get("documents", []), 1):
        chunks.append(
            ChunkInfo(
                rank=i,
                text=doc.node.get_content(),
                page=doc.node.metadata.get("page"),
                score=doc.score,
            )
        )

    return RAGResponse(
        query=query,
        strategy=strategy,
        answer=result.get("answer", "No answer generated."),
        chunks=chunks,
        elapsed_seconds=elapsed,
        was_resynthesized=result.get("was_resynthesized", False),
        low_confidence=result.get("low_confidence", False),
    )


async def async_main():
    """Main async entry point."""
    print("=" * 60)
    print("  P9: RAG Pipeline (LlamaIndex Workflows)")
    print("=" * 60)

    # ── PHASE 1: INGESTION ───────────────────────────────────────────────
    # Run the IngestionWorkflow once to build the index
    pdf_path = find_pdf()
    print(f"\nIngesting: {pdf_path.name}")

    ingestion = IngestionWorkflow(timeout=600)
    result = await ingestion.run(pdf_path=str(pdf_path))

    index = result["index"]
    nodes = result["nodes"]

    print(f"\n✓ Ready! Index has {len(nodes)} chunks.\n")

    # ── PHASE 2: QUERY LOOP ─────────────────────────────────────────────
    # Create the QueryWorkflow and set the shared state
    query_wf = QueryWorkflow(timeout=600)
    query_wf.index = index
    query_wf.nodes = nodes

    print("Strategies: [1] Keyword  [2] Semantic  [3] Hybrid  [4] Reranking  [5] All")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            query = input("> Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if not query:
            continue

        strategy_input = input("> Strategy [1-5, default=2]: ").strip() or "2"
        strategy = STRATEGIES.get(strategy_input)

        if not strategy:
            print(f"Invalid choice '{strategy_input}'. Use 1-5.")
            continue

        print()

        if strategy == "all":
            # Run all 4 strategies sequentially and compare
            for strat in ["keyword", "semantic", "hybrid", "reranking"]:
                try:
                    response = await run_query(query_wf, query, strat)
                    print(format_response(response))
                    print()
                except Exception as e:
                    print(f"── {strat.upper()} ── ERROR: {type(e).__name__}: {e}\n")
                    import traceback
                    traceback.print_exc()
        else:
            try:
                response = await run_query(query_wf, query, strategy)
                print(format_response(response))
                print()
            except ValueError as e:
                print(f"Validation error: {e}\n")
            except Exception as e:
                print(f"Error: {type(e).__name__}: {e}\n")
                import traceback
                traceback.print_exc()


def main():
    """Sync entry point — wraps the async main."""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
