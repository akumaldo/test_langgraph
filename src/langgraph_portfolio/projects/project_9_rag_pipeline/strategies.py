"""
PROJECT 9 — RAG Pipeline (LlamaIndex Workflows)
================================================
FILE: strategies.py — Retrieval Strategy Implementations

CONCEPT: RETRIEVAL STRATEGIES
Not all searches are the same. Given a query like "What causes inflation?":

1. KEYWORD (BM25): Finds chunks containing the exact words "causes" and
   "inflation." Fast and precise, but misses synonyms. If the document
   says "drivers of rising prices" instead, keyword search won't find it.

2. SEMANTIC: Converts the query to a vector and finds chunks with similar
   vectors. "What causes inflation?" and "drivers of rising prices" have
   similar vectors because the MEANING is close. This is the default in
   most RAG systems.

3. HYBRID: Runs BOTH keyword and semantic, then combines the results.
   This catches both exact matches AND conceptual matches. The real-world
   standard for production RAG systems.

4. RERANKING: A two-pass approach. First, semantic retrieval gets a larger
   set of candidates (e.g., top 10). Then the LLM scores each candidate
   for relevance to the query and returns the best ones. Higher quality
   but slower — the LLM has to evaluate each candidate individually.

WHY COMPARE?
Running the same question through all 4 strategies shows you their
strengths and weaknesses concretely. You'll see cases where keyword
finds something semantic misses (exact terminology) and vice versa
(conceptual similarity with different words).

DESIGN: STRATEGIES ARE ISOLATED
This file doesn't import the workflow or events. The QueryWorkflow
calls these functions and wraps the results in events. This means
you can add a new strategy without touching the workflow — just add
a function here and a case in the retrieve step.
"""

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.llms.ollama import Ollama
from llama_index.retrievers.bm25 import BM25Retriever

# How many chunks to retrieve per query
TOP_K = 5

# For reranking: retrieve more candidates, then LLM picks the best
RERANK_INITIAL_K = 10
RERANK_FINAL_K = 5

LLM_MODEL = "qwen3.5:2b"


async def retrieve_keyword(nodes: list[TextNode], query: str) -> list[NodeWithScore]:
    """BM25 keyword retrieval — traditional text matching.

    CONCEPT: BM25
    BM25 (Best Matching 25) is a ranking function from information retrieval.
    It scores documents by:
    - Term frequency: how often query words appear in the chunk
    - Inverse document frequency: rare words matter more than common ones
    - Document length normalization: long chunks don't get unfair advantage

    This is what search engines used before embeddings. It's fast, predictable,
    and works well for exact terminology (e.g., error codes, product names).

    Note: BM25 works on TEXT, not vectors. It needs the raw TextNode objects,
    not the VectorStoreIndex. That's why the IngestionWorkflow returns both.
    """
    bm25_retriever = BM25Retriever(
        nodes=nodes,
        similarity_top_k=TOP_K,
    )
    return await bm25_retriever.aretrieve(query)


async def retrieve_semantic(index: VectorStoreIndex, query: str) -> list[NodeWithScore]:
    """Embedding-based semantic retrieval — find conceptually similar chunks.

    CONCEPT: VECTOR SIMILARITY
    The query gets embedded into a vector (same model used during ingestion).
    Then we find the chunks whose vectors are CLOSEST to the query vector.
    "Closest" is measured by cosine similarity — how aligned the two vectors are.

    This finds chunks that mean the same thing even if they use different words.
    "How does photosynthesis work?" matches "Plants convert sunlight to energy"
    because the embeddings capture MEANING, not just word overlap.

    This is the default retrieval in most RAG systems.
    """
    retriever = index.as_retriever(similarity_top_k=TOP_K)
    return await retriever.aretrieve(query)


async def retrieve_hybrid(
    index: VectorStoreIndex, nodes: list[TextNode], query: str
) -> list[NodeWithScore]:
    """Hybrid retrieval — combines keyword (BM25) + semantic results.

    CONCEPT: HYBRID SEARCH
    Keyword search is great for exact matches. Semantic search is great for
    conceptual matches. Hybrid combines both — run both retrievers, merge
    the results, and deduplicate.

    Real-world RAG systems almost always use hybrid. Pure semantic can miss
    exact terminology; pure keyword can miss paraphrases. Hybrid catches both.

    IMPLEMENTATION: Simple merge + deduplicate by node ID, keeping the
    higher score when a chunk appears in both result sets. We take the
    top TOP_K from the merged set.
    """
    # Run both retrievers
    keyword_results = await retrieve_keyword(nodes, query)
    semantic_results = await retrieve_semantic(index, query)

    # Merge: deduplicate by node ID, keep the higher score
    seen: dict[str, NodeWithScore] = {}
    for result in keyword_results + semantic_results:
        node_id = result.node.node_id
        if node_id not in seen or (result.score or 0) > (seen[node_id].score or 0):
            seen[node_id] = result

    # Sort by score descending, take top K
    merged = sorted(seen.values(), key=lambda r: r.score or 0, reverse=True)
    return merged[:TOP_K]


async def retrieve_for_reranking(index: VectorStoreIndex, query: str) -> list[NodeWithScore]:
    """First pass of reranking — get a larger set of semantic candidates.

    This retrieves RERANK_INITIAL_K candidates (more than usual). The rerank
    step in the workflow will then use the LLM to score each one and keep
    only the best RERANK_FINAL_K.
    """
    retriever = index.as_retriever(similarity_top_k=RERANK_INITIAL_K)
    return await retriever.aretrieve(query)


async def rerank_with_llm(query: str, candidates: list[NodeWithScore]) -> list[NodeWithScore]:
    """Second pass — LLM scores each candidate for relevance.

    CONCEPT: LLM-BASED RERANKING
    Embedding similarity is a rough measure — it captures general topic
    similarity. But an LLM can understand NUANCE. It can tell the
    difference between "causes of inflation" and "history of inflation"
    even if their embeddings are close.

    We ask the LLM to score each chunk on a 0-10 scale for relevance to
    the query. Then we sort by score and take the top results.

    TRADE-OFF: This is slower (one LLM call per candidate) but higher
    quality. In production, you'd use a specialized reranker model (like
    Cohere Rerank). Here we use the same LLM to teach the concept without
    extra dependencies.
    """
    if not candidates:
        return []

    llm = Ollama(model=LLM_MODEL, request_timeout=600)

    scored_candidates: list[NodeWithScore] = []
    for candidate in candidates:
        chunk_text = candidate.node.get_content()[:500]  # truncate for prompt
        prompt = (
            f"Rate the relevance of this text chunk to the query on a scale of 0-10.\n"
            f"Query: {query}\n"
            f"Chunk: {chunk_text}\n\n"
            f"Respond with ONLY a number between 0 and 10. No explanation."
        )
        response = await llm.acomplete(prompt)
        try:
            score = float(response.text.strip())
            score = max(0, min(10, score))  # clamp to 0-10
        except ValueError:
            score = 5.0  # default if LLM returns non-numeric

        scored_candidates.append(
            NodeWithScore(node=candidate.node, score=score / 10.0)
        )

    # Sort by LLM score descending, take top K
    scored_candidates.sort(key=lambda r: r.score or 0, reverse=True)
    return scored_candidates[:RERANK_FINAL_K]
