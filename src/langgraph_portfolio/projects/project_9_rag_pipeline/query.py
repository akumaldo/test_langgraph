"""
PROJECT 9 — RAG Pipeline (LlamaIndex Workflows)
================================================
FILE: query.py — QueryWorkflow

THE QUERY PIPELINE:
StartEvent(query, strategy) → validate → retrieve → [rerank] → synthesize → evaluate → StopEvent

This workflow has 5 steps:
1. validate_query   — checks the query is valid (not empty, not too short)
2. retrieve         — uses the selected strategy to find relevant chunks
3. rerank           — (ONLY for reranking strategy) LLM scores candidates
4. synthesize       — LLM generates an answer from retrieved chunks
5. evaluate_answer  — checks answer quality, may trigger re-synthesis

CONCEPT: CONDITIONAL EVENT ROUTING
The retrieve step demonstrates event-driven branching:
- For keyword/semantic/hybrid → emits DocumentsReady (skips reranking)
- For reranking → emits RetrievalDone (triggers the rerank step)

The synthesize step only listens for DocumentsReady — it doesn't care
whether reranking happened. This is the event-driven equivalent of
LangGraph's conditional_edges, but more decoupled.

CONCEPT: QUALITY-CONTROL LOOP
Same pattern as P3's editor → writer revision loop, but in event-driven form:
- In P3: editor sends feedback to writer via state["revision_feedback"]
- Here: evaluate_answer emits ResynthesizeRequest with feedback
- In P3: MAX_REVISIONS prevents infinite loops
- Here: MAX_RETRIES does the same thing

The pattern transcends frameworks — quality gates with retry limits.

CONCEPT: WORKFLOW INSTANCE STATE
The index and nodes are stored on `self` (the workflow instance), not
passed through events. This keeps events lightweight — they carry only
the data specific to the current query, not heavy infrastructure objects.
In LlamaIndex Workflows, `self` is the natural place for shared state
that doesn't change between queries.
"""

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step
from llama_index.llms.ollama import Ollama

from .events import (
    DocumentsReady,
    QueryValidated,
    ResynthesizeRequest,
    RetrievalDone,
    SynthesisDone,
)
from .strategies import (
    rerank_with_llm,
    retrieve_for_reranking,
    retrieve_hybrid,
    retrieve_keyword,
    retrieve_semantic,
)

# ── CONFIGURATION ────────────────────────────────────────────────────────────

MAX_RETRIES = 2  # quality loop safety cap (like P3's MAX_REVISIONS)
LLM_MODEL = "qwen3.5:2b"
VALID_STRATEGIES = {"keyword", "semantic", "hybrid", "reranking"}


class QueryWorkflow(Workflow):
    """Handles a single question: validate → retrieve → synthesize → evaluate.

    USAGE:
        workflow = QueryWorkflow()
        workflow.index = index        # from IngestionWorkflow
        workflow.nodes = nodes        # from IngestionWorkflow
        result = await workflow.run(query="What is X?", strategy="semantic")
    """

    index: VectorStoreIndex | None = None
    nodes: list[TextNode] | None = None

    # ── STEP 1: VALIDATE ─────────────────────────────────────────────────

    @step
    async def validate_query(self, ev: StartEvent) -> QueryValidated:
        """Checks the query is valid before spending time on retrieval.

        Rejects:
        - Empty queries
        - Queries shorter than 3 words (likely not a real question)
        - Unknown strategy names

        In a production system you might also check for prompt injection,
        off-topic queries, or language detection. We keep it simple here.
        """
        query = ev.get("query", "").strip()
        strategy = ev.get("strategy", "semantic").strip().lower()

        if not query:
            raise ValueError("Query cannot be empty.")
        if len(query.split()) < 3:
            raise ValueError(
                f"Query too short ({len(query.split())} words). "
                "Please ask a more specific question."
            )
        if strategy not in VALID_STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Valid options: {', '.join(sorted(VALID_STRATEGIES))}"
            )
        if self.index is None or self.nodes is None:
            raise RuntimeError("Workflow not initialized — set .index and .nodes first.")

        return QueryValidated(query=query, strategy=strategy)

    # ── STEP 2: RETRIEVE ─────────────────────────────────────────────────

    @step
    async def retrieve(self, ev: QueryValidated) -> DocumentsReady | RetrievalDone:
        """Fetches relevant chunks using the selected strategy.

        CONCEPT: CONDITIONAL EVENT EMISSION
        This is the branching point. The return TYPE tells LlamaIndex
        which downstream step to trigger:
        - DocumentsReady → goes to synthesize (keyword/semantic/hybrid)
        - RetrievalDone → goes to rerank (reranking strategy only)

        In LangGraph, you'd write this as a conditional_edge function.
        Here, the step itself decides which event to emit. The effect
        is the same — different execution paths based on logic — but
        the mechanism is events, not edges.
        """
        query = ev.query
        strategy = ev.strategy

        if strategy == "keyword":
            results = await retrieve_keyword(self.nodes, query)
        elif strategy == "semantic":
            results = await retrieve_semantic(self.index, query)
        elif strategy == "hybrid":
            results = await retrieve_hybrid(self.index, self.nodes, query)
        elif strategy == "reranking":
            # For reranking: get initial candidates, then let the rerank step score them
            results = await retrieve_for_reranking(self.index, query)
            return RetrievalDone(query=query, documents=results)

        # For non-reranking strategies: documents are ready for synthesis
        return DocumentsReady(query=query, documents=results)

    # ── STEP 3: RERANK (conditional) ─────────────────────────────────────

    @step
    async def rerank(self, ev: RetrievalDone) -> DocumentsReady:
        """LLM-based reranking of initial retrieval candidates.

        This step ONLY fires for the reranking strategy (because only
        the reranking path emits RetrievalDone). For other strategies,
        this step never runs — they emit DocumentsReady directly.

        After reranking, we emit DocumentsReady — the same event that
        non-reranking strategies emit. This is the CONVERGENCE POINT.
        The synthesize step downstream doesn't know or care whether
        reranking happened.
        """
        reranked = await rerank_with_llm(ev.query, ev.documents)
        return DocumentsReady(query=ev.query, documents=reranked)

    # ── STEP 4: SYNTHESIZE ───────────────────────────────────────────────

    @step
    async def synthesize(self, ev: DocumentsReady | ResynthesizeRequest) -> SynthesisDone:
        """LLM generates an answer from the retrieved chunks.

        CONCEPT: DUAL-LISTEN PATTERN
        This step listens for TWO event types:
        - DocumentsReady: first attempt at synthesis
        - ResynthesizeRequest: retry after quality check failed

        This is how LlamaIndex Workflows handle loops. The step fires
        for EITHER event. When it's a retry, the event includes feedback
        about what to improve.

        Compare to P3 where the writer node checked state["revision_feedback"]
        to know if it was a first draft or a revision. Same concept, different
        mechanism.
        """
        query = ev.query
        documents = ev.documents

        if not documents:
            return SynthesisDone(
                query=query,
                answer="No relevant information found in the document for this query.",
                documents=[],
            )

        # Build context from retrieved chunks
        context_parts = []
        for i, doc in enumerate(documents, 1):
            text = doc.node.get_content()
            page = doc.node.metadata.get("page", "?")
            context_parts.append(f"[Chunk {i}, p.{page}]: {text}")
        context = "\n\n".join(context_parts)

        # Build the prompt — include feedback if this is a retry
        feedback_section = ""
        if isinstance(ev, ResynthesizeRequest):
            feedback_section = (
                f"\n\nPREVIOUS ATTEMPT FEEDBACK: {ev.feedback}\n"
                "Please address this feedback in your revised answer.\n"
            )

        prompt = (
            f"Based on the following document excerpts, answer the question.\n"
            f"Use ONLY information from the excerpts. If the excerpts don't "
            f"contain enough information, say so.\n"
            f"{feedback_section}\n"
            f"Question: {query}\n\n"
            f"Document excerpts:\n{context}\n\n"
            f"Answer:"
        )

        llm = Ollama(model=LLM_MODEL, request_timeout=600)
        response = await llm.acomplete(prompt)

        # TODO: re-enable quality loop once basic pipeline is validated
        # return SynthesisDone(
        #     query=query,
        #     answer=response.text.strip(),
        #     documents=documents,
        # )
        return StopEvent(
            result={
                "answer": response.text.strip(),
                "documents": documents,
                "was_resynthesized": False,
                "low_confidence": False,
            }
        )

    # ── STEP 5: EVALUATE (DISABLED) ────────────────────────────────────────
    #
    # TEMPORARILY DISABLED: The evaluate step adds an extra LLM call per query,
    # which causes timeouts with local Ollama models. The core pipeline
    # (validate → retrieve → synthesize) works without it.
    #
    # TO RE-ENABLE:
    # 1. Change synthesize's return to emit SynthesisDone instead of StopEvent
    # 2. Uncomment the evaluate_answer step below
    # 3. The quality loop (ResynthesizeRequest) will work automatically
    #
    # CONCEPT: QUALITY GATE (same pattern as P3's editor → writer loop)
    # The LLM evaluates its own output: "does this answer address the question
    # and use the provided context?" If quality is low, it emits
    # ResynthesizeRequest with feedback, looping back to synthesize.
    # Safety cap: MAX_RETRIES = 2 prevents infinite loops.
    #
    # @step
    # async def evaluate_answer(self, ev: SynthesisDone) -> StopEvent | ResynthesizeRequest:
    #     query = ev.query
    #     answer = ev.answer
    #     documents = ev.documents
    #
    #     if not documents:
    #         return StopEvent(
    #             result={
    #                 "answer": answer,
    #                 "documents": documents,
    #                 "was_resynthesized": False,
    #                 "low_confidence": False,
    #             }
    #         )
    #
    #     retry_count = getattr(ev, "_retry_count", 0)
    #
    #     context_summary = "\n".join(
    #         doc.node.get_content()[:200] for doc in documents[:3]
    #     )
    #     eval_prompt = (
    #         f"Evaluate if this answer adequately addresses the question using "
    #         f"the provided context.\n\n"
    #         f"Question: {query}\n"
    #         f"Answer: {answer}\n"
    #         f"Context (summary): {context_summary}\n\n"
    #         f"Rate the answer quality: GOOD or POOR.\n"
    #         f"If POOR, explain what's wrong in one sentence.\n"
    #         f"Format: GOOD or POOR: <reason>"
    #     )
    #
    #     llm = Ollama(model=LLM_MODEL, request_timeout=600)
    #     evaluation = await llm.acomplete(eval_prompt)
    #     eval_text = evaluation.text.strip().upper()
    #
    #     is_good = eval_text.startswith("GOOD")
    #
    #     if is_good or retry_count >= MAX_RETRIES:
    #         return StopEvent(
    #             result={
    #                 "answer": answer,
    #                 "documents": documents,
    #                 "was_resynthesized": retry_count > 0,
    #                 "low_confidence": not is_good and retry_count >= MAX_RETRIES,
    #             }
    #         )
    #
    #     feedback = eval_text.replace("POOR:", "").replace("POOR", "").strip()
    #     if not feedback:
    #         feedback = "The answer doesn't adequately address the question using the context."
    #
    #     return ResynthesizeRequest(
    #         query=query,
    #         documents=documents,
    #         feedback=feedback,
    #         retry_count=retry_count + 1,
    #     )
