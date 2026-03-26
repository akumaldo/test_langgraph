from __future__ import annotations

from ...core import Document, GraphBuilder, InMemoryDocumentLoader, KnowledgeBase, Transition
from ...core.fixtures import sample_documents
from .state import ResearchState


def load_documents_node(state: ResearchState) -> Transition[ResearchState]:
    loader = InMemoryDocumentLoader(sample_documents())
    documents = list(loader.load())
    return Transition(
        updates={
            "history": state.history + ["load_documents_node"],
            "documents": documents,
            "status": "indexed",
            "metadata": {**state.metadata, "retrieval_engine": "LlamaIndex-ready"},
        },
        next_node="retrieve_documents_node",
    )


def retrieve_documents_node(state: ResearchState) -> Transition[ResearchState]:
    knowledge_base = KnowledgeBase()
    knowledge_base.index(state.documents)
    hits = knowledge_base.search(state.query, limit=3)
    retrieved_documents = [hit.document for hit in hits]
    citations = [f"{document.title} ({document.source})" for document in retrieved_documents]
    return Transition(
        updates={
            "history": state.history + ["retrieve_documents_node"],
            "retrieved_documents": retrieved_documents,
            "citations": citations,
            "metadata": {**state.metadata, "retrieved_count": len(retrieved_documents)},
        },
        next_node="synthesize_summary_node",
    )


def synthesize_summary_node(state: ResearchState) -> Transition[ResearchState]:
    if not state.retrieved_documents:
        summary = "No supporting documents were found for the query."
    else:
        lines = [
            f"- {document.title}: {document.content}" for document in state.retrieved_documents
        ]
        summary = "Research summary:\n" + "\n".join(lines)
    return Transition(
        updates={
            "history": state.history + ["synthesize_summary_node"],
            "summary": summary,
            "status": "summarized",
        },
        next_node="end_node",
    )


def end_node(state: ResearchState) -> Transition[ResearchState]:
    return Transition(updates={"history": state.history + ["end_node"], "status": "complete"})


def build_graph() -> GraphBuilder[ResearchState]:
    return (
        GraphBuilder[ResearchState]()
        .add_node("load_documents_node", load_documents_node)
        .add_node("retrieve_documents_node", retrieve_documents_node)
        .add_node("synthesize_summary_node", synthesize_summary_node)
        .add_node("end_node", end_node)
        .set_entrypoint("load_documents_node")
    )


def run_research_agent(query: str) -> ResearchState:
    state = ResearchState(query=query, status="starting")
    return build_graph().build().run(state)

