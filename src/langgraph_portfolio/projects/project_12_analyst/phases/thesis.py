"""Phase 4: Thesis Synthesis — LangGraph + LlamaIndex RAG.

CONCEPT: On-demand RAG (from P9)
We only build the LlamaIndex index when the user reaches the thesis phase.
Earnings transcripts (fetched in Phase 1) are ingested into a temporary
in-memory vector index. Key claims from the debate are used as queries
to find supporting/contradicting evidence.

Flow: ingest_transcripts → query_evidence → draft_thesis

This is both a LangGraph sub-graph AND uses LlamaIndex internally.
The sub-graph nodes call LlamaIndex for retrieval and LangGraph's LLM
for synthesis.
"""

from __future__ import annotations

import asyncio
from typing import TypedDict

from langgraph.graph import END, StateGraph

from langgraph_portfolio.projects.project_12_analyst.llm import get_llm


# -- Sub-graph State --

class ThesisInput(TypedDict):
    ticker: str
    company_name: str
    data_summary: str
    analysis_summary: str
    debate_transcript: str
    key_disagreements: list
    earnings_transcripts: list


class ThesisOutput(TypedDict):
    investment_thesis: str
    risk_factors: list
    catalysts: list
    confidence_level: str


class ThesisState(TypedDict):
    ticker: str
    company_name: str
    data_summary: str
    analysis_summary: str
    debate_transcript: str
    key_disagreements: list
    earnings_transcripts: list
    rag_evidence: str
    investment_thesis: str
    risk_factors: list
    catalysts: list
    confidence_level: str


# -- Node Functions --

def ingest_and_query(state: ThesisState) -> dict:
    """Ingest transcripts into LlamaIndex index and query for evidence.

    Uses LlamaIndex's VectorStoreIndex for semantic search over
    earnings call transcripts. Queries key disagreements from the
    debate to find supporting/contradicting evidence.
    """
    transcripts = state.get("earnings_transcripts", [])
    key_disagreements = state.get("key_disagreements", [])

    if not transcripts:
        return {"rag_evidence": "No earnings transcripts available for RAG."}

    # Build transcript text for indexing
    transcript_texts = []
    for t in transcripts:
        if isinstance(t, list) and t:
            entry = t[0] if isinstance(t[0], dict) else t
        elif isinstance(t, dict):
            entry = t
        else:
            continue
        content = entry.get("content", str(entry))
        transcript_texts.append(str(content))

    if not transcript_texts:
        return {"rag_evidence": "No parseable transcript content found."}

    # Use LlamaIndex for RAG
    try:
        from llama_index.core import Document, Settings, VectorStoreIndex
        from llama_index.core.node_parser import SentenceSplitter

        # Build documents
        documents = [
            Document(text=text, metadata={"source": f"transcript_{i}"})
            for i, text in enumerate(transcript_texts)
        ]

        # Chunk documents
        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        nodes = splitter.get_nodes_from_documents(documents)

        if not nodes:
            return {"rag_evidence": "Transcript chunking produced no nodes."}

        # Build in-memory index
        try:
            from llama_index.embeddings.ollama import OllamaEmbedding
            Settings.embed_model = OllamaEmbedding(model_name="qwen3-embedding")
        except ImportError:
            pass  # Fall back to LlamaIndex default

        index = VectorStoreIndex(nodes=nodes)
        query_engine = index.as_query_engine(similarity_top_k=5)

        # Query each key disagreement for evidence
        evidence_parts = []
        queries = key_disagreements[:5] if key_disagreements else [
            "What is management's outlook?",
            "What are the key risks mentioned?",
        ]

        for query_text in queries:
            response = query_engine.query(str(query_text))
            evidence_parts.append(
                f"**Query:** {query_text}\n**Evidence:** {response.response}\n"
            )

        return {"rag_evidence": "\n---\n".join(evidence_parts)}

    except ImportError:
        # LlamaIndex not available — fall back to direct transcript excerpts
        excerpt = "\n\n".join(t[:3000] for t in transcript_texts)
        return {
            "rag_evidence": (
                f"[LlamaIndex not available — raw transcript excerpts]\n\n{excerpt}"
            )
        }


def synthesize_thesis(state: ThesisState) -> dict:
    """Draft the investment thesis using all accumulated evidence."""
    llm = get_llm(temperature=0.5)

    prompt = (
        f"You are the chief investment strategist. Write a comprehensive investment "
        f"thesis for {state['ticker']} ({state['company_name']}).\n\n"
        f"You have:\n"
        f"1. Financial data summary:\n{state['data_summary'][:3000]}\n\n"
        f"2. Detailed analysis:\n{state['analysis_summary'][:3000]}\n\n"
        f"3. Bull/Bear debate key disagreements:\n"
        f"{chr(10).join('- ' + d for d in state['key_disagreements'])}\n\n"
        f"4. Evidence from earnings transcripts:\n{state['rag_evidence'][:3000]}\n\n"
        "Write your thesis with these exact sections:\n"
        "INVESTMENT THESIS: 2-3 paragraph summary of your conclusion\n"
        "RISK FACTORS: Bullet list of 3-5 key risks\n"
        "CATALYSTS: Bullet list of 3-5 potential positive catalysts\n"
        "CONFIDENCE: One of HIGH, MEDIUM, or LOW with a brief justification\n"
    )
    response = llm.invoke(prompt)
    content = response.content

    # Parse structured sections from LLM output
    risk_factors = _extract_bullets(content, "RISK FACTORS")
    catalysts = _extract_bullets(content, "CATALYSTS")
    confidence = _extract_confidence(content)

    return {
        "investment_thesis": content,
        "risk_factors": risk_factors,
        "catalysts": catalysts,
        "confidence_level": confidence,
    }


def _extract_bullets(text: str, section_name: str) -> list[str]:
    """Extract bullet points from a named section."""
    import re

    pattern = rf"{section_name}[:\s]*\n((?:[-•*]\s*.+\n?)+)"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        bullets = re.findall(r"[-•*]\s*(.+)", match.group(1))
        return [b.strip() for b in bullets if b.strip()]
    return [f"See thesis for {section_name.lower()}"]


def _extract_confidence(text: str) -> str:
    """Extract confidence level from the thesis."""
    import re

    match = re.search(r"CONFIDENCE[:\s]*(HIGH|MEDIUM|LOW)", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return "MEDIUM"


# -- Build Sub-graph --

def build_thesis_subgraph():
    """Compile the thesis synthesis sub-graph."""
    graph = StateGraph(
        ThesisState,
        input=ThesisInput,
        output=ThesisOutput,
    )

    graph.add_node("ingest_and_query", ingest_and_query)
    graph.add_node("synthesize_thesis", synthesize_thesis)

    graph.set_entry_point("ingest_and_query")
    graph.add_edge("ingest_and_query", "synthesize_thesis")
    graph.add_edge("synthesize_thesis", END)

    return graph.compile()
