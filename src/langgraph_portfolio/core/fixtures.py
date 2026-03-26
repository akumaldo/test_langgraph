from __future__ import annotations

from .knowledge import Document
from .models import ChatMessage


def sample_chat_messages() -> list[ChatMessage]:
    return [
        ChatMessage(role="user", content="Can you help me compare LangGraph and CrewAI?"),
        ChatMessage(role="user", content="I also need a research workflow."),
    ]


def sample_documents() -> list[Document]:
    return [
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
    ]


def sample_outline() -> list[str]:
    return [
        "Problem framing",
        "Framework tradeoffs",
        "Implementation guidance",
        "Portfolio wrap-up",
    ]

