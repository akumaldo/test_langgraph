from __future__ import annotations

from typing import Any

from ...core import BaseWorkflowState, Document, Field


class ResearchState(BaseWorkflowState):
    query: str = ""
    documents: list[Document] = Field(default_factory=list)
    retrieved_documents: list[Document] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    summary: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

