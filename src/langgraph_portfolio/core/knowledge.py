from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(slots=True)
class Document:
    title: str
    content: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SearchHit:
    document: Document
    score: float


class DocumentLoader(Protocol):
    def load(self) -> Sequence[Document]:
        raise NotImplementedError


@dataclass(slots=True)
class InMemoryDocumentLoader:
    documents: Sequence[Document]

    def load(self) -> Sequence[Document]:
        return list(self.documents)


@dataclass(slots=True)
class KnowledgeBase:
    documents: list[Document] = field(default_factory=list)

    def index(self, documents: Iterable[Document]) -> None:
        self.documents.extend(documents)

    def search(self, query: str, *, limit: int = 3) -> list[SearchHit]:
        query_terms = {token.lower() for token in query.split() if token}
        scored: list[SearchHit] = []
        for document in self.documents:
            content_terms = {token.lower().strip(".,:;!?") for token in document.content.split()}
            title_terms = {token.lower().strip(".,:;!?") for token in document.title.split()}
            overlap = len(query_terms & (content_terms | title_terms))
            if overlap:
                score = overlap / max(len(query_terms), 1)
                scored.append(SearchHit(document=document, score=score))
        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:limit]


@dataclass(slots=True)
class FrameworkAvailability:
    langgraph: bool
    pydantic: bool
    llamaindex: bool
    crewai: bool

    @classmethod
    def detect(cls) -> "FrameworkAvailability":
        from .compatibility import framework_available

        return cls(
            langgraph=framework_available("langgraph"),
            pydantic=framework_available("pydantic"),
            llamaindex=framework_available("llama_index"),
            crewai=framework_available("crewai"),
        )

