from __future__ import annotations

from typing import Any

from ...core import BaseWorkflowState, Field


class WritingTeamState(BaseWorkflowState):
    topic: str = ""
    outline: list[str] = Field(default_factory=list)
    research_notes: list[str] = Field(default_factory=list)
    draft_sections: dict[str, str] = Field(default_factory=dict)
    editor_feedback: list[str] = Field(default_factory=list)
    final_article: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

