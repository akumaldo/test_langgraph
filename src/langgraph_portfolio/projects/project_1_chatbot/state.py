from __future__ import annotations

from typing import Any

from ...core import BaseWorkflowState, ChatMessage, Field


class ChatState(BaseWorkflowState):
    messages: list[ChatMessage] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)
    intent: str | None = None
    confidence: float = 0.0
    conversation_turns: int = 0
    response: str = ""
    clarification_needed: bool = False

