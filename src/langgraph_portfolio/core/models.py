from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .compatibility import BaseModel, Field


@dataclass(slots=True)
class ChatMessage:
    role: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class BaseWorkflowState(BaseModel):
    run_id: str = ""
    status: str = "idle"
    history: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

