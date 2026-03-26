from __future__ import annotations

from typing import Any

from ...core import BaseWorkflowState, Field


class AnalysisState(BaseWorkflowState):
    question: str = ""
    plan: list[str] = Field(default_factory=list)
    executed_steps: list[str] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    retries: int = 0
    progress: float = 0.0
    checkpoint_path: str = ""
    report: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

