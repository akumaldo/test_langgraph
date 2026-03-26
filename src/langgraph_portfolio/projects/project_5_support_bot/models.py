"""
Project 5 — Pydantic models for structured LLM output.

Compare to P4's models.py which had AnalysisPlan, CodeFix, ChartPlan, AnalysisReport.
P5 is simpler — we only need ONE structured output model: IssueClassification.

Why use structured output here?
  Without it: the classifier returns "I think this is a billing issue because..."
              and you have to PARSE that text to extract the category.
  With it:    the classifier returns IssueClassification(category="billing", ...)
              and you get a typed object with guaranteed fields.

Same pattern as P4: llm.with_structured_output(IssueClassification).invoke(...)
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class IssueClassification(BaseModel):
    """Structured output from the classifier node.

    The LLM returns this object (not a string) thanks to with_structured_output().
    """

    category: Literal["billing", "technical", "returns", "unknown"] = Field(
        description="The department this issue belongs to"
    )
    confidence: float = Field(
        description="How confident the classification is, 0.0 to 1.0"
    )
    reasoning: str = Field(
        description="Brief explanation of why this category was chosen"
    )
