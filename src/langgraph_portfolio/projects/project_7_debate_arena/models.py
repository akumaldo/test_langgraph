"""
Project 7 — Pydantic models for debate scoring.

The Judge agent scores each debater's arguments and declares a winner.
These models define the SHAPE of the judge's scoring output.

Unlike P4's structured output (where LangGraph forced the LLM to return
Pydantic-compatible JSON via with_structured_output()), here we parse the
judge's text output manually. AG2 doesn't have a built-in structured output
mechanism like LangGraph does — agents communicate in natural language.

This is a key framework difference:
  LangGraph:  with_structured_output(AnalysisResult) → guaranteed schema
  CrewAI:     output_pydantic=Model → Instructor-based (broken with Ollama)
  AG2:        no built-in structured output → parse text manually

The models are still useful as documentation of the expected output shape,
and we use them to structure the final result after parsing the judge's text.
"""

from pydantic import BaseModel, Field


class DebaterScore(BaseModel):
    """Score for a single debater.

    The judge assigns a score (1-10) and provides reasoning.
    Compare to P4's AnalysisResult — both structure LLM output,
    but P4 used with_structured_output() while P7 parses text.
    """
    name: str = Field(description="The debater's name (e.g., 'Debater_Pro')")
    score: int = Field(description="Score from 1-10", ge=1, le=10)
    reasoning: str = Field(description="Why this score was given")


class DebateResult(BaseModel):
    """The final result of a debate.

    Structured summary of the entire debate — who won, scores, and
    the key arguments from each side.
    """
    topic: str = Field(description="The debate topic")
    scores: list[DebaterScore] = Field(description="Scores for each debater")
    winner: str = Field(description="Name of the winning debater")
    summary: str = Field(description="Brief summary of the debate")
    total_rounds: int = Field(description="How many rounds the debate lasted")
