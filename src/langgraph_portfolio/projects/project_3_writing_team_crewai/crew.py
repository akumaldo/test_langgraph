from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ...core.fixtures import sample_outline
from .state import CrewWritingState


@dataclass(slots=True)
class CrewAgent:
    name: str
    role: str
    executor: Callable[[CrewWritingState], CrewWritingState]

    def execute(self, state: CrewWritingState) -> CrewWritingState:
        return self.executor(state)


class CrewWorkflow:
    def __init__(self) -> None:
        self.agents = [
            CrewAgent(name="planner", role="planner", executor=self._plan),
            CrewAgent(name="researcher", role="researcher", executor=self._research),
            CrewAgent(name="writer", role="writer", executor=self._write),
            CrewAgent(name="editor", role="editor", executor=self._edit),
            CrewAgent(name="publisher", role="publisher", executor=self._publish),
        ]

    def run(self, state: CrewWritingState) -> CrewWritingState:
        current = state.model_copy(update={"metadata": {**state.metadata, "orchestrator": "crewai"}})
        current.history.append("crew_start")
        for agent in self.agents:
            current = agent.execute(current)
        current.status = "complete"
        current.history.append("crew_end")
        return current

    def _plan(self, state: CrewWritingState) -> CrewWritingState:
        state.outline = sample_outline()
        state.history.append("planner")
        state.status = "planned"
        return state

    def _research(self, state: CrewWritingState) -> CrewWritingState:
        state.research_notes = [f"CrewAI research note for: {section}" for section in state.outline]
        state.history.append("researcher")
        state.status = "researched"
        return state

    def _write(self, state: CrewWritingState) -> CrewWritingState:
        state.draft_sections = {
            section: f"{section}: {note}" for section, note in zip(state.outline, state.research_notes, strict=False)
        }
        state.history.append("writer")
        state.status = "drafted"
        return state

    def _edit(self, state: CrewWritingState) -> CrewWritingState:
        state.editor_feedback = ["Clarify the team coordination model", "Highlight task delegation"]
        state.history.append("editor")
        state.status = "edited"
        return state

    def _publish(self, state: CrewWritingState) -> CrewWritingState:
        sections = ["# Writing Team Article (CrewAI)"]
        sections.extend(f"## {title}\n{body}" for title, body in state.draft_sections.items())
        sections.append("## Editorial Notes")
        sections.extend(f"- {note}" for note in state.editor_feedback)
        state.final_article = "\n\n".join(sections)
        state.history.append("publisher")
        return state


def run_writing_team(topic: str) -> CrewWritingState:
    state = CrewWritingState(topic=topic, status="starting")
    return CrewWorkflow().run(state)

