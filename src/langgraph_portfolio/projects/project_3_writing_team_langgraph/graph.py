from __future__ import annotations

from ...core import GraphBuilder, Transition
from ...core.fixtures import sample_outline
from .state import WritingTeamState


def planner_node(state: WritingTeamState) -> Transition[WritingTeamState]:
    outline = sample_outline()
    return Transition(
        updates={
            "history": state.history + ["planner_node"],
            "outline": outline,
            "status": "planned",
            "metadata": {**state.metadata, "orchestrator": "langgraph"},
        },
        next_node="research_node",
    )


def research_node(state: WritingTeamState) -> Transition[WritingTeamState]:
    research_notes = [f"Research note for: {section}" for section in state.outline]
    return Transition(
        updates={
            "history": state.history + ["research_node"],
            "research_notes": research_notes,
            "status": "researched",
        },
        next_node="writer_node",
    )


def writer_node(state: WritingTeamState) -> Transition[WritingTeamState]:
    draft_sections = {
        section: f"{section}: {note}" for section, note in zip(state.outline, state.research_notes, strict=False)
    }
    return Transition(
        updates={
            "history": state.history + ["writer_node"],
            "draft_sections": draft_sections,
            "status": "drafted",
        },
        next_node="editor_node",
    )


def editor_node(state: WritingTeamState) -> Transition[WritingTeamState]:
    feedback = ["Tighten the introduction", "Add one concrete example"]
    return Transition(
        updates={
            "history": state.history + ["editor_node"],
            "editor_feedback": feedback,
            "status": "edited",
        },
        next_node="publisher_node",
    )


def publisher_node(state: WritingTeamState) -> Transition[WritingTeamState]:
    article_sections = ["# Writing Team Article"]
    article_sections.extend(f"## {title}\n{body}" for title, body in state.draft_sections.items())
    article_sections.append("## Editorial Notes")
    article_sections.extend(f"- {note}" for note in state.editor_feedback)
    return Transition(
        updates={
            "history": state.history + ["publisher_node"],
            "final_article": "\n\n".join(article_sections),
            "status": "complete",
        },
        next_node="end_node",
    )


def end_node(state: WritingTeamState) -> Transition[WritingTeamState]:
    return Transition(updates={"history": state.history + ["end_node"]})


def build_graph() -> GraphBuilder[WritingTeamState]:
    return (
        GraphBuilder[WritingTeamState]()
        .add_node("planner_node", planner_node)
        .add_node("research_node", research_node)
        .add_node("writer_node", writer_node)
        .add_node("editor_node", editor_node)
        .add_node("publisher_node", publisher_node)
        .add_node("end_node", end_node)
        .set_entrypoint("planner_node")
    )


def run_writing_team(topic: str) -> WritingTeamState:
    state = WritingTeamState(topic=topic, status="starting")
    return build_graph().build().run(state)

