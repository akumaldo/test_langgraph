from __future__ import annotations

from json import dumps, loads
from pathlib import Path

from ...core import GraphBuilder, Transition
from .state import AnalysisState


def decompose_task_node(state: AnalysisState) -> Transition[AnalysisState]:
    plan = [
        "Understand the question",
        "Run the analysis",
        "Generate a visualization",
        "Write the report",
    ]
    return Transition(
        updates={
            "history": state.history + ["decompose_task_node"],
            "plan": plan,
            "status": "planned",
            "progress": 0.25,
        },
        next_node="execute_analysis_node",
    )


def execute_analysis_node(state: AnalysisState) -> Transition[AnalysisState]:
    should_fail = bool(state.metadata.get("simulate_failure")) and state.retries == 0
    if should_fail:
        return Transition(
            updates={
                "history": state.history + ["execute_analysis_node"],
                "errors": state.errors + ["Initial analysis execution failed"],
                "status": "needs_recovery",
                "progress": 0.35,
            },
            next_node="recover_from_error_node",
        )

    executed_steps = state.executed_steps + ["Loaded and analyzed the structured dataset"]
    return Transition(
        updates={
            "history": state.history + ["execute_analysis_node"],
            "executed_steps": executed_steps,
            "status": "analyzed",
            "progress": 0.6,
        },
        next_node="visualize_results_node",
    )


def recover_from_error_node(state: AnalysisState) -> Transition[AnalysisState]:
    return Transition(
        updates={
            "history": state.history + ["recover_from_error_node"],
            "retries": state.retries + 1,
            "metadata": {**state.metadata, "simulate_failure": False},
            "status": "recovered",
            "progress": 0.5,
        },
        next_node="execute_analysis_node",
    )


def visualize_results_node(state: AnalysisState) -> Transition[AnalysisState]:
    artifacts = state.artifacts + ["analysis_chart.png"]
    return Transition(
        updates={
            "history": state.history + ["visualize_results_node"],
            "artifacts": artifacts,
            "status": "visualized",
            "progress": 0.8,
        },
        next_node="report_node",
    )


def report_node(state: AnalysisState) -> Transition[AnalysisState]:
    report = "\n".join(
        [
            "# Analysis Report",
            f"Question: {state.question}",
            f"Plan steps: {len(state.plan)}",
            f"Executed steps: {len(state.executed_steps)}",
            f"Artifacts: {', '.join(state.artifacts) if state.artifacts else 'none'}",
            f"Retries: {state.retries}",
        ]
    )
    return Transition(
        updates={
            "history": state.history + ["report_node"],
            "report": report,
            "status": "complete",
            "progress": 1.0,
        },
        next_node="end_node",
    )


def end_node(state: AnalysisState) -> Transition[AnalysisState]:
    return Transition(updates={"history": state.history + ["end_node"]})


def save_checkpoint(state: AnalysisState, path: Path) -> Path:
    path.write_text(dumps(state.model_dump(), indent=2, sort_keys=True, default=str))
    return path


def load_checkpoint(path: Path) -> AnalysisState:
    payload = loads(path.read_text())
    return AnalysisState.model_validate(payload)


def build_graph() -> GraphBuilder[AnalysisState]:
    return (
        GraphBuilder[AnalysisState]()
        .add_node("decompose_task_node", decompose_task_node)
        .add_node("execute_analysis_node", execute_analysis_node)
        .add_node("recover_from_error_node", recover_from_error_node)
        .add_node("visualize_results_node", visualize_results_node)
        .add_node("report_node", report_node)
        .add_node("end_node", end_node)
        .set_entrypoint("decompose_task_node")
    )


def run_analysis(question: str, *, simulate_failure: bool = False, checkpoint_path: Path | None = None) -> AnalysisState:
    state = AnalysisState(
        question=question,
        status="starting",
        metadata={"simulate_failure": simulate_failure},
        checkpoint_path=str(checkpoint_path) if checkpoint_path is not None else "",
    )
    result = build_graph().build(max_steps=16).run(state)
    if checkpoint_path is not None:
        save_checkpoint(result, checkpoint_path)
    return result

