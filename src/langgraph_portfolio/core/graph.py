from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field as dataclass_field, is_dataclass, replace
from typing import Any, Generic, TypeVar


StateT = TypeVar("StateT")


@dataclass(frozen=True, slots=True)
class Transition(Generic[StateT]):
    updates: Mapping[str, Any] = dataclass_field(default_factory=dict)
    next_node: str | None = None


def _merge_state(state: StateT, updates: Mapping[str, Any]) -> StateT:
    if hasattr(state, "model_copy"):
        return state.model_copy(update=dict(updates))
    if is_dataclass(state):
        return replace(state, **dict(updates))
    raise TypeError(f"Unsupported state type: {type(state)!r}")


class ScaffoldGraph(Generic[StateT]):
    def __init__(
        self,
        *,
        nodes: dict[str, Callable[[StateT], Any]],
        entrypoint: str,
        edges: dict[str, str] | None = None,
        finish: str | None = None,
        max_steps: int = 32,
    ) -> None:
        self._nodes = dict(nodes)
        self._entrypoint = entrypoint
        self._edges = dict(edges or {})
        self._finish = finish
        self._max_steps = max_steps

    def run(self, state: StateT, *, start: str | None = None) -> StateT:
        current = start or self._entrypoint
        steps = 0
        while current is not None:
            if self._finish is not None and current == self._finish:
                break
            if steps >= self._max_steps:
                raise RuntimeError(f"Graph exceeded max_steps={self._max_steps}")

            node = self._nodes[current]
            outcome = node(state)
            state, current = self._interpret_outcome(state, current, outcome)
            steps += 1
        return state

    def _interpret_outcome(self, state: StateT, current: str, outcome: Any) -> tuple[StateT, str | None]:
        default_next = self._edges.get(current, self._finish)
        if isinstance(outcome, Transition):
            updated = _merge_state(state, outcome.updates)
            return updated, outcome.next_node or default_next
        if hasattr(outcome, "model_dump"):
            return outcome, default_next
        if isinstance(outcome, Mapping):
            return _merge_state(state, outcome), default_next
        if isinstance(outcome, tuple) and len(outcome) == 2:
            maybe_state, next_node = outcome
            return maybe_state, next_node
        if outcome is None:
            return state, default_next
        return outcome, default_next


class GraphBuilder(Generic[StateT]):
    def __init__(self) -> None:
        self._nodes: dict[str, Callable[[StateT], Any]] = {}
        self._edges: dict[str, str] = {}
        self._entrypoint: str | None = None
        self._finish: str | None = None

    def add_node(self, name: str, node: Callable[[StateT], Any]) -> "GraphBuilder[StateT]":
        self._nodes[name] = node
        return self

    def add_edge(self, source: str, target: str) -> "GraphBuilder[StateT]":
        self._edges[source] = target
        return self

    def set_entrypoint(self, name: str) -> "GraphBuilder[StateT]":
        self._entrypoint = name
        return self

    def set_finish_point(self, name: str) -> "GraphBuilder[StateT]":
        self._finish = name
        return self

    def build(self, *, max_steps: int = 32) -> ScaffoldGraph[StateT]:
        if self._entrypoint is None:
            raise ValueError("An entrypoint must be configured before build()")
        return ScaffoldGraph(
            nodes=self._nodes,
            entrypoint=self._entrypoint,
            edges=self._edges,
            finish=self._finish,
            max_steps=max_steps,
        )

