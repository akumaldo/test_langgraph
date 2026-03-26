from __future__ import annotations

import unittest

from langgraph_portfolio.core import BaseWorkflowState, Field, GraphBuilder, KnowledgeBase, Transition
from langgraph_portfolio.core.fixtures import sample_documents


class ExampleState(BaseWorkflowState):
    value: int = Field(default=0)


class CoreRuntimeTest(unittest.TestCase):
    def test_state_model_dump_and_copy(self) -> None:
        state = ExampleState(run_id="abc", value=1)
        copied = state.model_copy(update={"value": 2})
        self.assertEqual(state.model_dump()["value"], 1)
        self.assertEqual(copied.value, 2)

    def test_graph_routing(self) -> None:
        def start_node(state: ExampleState) -> Transition[ExampleState]:
            return Transition(updates={"value": state.value + 1}, next_node="finish")

        def finish_node(state: ExampleState) -> Transition[ExampleState]:
            return Transition(updates={"status": "done"})

        graph = (
            GraphBuilder[ExampleState]()
            .add_node("start", start_node)
            .add_node("finish", finish_node)
            .set_entrypoint("start")
            .build()
        )
        result = graph.run(ExampleState(value=0))
        self.assertEqual(result.value, 1)
        self.assertEqual(result.status, "done")

    def test_knowledge_base_search(self) -> None:
        kb = KnowledgeBase()
        kb.index(sample_documents())
        hits = kb.search("LlamaIndex retrieval", limit=2)
        self.assertGreaterEqual(len(hits), 1)
        self.assertIn("LlamaIndex", hits[0].document.title)


if __name__ == "__main__":
    unittest.main()

