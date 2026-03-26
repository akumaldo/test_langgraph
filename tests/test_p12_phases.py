from __future__ import annotations
import unittest


class GatheringPhaseTest(unittest.TestCase):
    """Test the data gathering phase builds a valid CrewAI crew."""

    def test_build_gathering_crew_returns_crew(self) -> None:
        from langgraph_portfolio.projects.project_12_analyst.phases.gathering import (
            build_gathering_crew,
        )

        crew = build_gathering_crew("AAPL")
        self.assertIsNotNone(crew)
        self.assertEqual(len(crew.agents), 3)
        self.assertEqual(len(crew.tasks), 3)

    def test_gather_node_function_exists(self) -> None:
        from langgraph_portfolio.projects.project_12_analyst.phases.gathering import (
            gather_data,
        )

        self.assertTrue(callable(gather_data))


if __name__ == "__main__":
    unittest.main()
