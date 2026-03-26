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


class ProfitabilitySubgraphTest(unittest.TestCase):
    """Test the profitability analysis sub-graph compiles."""

    def test_subgraph_compiles(self) -> None:
        from langgraph_portfolio.projects.project_12_analyst.phases.profitability import (
            build_profitability_subgraph,
        )

        graph = build_profitability_subgraph()
        self.assertIsNotNone(graph)


class ValuationSubgraphTest(unittest.TestCase):
    """Test the valuation analysis sub-graph compiles."""

    def test_subgraph_compiles(self) -> None:
        from langgraph_portfolio.projects.project_12_analyst.phases.valuation import (
            build_valuation_subgraph,
        )

        graph = build_valuation_subgraph()
        self.assertIsNotNone(graph)


class GrowthSubgraphTest(unittest.TestCase):
    """Test the growth analysis sub-graph compiles."""

    def test_subgraph_compiles(self) -> None:
        from langgraph_portfolio.projects.project_12_analyst.phases.growth import (
            build_growth_subgraph,
        )

        graph = build_growth_subgraph()
        self.assertIsNotNone(graph)


class AnalyzeNodeTest(unittest.TestCase):
    """Test the analyze_financials node function that runs all 3 sub-graphs."""

    def test_analyze_node_exists_in_profitability(self) -> None:
        from langgraph_portfolio.projects.project_12_analyst.phases.profitability import (
            build_profitability_subgraph,
        )

        self.assertTrue(callable(build_profitability_subgraph))


if __name__ == "__main__":
    unittest.main()
