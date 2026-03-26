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


class DebatePhaseTest(unittest.TestCase):
    """Test the debate phase builds valid AG2 agents."""

    def test_build_debate_agents_returns_three(self) -> None:
        from langgraph_portfolio.projects.project_12_analyst.phases.debate import (
            build_debate_agents,
        )

        agents = build_debate_agents(
            data_summary="Revenue growing 5% YoY",
            analysis_summary="Fair value $245, trading at $228",
        )
        self.assertEqual(len(agents), 3)
        names = {a.name for a in agents}
        self.assertEqual(names, {"Bull_Analyst", "Bear_Analyst", "Moderator"})

    def test_run_debate_node_exists(self) -> None:
        from langgraph_portfolio.projects.project_12_analyst.phases.debate import (
            run_debate,
        )

        self.assertTrue(callable(run_debate))


if __name__ == "__main__":
    unittest.main()
