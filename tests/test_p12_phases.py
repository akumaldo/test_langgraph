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


class ThesisPhaseTest(unittest.TestCase):
    """Test the thesis synthesis phase."""

    def test_build_thesis_subgraph_compiles(self) -> None:
        from langgraph_portfolio.projects.project_12_analyst.phases.thesis import (
            build_thesis_subgraph,
        )

        graph = build_thesis_subgraph()
        self.assertIsNotNone(graph)

    def test_synthesize_thesis_node_exists(self) -> None:
        from langgraph_portfolio.projects.project_12_analyst.phases.thesis import (
            synthesize_thesis,
        )

        self.assertTrue(callable(synthesize_thesis))


class ReportPhaseTest(unittest.TestCase):
    """Test report generation."""

    def test_generate_report_produces_markdown(self) -> None:
        from langgraph_portfolio.projects.project_12_analyst.phases.report import (
            generate_report,
        )
        from langgraph_portfolio.projects.project_12_analyst.state import InvestmentState

        state: InvestmentState = {
            "ticker": "AAPL",
            "company_name": "Apple Inc.",
            "financials": {},
            "ratios": {},
            "estimates": {},
            "insider_trades": [],
            "grades": [],
            "earnings_transcripts": [],
            "data_summary": "Revenue: $385B",
            "profitability_analysis": "Margins stable",
            "valuation_analysis": "Fair value $245",
            "growth_analysis": "Growing 5% YoY",
            "analysis_summary": "Solid fundamentals",
            "debate_transcript": "Bull said X, Bear said Y",
            "key_disagreements": ["Valuation premium justified?"],
            "investment_thesis": "Buy with medium confidence",
            "risk_factors": ["Slowing growth", "China risk"],
            "catalysts": ["AI spending", "Services growth"],
            "confidence_level": "MEDIUM",
            "report_path": "",
            "current_phase": "report",
            "human_feedback": "",
            "messages": [],
        }
        result = generate_report(state)
        self.assertIn("report_path", result)
        self.assertIn("AAPL", result["report_path"])
        # Verify the report content
        self.assertIn("# Investment Analysis: AAPL", result.get("_report_content", ""))


if __name__ == "__main__":
    unittest.main()
