from __future__ import annotations
import unittest


class LLMFactoryTest(unittest.TestCase):
    """Test that the LLM factory produces a ChatOllama instance."""

    def test_get_llm_returns_chat_ollama(self) -> None:
        from langgraph_portfolio.projects.project_12_analyst.llm import get_llm

        llm = get_llm()
        self.assertEqual(llm.model, "qwen3.5:35b")
        self.assertIn("localhost", llm.base_url)

    def test_get_llm_respects_env_vars(self) -> None:
        import os
        from langgraph_portfolio.projects.project_12_analyst.llm import get_llm

        os.environ["OLLAMA_BASE_URL"] = "https://ollama.com"
        os.environ["OLLAMA_API_KEY"] = "test-key-123"
        try:
            llm = get_llm()
            self.assertIn("ollama.com", llm.base_url)
        finally:
            del os.environ["OLLAMA_BASE_URL"]
            del os.environ["OLLAMA_API_KEY"]


class StateTest(unittest.TestCase):
    """Test that InvestmentState can be constructed with all fields."""

    def test_state_construction(self) -> None:
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
            "data_summary": "",
            "profitability_analysis": "",
            "valuation_analysis": "",
            "growth_analysis": "",
            "analysis_summary": "",
            "debate_transcript": "",
            "key_disagreements": [],
            "investment_thesis": "",
            "risk_factors": [],
            "catalysts": [],
            "confidence_level": "",
            "report_path": "",
            "current_phase": "gathering",
            "human_feedback": "",
            "messages": [],
        }
        self.assertEqual(state["ticker"], "AAPL")
        self.assertEqual(state["current_phase"], "gathering")


class OrchestratorTest(unittest.TestCase):
    """Test that the parent graph compiles with all phases wired in."""

    def test_parent_graph_compiles(self) -> None:
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph_portfolio.projects.project_12_analyst.orchestrator import (
            build_analyst_graph,
        )

        checkpointer = MemorySaver()
        app = build_analyst_graph(checkpointer=checkpointer)
        self.assertIsNotNone(app)

    def test_route_after_interrupt_function(self) -> None:
        from langgraph_portfolio.projects.project_12_analyst.orchestrator import (
            route_after_review,
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
            "data_summary": "",
            "profitability_analysis": "",
            "valuation_analysis": "",
            "growth_analysis": "",
            "analysis_summary": "",
            "debate_transcript": "",
            "key_disagreements": [],
            "investment_thesis": "",
            "risk_factors": [],
            "catalysts": [],
            "confidence_level": "",
            "report_path": "",
            "current_phase": "analysis",
            "human_feedback": "",
            "messages": [],
        }
        result = route_after_review(state)
        self.assertIn(result, ["analyze", "debate", "thesis", "report", "gather"])


if __name__ == "__main__":
    unittest.main()
