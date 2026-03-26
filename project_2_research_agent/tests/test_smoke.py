from __future__ import annotations

import unittest

from langgraph_portfolio.projects.project_2_research_agent.graph import run_research_agent


class ResearchAgentSmokeTest(unittest.TestCase):
    def test_retrieval_and_summary(self) -> None:
        state = run_research_agent("How does LlamaIndex support document retrieval?")
        self.assertGreater(len(state.documents), 0)
        self.assertGreater(len(state.retrieved_documents), 0)
        self.assertIn("Research summary", state.summary)


if __name__ == "__main__":
    unittest.main()

