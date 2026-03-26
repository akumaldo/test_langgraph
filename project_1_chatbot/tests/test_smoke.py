from __future__ import annotations

import unittest

from langgraph_portfolio.projects.project_1_chatbot.graph import run_chatbot


class ChatbotSmokeTest(unittest.TestCase):
    def test_compare_intent_routes_to_response(self) -> None:
        state = run_chatbot(["Please compare LangGraph and CrewAI"])
        self.assertEqual(state.intent, "comparison")
        self.assertIn("compare", state.response.lower())
        self.assertEqual(state.status, "complete")


if __name__ == "__main__":
    unittest.main()

