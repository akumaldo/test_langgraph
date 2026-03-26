from __future__ import annotations

import unittest

from langgraph_portfolio.projects.project_3_writing_team_crewai.crew import run_writing_team


class WritingTeamCrewAISmokeTest(unittest.TestCase):
    def test_article_is_created(self) -> None:
        state = run_writing_team("Compare LangGraph and CrewAI")
        self.assertIn("CrewAI", state.final_article)
        self.assertEqual(state.status, "complete")
        self.assertGreater(len(state.outline), 0)


if __name__ == "__main__":
    unittest.main()

