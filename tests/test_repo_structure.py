from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RepoStructureTest(unittest.TestCase):
    def test_expected_directories_exist(self) -> None:
        expected = [
            ROOT / "project_1_chatbot",
            ROOT / "project_2_research_agent",
            ROOT / "project_3_writing_team_langgraph",
            ROOT / "project_3_writing_team_crewai",
            ROOT / "project_4_data_analyst",
            ROOT / "project_5_support_bot",
            ROOT / "src" / "langgraph_portfolio" / "core",
            ROOT / "src" / "langgraph_portfolio" / "projects",
            ROOT / "docs",
        ]
        for path in expected:
            self.assertTrue(path.exists(), msg=f"Missing expected path: {path}")

    def test_expected_docs_exist(self) -> None:
        expected = [ROOT / "docs" / "architecture.md", ROOT / "docs" / "comparisons.md"]
        for path in expected:
            self.assertTrue(path.exists(), msg=f"Missing expected doc: {path}")


if __name__ == "__main__":
    unittest.main()

