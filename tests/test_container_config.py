from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ContainerConfigTest(unittest.TestCase):
    def test_dockerfile_points_at_scaffold(self) -> None:
        dockerfile = (ROOT / "Dockerfile").read_text()
        self.assertIn("python:3.14-slim", dockerfile)
        self.assertIn("APP_MODULE=langgraph_portfolio.projects.project_1_chatbot.main:main", dockerfile)
        self.assertIn("scripts/run_project.py", dockerfile)
        self.assertIn("COPY src ./src", dockerfile)


if __name__ == "__main__":
    unittest.main()

