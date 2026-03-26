from __future__ import annotations

import unittest

from langgraph_portfolio.projects.project_4_data_analyst.graph import run_analysis


class DataAnalystSmokeTest(unittest.TestCase):
    def test_recovery_path_produces_report(self) -> None:
        state = run_analysis("Forecast portfolio engagement", simulate_failure=True)
        self.assertGreaterEqual(state.retries, 1)
        self.assertIn("Analysis Report", state.report)
        self.assertEqual(state.status, "complete")


if __name__ == "__main__":
    unittest.main()

