from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RepoStructureTest(unittest.TestCase):
    def test_expected_directories_exist(self) -> None:
        expected = [
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
