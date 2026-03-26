from __future__ import annotations
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import asyncio


def run(coro):
    """Helper to run async tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


class FMPClientTest(unittest.TestCase):
    """Test FMP wrappers with mocked HTTP responses."""

    @patch("langgraph_portfolio.projects.project_12_analyst.tools.fmp.httpx.AsyncClient")
    def test_get_income_statement(self, mock_client_cls) -> None:
        from langgraph_portfolio.projects.project_12_analyst.tools.fmp import FMPClient

        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"date": "2025-09-30", "revenue": 385000000000, "netIncome": 97000000000}
        ]
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        client = FMPClient(api_key="test-key")
        result = run(client.get_income_statement("AAPL"))

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["revenue"], 385000000000)
        mock_client.get.assert_called_once()
        call_url = mock_client.get.call_args[0][0]
        self.assertIn("income-statement", call_url)
        self.assertIn("AAPL", call_url)

    @patch("langgraph_portfolio.projects.project_12_analyst.tools.fmp.httpx.AsyncClient")
    def test_get_key_metrics_ttm(self, mock_client_cls) -> None:
        from langgraph_portfolio.projects.project_12_analyst.tools.fmp import FMPClient

        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"peRatioTTM": 28.5, "roeTTM": 1.47, "dividendYielTTM": 0.005}
        ]
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        client = FMPClient(api_key="test-key")
        result = run(client.get_key_metrics_ttm("AAPL"))

        self.assertEqual(result[0]["peRatioTTM"], 28.5)

    def test_client_requires_api_key(self) -> None:
        from langgraph_portfolio.projects.project_12_analyst.tools.fmp import FMPClient

        with self.assertRaises(ValueError):
            FMPClient(api_key="")


if __name__ == "__main__":
    unittest.main()
