"""FMP REST API wrappers — typed functions for Financial Modeling Prep.

CONCEPT: Service layer separation
Instead of having agents call HTTP endpoints directly, we wrap each FMP
endpoint in a typed async function. This gives us:
  1. Type safety — each function documents what it returns
  2. Testability — easy to mock at the function level
  3. Single source of truth — one place to handle auth, errors, rate limits

The FMP stable API: GET https://financialmodelingprep.com/stable/<endpoint>?symbol=<TICKER>&apikey=<key>

Environment variable: FMP_API_KEY (required)
"""

import os
from typing import Any

import httpx

BASE_URL = "https://financialmodelingprep.com/stable"


class FMPClient:
    """Async client for FMP REST API."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("FMP_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "FMP API key required. Set FMP_API_KEY env var or pass api_key."
            )

    async def _get(self, endpoint: str, params: dict | None = None) -> Any:
        """Make authenticated GET request to FMP stable API."""
        url = f"{BASE_URL}/{endpoint}"
        request_params = {"apikey": self.api_key}
        if params:
            request_params.update(params)

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, params=request_params)
            response.raise_for_status()
            return response.json()

    # -- Financial Statements --

    async def get_income_statement(
        self, ticker: str, period: str = "annual", limit: int = 5
    ) -> list[dict]:
        """Income statements (revenue, net income, EPS, margins)."""
        return await self._get(
            "income-statement",
            {"symbol": ticker, "period": period, "limit": str(limit)},
        )

    async def get_balance_sheet(
        self, ticker: str, period: str = "annual", limit: int = 5
    ) -> list[dict]:
        """Balance sheet (assets, liabilities, equity)."""
        return await self._get(
            "balance-sheet-statement",
            {"symbol": ticker, "period": period, "limit": str(limit)},
        )

    async def get_cash_flow(
        self, ticker: str, period: str = "annual", limit: int = 5
    ) -> list[dict]:
        """Cash flow statement (operating, investing, financing, FCF)."""
        return await self._get(
            "cash-flow-statement",
            {"symbol": ticker, "period": period, "limit": str(limit)},
        )

    # -- Key Metrics & Ratios --

    async def get_key_metrics_ttm(self, ticker: str) -> list[dict]:
        """Trailing twelve months key metrics (PE, ROE, FCF yield, etc.)."""
        return await self._get("key-metrics-ttm", {"symbol": ticker})

    async def get_ratios_ttm(self, ticker: str) -> list[dict]:
        """Trailing twelve months financial ratios."""
        return await self._get("ratios-ttm", {"symbol": ticker})

    async def get_key_metrics(
        self, ticker: str, period: str = "annual", limit: int = 5
    ) -> list[dict]:
        """Historical key metrics."""
        return await self._get(
            "key-metrics",
            {"symbol": ticker, "period": period, "limit": str(limit)},
        )

    # -- Market Intelligence --

    async def get_analyst_estimates(self, ticker: str, limit: int = 4) -> list[dict]:
        """Analyst consensus estimates (revenue, EPS, growth)."""
        return await self._get(
            "analyst-estimates",
            {"symbol": ticker, "limit": str(limit)},
        )

    async def get_price_target_consensus(self, ticker: str) -> list[dict]:
        """Price target consensus (high, low, average, median)."""
        return await self._get("price-target-consensus", {"symbol": ticker})

    async def get_grades(self, ticker: str, limit: int = 20) -> list[dict]:
        """Analyst grades and rating changes."""
        return await self._get(
            "grades",
            {"symbol": ticker, "limit": str(limit)},
        )

    async def get_insider_trades(self, ticker: str, limit: int = 20) -> list[dict]:
        """Recent insider trades."""
        return await self._get(
            "insider-trading",
            {"symbol": ticker, "limit": str(limit)},
        )

    async def get_insider_trade_statistics(self, ticker: str) -> list[dict]:
        """Insider trading statistics summary."""
        return await self._get(
            "insider-trading-transaction-type",
            {"symbol": ticker},
        )

    # -- Company Profile --

    async def get_profile(self, ticker: str) -> list[dict]:
        """Company profile (name, sector, market cap, description)."""
        return await self._get("profile", {"symbol": ticker})

    async def get_peers(self, ticker: str) -> list[str]:
        """Peer companies for comparison."""
        result = await self._get("stock-peers", {"symbol": ticker})
        if result and isinstance(result, list) and "peersList" in result[0]:
            return result[0]["peersList"]
        return []

    # -- Earnings Transcripts --

    async def get_transcript_dates(self, ticker: str) -> list[dict]:
        """Available earnings transcript dates."""
        return await self._get("earning-call-transcript-dates", {"symbol": ticker})

    async def get_transcript(self, ticker: str, year: int, quarter: int) -> list[dict]:
        """Single earnings call transcript."""
        return await self._get(
            "earning-call-transcript",
            {"symbol": ticker, "year": str(year), "quarter": str(quarter)},
        )

    # -- Valuation --

    async def get_dcf(self, ticker: str) -> list[dict]:
        """Discounted cash flow valuation."""
        return await self._get("discounted-cash-flow", {"symbol": ticker})

    async def get_enterprise_values(
        self, ticker: str, limit: int = 5
    ) -> list[dict]:
        """Enterprise value data for comparables."""
        return await self._get(
            "enterprise-values",
            {"symbol": ticker, "limit": str(limit)},
        )

    async def get_quote(self, ticker: str) -> list[dict]:
        """Current stock quote."""
        return await self._get("quote", {"symbol": ticker})

    # -- Financial Health & Growth (new stable endpoints) --

    async def get_financial_scores(self, ticker: str) -> list[dict]:
        """Altman Z-Score, Piotroski Score — quantitative health check."""
        return await self._get("financial-scores", {"symbol": ticker})

    async def get_financial_growth(
        self, ticker: str, period: str = "annual", limit: int = 5
    ) -> list[dict]:
        """Pre-computed growth rates (revenue, EPS, FCF, etc.)."""
        return await self._get(
            "financial-growth",
            {"symbol": ticker, "period": period, "limit": str(limit)},
        )

    async def get_owner_earnings(self, ticker: str) -> list[dict]:
        """Buffett's owner earnings — cash available to shareholders."""
        return await self._get("owner-earnings", {"symbol": ticker})

    async def get_income_statement_growth(
        self, ticker: str, period: str = "annual", limit: int = 5
    ) -> list[dict]:
        """YoY growth in income statement items."""
        return await self._get(
            "income-statement-growth",
            {"symbol": ticker, "period": period, "limit": str(limit)},
        )

    # -- Revenue Segmentation --

    async def get_revenue_product_segmentation(
        self, ticker: str, period: str = "annual"
    ) -> list[dict]:
        """Revenue breakdown by product/segment."""
        return await self._get(
            "revenue-product-segmentation",
            {"symbol": ticker, "period": period},
        )

    async def get_revenue_geographic_segmentation(
        self, ticker: str, period: str = "annual"
    ) -> list[dict]:
        """Revenue breakdown by geography."""
        return await self._get(
            "revenue-geographic-segmentation",
            {"symbol": ticker, "period": period},
        )

    # -- Ratings --

    async def get_ratings_snapshot(self, ticker: str) -> list[dict]:
        """Composite rating snapshot (buy/sell/hold + score)."""
        return await self._get("ratings-snapshot", {"symbol": ticker})
