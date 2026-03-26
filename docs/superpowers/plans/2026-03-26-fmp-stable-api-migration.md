# FMP Stable API Migration + High-Value Endpoints

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate the FMP client from legacy v3/v4 API paths to the `/stable/` API and add high-value endpoints (financial scores, growth metrics, revenue segmentation, news, senate trades).

**Architecture:** The `/stable/` API uses a flat URL pattern (`/stable/<endpoint>?symbol=TICKER`) instead of the legacy pattern (`/api/v3/<endpoint>/TICKER`). We change `BASE_URL` and `_get()` so all existing methods use query params. Then add 7 new methods. Update the gathering phase to fetch the new data and pass it to CrewAI agents.

**Tech Stack:** httpx, Python asyncio, unittest (mocked HTTP)

---

## File Structure

```
src/langgraph_portfolio/projects/project_12_analyst/
├── tools/
│   └── fmp.py              # MODIFY — migrate _get + all endpoints to /stable/, add 7 new methods
├── phases/
│   └── gathering.py         # MODIFY — fetch new endpoints in _fetch_fmp_data, inject into crew tasks
tests/
├── test_p12_fmp_tools.py    # MODIFY — update URL assertions for /stable/, add tests for new methods
```

---

### Task 1: Migrate FMP client core + existing endpoints to `/stable/` API

**Files:**
- Modify: `src/langgraph_portfolio/projects/project_12_analyst/tools/fmp.py`
- Modify: `tests/test_p12_fmp_tools.py`

- [ ] **Step 1: Update existing tests to assert `/stable/` URL pattern**

Replace the test file `tests/test_p12_fmp_tools.py` with updated URL assertions. The key changes:
- `call_url` now checks for `/stable/income-statement` (not `/v3/income-statement/AAPL`)
- The ticker is now in `params` (query param `symbol=AAPL`), not in the URL path
- `call_params` checks verify `symbol` param is set

Edit `tests/test_p12_fmp_tools.py` — replace `test_get_income_statement`:
```python
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
        # Stable API: endpoint is in the URL path, ticker in query params
        self.assertIn("/stable/income-statement", call_url)
        call_params = mock_client.get.call_args[1].get("params", {})
        self.assertEqual(call_params["symbol"], "AAPL")
```

Edit `tests/test_p12_fmp_tools.py` — replace `test_get_key_metrics_ttm`:
```python
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
        call_url = mock_client.get.call_args[0][0]
        self.assertIn("/stable/key-metrics-ttm", call_url)
```

- [ ] **Step 2: Run tests to verify they fail (still using v3 paths)**

Run: `poetry run pytest tests/test_p12_fmp_tools.py -v`
Expected: FAIL — URL assertions don't match `/stable/` pattern yet

- [ ] **Step 3: Migrate `_get()` and `BASE_URL` to `/stable/` pattern**

Edit `src/langgraph_portfolio/projects/project_12_analyst/tools/fmp.py`:

Change the module docstring's URL pattern line:
```python
# old:
# The FMP API uses a simple pattern: GET https://financialmodelingprep.com/api/v3/<endpoint>/<TICKER>?apikey=<key>
# new:
# The FMP stable API: GET https://financialmodelingprep.com/stable/<endpoint>?symbol=<TICKER>&apikey=<key>
```

Change `BASE_URL`:
```python
# old:
BASE_URL = "https://financialmodelingprep.com/api"
# new:
BASE_URL = "https://financialmodelingprep.com/stable"
```

Change `_get()` to pass params as keyword arg (so tests can inspect `call_args[1]["params"]`):
```python
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
```

- [ ] **Step 4: Migrate all existing endpoint methods to `/stable/` paths**

Each method changes from `v3/<endpoint>/<ticker>` to just `<endpoint>` with `symbol` in params.

Financial Statements:
```python
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
```

Key Metrics & Ratios:
```python
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
```

Market Intelligence:
```python
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
```

Company Profile:
```python
    async def get_profile(self, ticker: str) -> list[dict]:
        """Company profile (name, sector, market cap, description)."""
        return await self._get("profile", {"symbol": ticker})

    async def get_peers(self, ticker: str) -> list[str]:
        """Peer companies for comparison."""
        result = await self._get("stock-peers", {"symbol": ticker})
        if result and isinstance(result, list) and "peersList" in result[0]:
            return result[0]["peersList"]
        return []
```

Earnings Transcripts:
```python
    async def get_transcript_dates(self, ticker: str) -> list[dict]:
        """Available earnings transcript dates."""
        return await self._get("earning-call-transcript-dates", {"symbol": ticker})

    async def get_transcript(self, ticker: str, year: int, quarter: int) -> list[dict]:
        """Single earnings call transcript."""
        return await self._get(
            "earning-call-transcript",
            {"symbol": ticker, "year": str(year), "quarter": str(quarter)},
        )
```

Valuation:
```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `poetry run pytest tests/test_p12_fmp_tools.py -v`
Expected: 3 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/langgraph_portfolio/projects/project_12_analyst/tools/fmp.py \
        tests/test_p12_fmp_tools.py
git commit -m "refactor(p12): migrate FMP client from legacy v3/v4 to /stable/ API"
```

---

### Task 2: Add high-value endpoints to FMP client

**Files:**
- Modify: `src/langgraph_portfolio/projects/project_12_analyst/tools/fmp.py`
- Modify: `tests/test_p12_fmp_tools.py`

- [ ] **Step 1: Write failing tests for new endpoints**

Append to `tests/test_p12_fmp_tools.py`:
```python
    @patch("langgraph_portfolio.projects.project_12_analyst.tools.fmp.httpx.AsyncClient")
    def test_get_financial_scores(self, mock_client_cls) -> None:
        from langgraph_portfolio.projects.project_12_analyst.tools.fmp import FMPClient

        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"altmanZScore": 5.2, "piotroskiScore": 7}
        ]
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        client = FMPClient(api_key="test-key")
        result = run(client.get_financial_scores("AAPL"))

        self.assertEqual(result[0]["altmanZScore"], 5.2)
        call_url = mock_client.get.call_args[0][0]
        self.assertIn("/stable/financial-scores", call_url)

    @patch("langgraph_portfolio.projects.project_12_analyst.tools.fmp.httpx.AsyncClient")
    def test_get_revenue_segmentation(self, mock_client_cls) -> None:
        from langgraph_portfolio.projects.project_12_analyst.tools.fmp import FMPClient

        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"iPhone": 200000000000, "Services": 85000000000}
        ]
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        client = FMPClient(api_key="test-key")
        result = run(client.get_revenue_product_segmentation("AAPL"))

        self.assertIsInstance(result, list)
        call_url = mock_client.get.call_args[0][0]
        self.assertIn("/stable/revenue-product-segmentation", call_url)
```

- [ ] **Step 2: Run tests to verify new ones fail**

Run: `poetry run pytest tests/test_p12_fmp_tools.py -v`
Expected: 2 new tests FAIL (methods don't exist), 3 old tests PASS

- [ ] **Step 3: Implement 7 new endpoint methods**

Append to `FMPClient` in `src/langgraph_portfolio/projects/project_12_analyst/tools/fmp.py`:
```python
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

    # -- Ratings & News --

    async def get_ratings_snapshot(self, ticker: str) -> list[dict]:
        """Composite rating snapshot (buy/sell/hold + score)."""
        return await self._get("ratings-snapshot", {"symbol": ticker})
```

- [ ] **Step 4: Run tests to verify all pass**

Run: `poetry run pytest tests/test_p12_fmp_tools.py -v`
Expected: 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/langgraph_portfolio/projects/project_12_analyst/tools/fmp.py \
        tests/test_p12_fmp_tools.py
git commit -m "feat(p12): add high-value FMP endpoints (scores, growth, segmentation, ratings)"
```

---

### Task 3: Wire new endpoints into gathering phase

**Files:**
- Modify: `src/langgraph_portfolio/projects/project_12_analyst/phases/gathering.py`

- [ ] **Step 1: Add new FMP calls to `_fetch_fmp_data`**

In `gathering.py`, update the `_fetch_fmp_data` function to fetch the new endpoints alongside existing ones. Add them to the `asyncio.gather` call:

After the existing 12 gather calls, add 4 more:
```python
    (
        profile,
        income,
        balance,
        cashflow,
        metrics_ttm,
        ratios_ttm,
        estimates,
        price_target,
        grades,
        insider_trades,
        insider_stats,
        transcript_dates,
        financial_scores,
        financial_growth,
        revenue_segments,
        geo_segments,
    ) = await asyncio.gather(
        client.get_profile(ticker),
        client.get_income_statement(ticker),
        client.get_balance_sheet(ticker),
        client.get_cash_flow(ticker),
        client.get_key_metrics_ttm(ticker),
        client.get_ratios_ttm(ticker),
        client.get_analyst_estimates(ticker),
        client.get_price_target_consensus(ticker),
        client.get_grades(ticker),
        client.get_insider_trades(ticker),
        client.get_insider_trade_statistics(ticker),
        client.get_transcript_dates(ticker),
        client.get_financial_scores(ticker),
        client.get_financial_growth(ticker),
        client.get_revenue_product_segmentation(ticker),
        client.get_revenue_geographic_segmentation(ticker),
    )
```

Add the new data to the return dict:
```python
    return {
        "profile": profile,
        "income": income,
        "balance": balance,
        "cashflow": cashflow,
        "metrics_ttm": metrics_ttm,
        "ratios_ttm": ratios_ttm,
        "estimates": estimates,
        "price_target": price_target,
        "grades": grades,
        "insider_trades": insider_trades,
        "insider_stats": insider_stats,
        "transcripts": transcripts,
        "financial_scores": financial_scores,
        "financial_growth": financial_growth,
        "revenue_segments": revenue_segments,
        "geo_segments": geo_segments,
    }
```

- [ ] **Step 2: Inject new data into CrewAI task contexts**

In `gather_data()`, add the new data to the financial task description (task 0):
```python
    crew.tasks[0].description += (
        f"\n\n--- FINANCIAL DATA ---\n"
        f"Income Statement:\n{_truncate_json(raw_data['income'])}\n\n"
        f"Balance Sheet:\n{_truncate_json(raw_data['balance'])}\n\n"
        f"Cash Flow:\n{_truncate_json(raw_data['cashflow'])}\n\n"
        f"Key Metrics TTM:\n{_truncate_json(raw_data['metrics_ttm'])}\n\n"
        f"Ratios TTM:\n{_truncate_json(raw_data['ratios_ttm'])}\n\n"
        f"Financial Scores (Altman Z, Piotroski):\n{_truncate_json(raw_data['financial_scores'])}\n\n"
        f"Financial Growth:\n{_truncate_json(raw_data['financial_growth'])}\n\n"
        f"Revenue by Product:\n{_truncate_json(raw_data['revenue_segments'])}\n\n"
        f"Revenue by Geography:\n{_truncate_json(raw_data['geo_segments'])}"
    )
```

- [ ] **Step 3: Run all P12 tests to verify no regressions**

Run: `poetry run pytest tests/test_project_12_analyst.py tests/test_p12_fmp_tools.py tests/test_p12_phases.py -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/langgraph_portfolio/projects/project_12_analyst/phases/gathering.py
git commit -m "feat(p12): wire new FMP endpoints into gathering phase"
```

---

### Task 4: Run full test suite + final verification

- [ ] **Step 1: Run the full project test suite**

Run: `poetry run pytest -v`
Expected: All tests PASS, no regressions

- [ ] **Step 2: Commit (if any fixups needed)**

Only if fixes were required.

---

## Summary

| Task | What | Files |
|------|------|-------|
| 1 | Migrate core + 18 endpoints to `/stable/` | `fmp.py`, `test_p12_fmp_tools.py` |
| 2 | Add 7 high-value endpoints | `fmp.py`, `test_p12_fmp_tools.py` |
| 3 | Wire new data into gathering phase | `gathering.py` |
| 4 | Full regression test | — |
