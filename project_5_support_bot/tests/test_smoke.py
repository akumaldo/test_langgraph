"""
Smoke tests for Project 5 — Customer Support Bot.

These tests verify the graph BUILDS correctly and routing works
WITHOUT needing the LLM. We test the structural parts, not the
LLM-dependent parts.
"""

from __future__ import annotations

import unittest

from langgraph.checkpoint.memory import MemorySaver

from langgraph_portfolio.projects.project_5_support_bot.langgraph_support_bot import (
    SupportState,
    build_support_bot,
    route_to_department,
)
from langgraph_portfolio.projects.project_5_support_bot.subgraphs.billing import (
    build_billing_subgraph,
)
from langgraph_portfolio.projects.project_5_support_bot.subgraphs.returns import (
    build_returns_subgraph,
)
from langgraph_portfolio.projects.project_5_support_bot.subgraphs.technical import (
    build_technical_subgraph,
)


class SubgraphBuildTest(unittest.TestCase):
    """Test that each sub-graph compiles without errors."""

    def test_billing_subgraph_compiles(self) -> None:
        graph = build_billing_subgraph()
        self.assertIsNotNone(graph)

    def test_technical_subgraph_compiles(self) -> None:
        graph = build_technical_subgraph()
        self.assertIsNotNone(graph)

    def test_returns_subgraph_compiles(self) -> None:
        graph = build_returns_subgraph()
        self.assertIsNotNone(graph)


class ParentGraphBuildTest(unittest.TestCase):
    """Test that the parent graph compiles with sub-graphs wired in."""

    def test_parent_graph_compiles(self) -> None:
        checkpointer = MemorySaver()
        app = build_support_bot(checkpointer=checkpointer)
        self.assertIsNotNone(app)


class RoutingTest(unittest.TestCase):
    """Test the routing logic with hardcoded categories (no LLM)."""

    def test_routes_to_billing(self) -> None:
        state: SupportState = {
            "messages": [],
            "customer_issue": "",
            "category": "billing",
            "department_response": "",
            "satisfaction_rating": 0,
        }
        self.assertEqual(route_to_department(state), "billing_subgraph")

    def test_routes_to_technical(self) -> None:
        state: SupportState = {
            "messages": [],
            "customer_issue": "",
            "category": "technical",
            "department_response": "",
            "satisfaction_rating": 0,
        }
        self.assertEqual(route_to_department(state), "technical_subgraph")

    def test_routes_to_returns(self) -> None:
        state: SupportState = {
            "messages": [],
            "customer_issue": "",
            "category": "returns",
            "department_response": "",
            "satisfaction_rating": 0,
        }
        self.assertEqual(route_to_department(state), "returns_subgraph")

    def test_routes_to_fallback(self) -> None:
        state: SupportState = {
            "messages": [],
            "customer_issue": "",
            "category": "unknown",
            "department_response": "",
            "satisfaction_rating": 0,
        }
        self.assertEqual(route_to_department(state), "fallback_response")


if __name__ == "__main__":
    unittest.main()
