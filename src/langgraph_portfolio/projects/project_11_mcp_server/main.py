"""
PROJECT 11 — MCP SERVER: MAIN MODULE
=====================================

Entry point for the MCP server. Can be run directly or imported.

Usage:
  # Start the MCP server (stdio transport)
  poetry run python -m langgraph_portfolio.projects.project_11_mcp_server.main

  # Re-import jobs from JSON before starting
  poetry run python -m langgraph_portfolio.projects.project_11_mcp_server.main --sync

  # Register with Claude Code (one-time setup)
  claude mcp add job-search -- poetry run python -m langgraph_portfolio.projects.project_11_mcp_server.main

CONCEPT — stdio transport
  When Claude Code runs this server, it spawns it as a subprocess and
  communicates via stdin/stdout using JSON-RPC messages. No HTTP server,
  no ports, no networking. The FastMCP.run() method handles all of this —
  it reads JSON-RPC from stdin and writes responses to stdout.

  This is fundamentally different from a REST API:
  - REST: client sends HTTP request → server returns HTTP response
  - MCP:  client spawns server process → exchanges JSON-RPC over stdio
"""

from __future__ import annotations

from pathlib import Path

from . import db
from .server import create_server


def main() -> None:
    """Initialize the database and start the MCP server."""
    import argparse

    parser = argparse.ArgumentParser(description="Job Search MCP Server")
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Re-import jobs from job-history.json before starting",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=db.DEFAULT_DB_PATH,
        help="Path to the SQLite database file",
    )
    parser.add_argument(
        "--json-path",
        type=Path,
        default=db.DEFAULT_JSON_PATH,
        help="Path to job-history.json",
    )
    args = parser.parse_args()

    # If --sync, delete the existing DB to force a fresh import
    if args.sync and args.db_path.exists():
        args.db_path.unlink()
        print("[sync] Deleted existing database, will re-import from JSON")

    # Initialize DB + create server
    mcp_server = create_server(db_path=args.db_path, json_path=args.json_path)

    # Run the MCP server (blocks, reads from stdin, writes to stdout)
    mcp_server.run()
