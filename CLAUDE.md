# LangGraph Portfolio — Agentic AI Learning Project

## What this is

A progressive learning portfolio with 12 projects that teach agentic AI frameworks (LangGraph, CrewAI, AG2, BeeAI, LlamaIndex). Each project builds on concepts from the previous one. See `docs/portfolio_expansion_plan.md` for the full roadmap.

## Commands

```bash
# Install
poetry install                        # core deps (LangGraph, CrewAI, AG2)
pip install beeai-framework           # BeeAI (dep conflict with CrewAI in Poetry)
pip install llama-index-llms-ollama llama-index-embeddings-ollama llama-index-readers-file llama-index-retrievers-bm25 pymupdf  # LlamaIndex extras
pip install mcp                       # MCP SDK

# Ollama models (must be running: ollama serve)
ollama pull qwen3.5:35b              # main LLM (P1-P8, P10-P12)
ollama pull qwen3.5:2b               # lighter LLM (P9 only)
ollama pull qwen3-embedding           # embeddings (P9)

# Run projects
poetry run project-1-chatbot
poetry run project-2-research-agent
poetry run project-3-writing-team-langgraph
poetry run project-3-writing-team-crewai
poetry run project-4-data-analyst
poetry run project-5-capstone          # support bot
poetry run project-7-debate-arena
poetry run project-8-beeai-research
poetry run project-9-rag-pipeline
poetry run project-11-mcp-server
poetry run project-12-analyst

# Tests
poetry run pytest
```

## Project Structure

```
src/langgraph_portfolio/    → all implementation code
  core/                     → shared models, graph engine, knowledge base, fixtures
  projects/                 → one subpackage per project (each has main.py)
tests/                      → all tests (shared + per-project smoke tests)
docs/                       → planning & architecture docs
scripts/                    → utility scripts
```

- All code lives in `src/langgraph_portfolio/projects/`. Entry points are `poetry run` scripts defined in `pyproject.toml`.
- The `core/` module provides shared infrastructure: `BaseWorkflowState`, `GraphBuilder`, `Document`, `KnowledgeBase`, etc.
- P1-P3 have both a **scaffold** version (custom GraphBuilder) and a **real LangGraph** version. P4+ are real LangGraph only.

## Progress

| # | Project | Framework | Status | Key Files | Core Concepts |
|---|---------|-----------|--------|-----------|---------------|
| 1 | Chatbot | LangGraph + scaffold | DONE | `graph.py`, `langgraph_chatbot.py` | state, nodes, conditional edges, routing |
| 2 | Research Agent | LangGraph + scaffold | DONE | `langgraph_research.py` | `@tool`, agent loop, `ToolNode`, RAG, citations |
| 3 | Writing Team | LangGraph + CrewAI | DONE | `langgraph_writing_team.py` | multi-agent, pipeline edges, editor→writer loop |
| 4 | Data Analyst | LangGraph | DONE | `langgraph_data_analyst.py` | `with_structured_output()`, code execution, checkpointing |
| 5 | Support Bot | LangGraph | DONE | `langgraph_support_bot.py` | sub-graphs, `interrupt()`, streaming |
| 6 | Intel Crew | CrewAI | DONE | `flow.py`, `*_crew.py` | Flows (`@start`/`@listen`), custom tools |
| 7 | Debate Arena | AG2 | DONE | `debate.py`, `agents.py` | `ConversableAgent`, `GroupChat`, turn management |
| 8 | BeeAI Research | BeeAI | DONE | `agent.py`, `tools.py` | `ReActAgent`, event-driven observability |
| 9 | RAG Pipeline | LlamaIndex | DONE | `ingestion.py`, `query.py`, `strategies.py` | `@step` + Events, embeddings, retrieval strategies |
| 10 | Framework Showdown | All 5 | DONE | per-framework dirs | cross-framework comparison |
| 11 | Job Search MCP | MCP | DONE | `server.py`, `db.py` | `FastMCP`, resources, tools, prompts, validation |
| 12 | Investment Committee | All 5 | DONE | `orchestrator.py`, `phases/`, `server.py` | cross-framework composition, MCP server, co-pilot HITL |

## Tech Stack

- Python with Poetry for dependency management
- LangGraph (real framework) + custom scaffold for learning (P1-P3 only)
- LangChain (langchain_core, langchain_ollama)
- BeeAI Framework (beeai-framework, installed via pip due to dep conflict with CrewAI)
- LlamaIndex Workflows (llama-index-core, llama-index-workflows, plus Ollama/BM25/file-reader sub-packages via pip)
- MCP SDK (mcp, installed via pip — FastMCP for stdio server)
- Ollama for local LLM (model: qwen3.5:2b for P9, qwen3.5:35b for others; embedding: qwen3-embedding)
- httpx for FMP REST API calls (P12)
- Pydantic for structured output models

## How we work

- This is a **learning project** — Bruno is new to LangGraph/agentic frameworks
- We build step by step, explaining every concept before coding
- P1-P3: scaffold version first, then real LangGraph version alongside it
- P4+: real LangGraph only (no scaffold), tutor-style comments in the code

## Gotchas

- BeeAI and CrewAI can't coexist in Poetry (json-repair version clash) — BeeAI installed via pip
- LlamaIndex sub-packages also installed via pip, not Poetry
- CrewAI `output_pydantic` doesn't work with Ollama (Instructor incompatibility) — crews return freeform text
- P9 quality-control loop disabled due to local LLM timeout — see code comments to re-enable
- `pyproject.toml` script `project-5-capstone` is the Support Bot (naming mismatch)

## Conventions

- Real LangGraph implementations use `TypedDict` for state (LangGraph's native approach)
- Scaffold versions (P1-P3) use Pydantic `BaseWorkflowState` from core
- Node functions are plain functions that take state and return update dicts
- Each file has section comments explaining concepts (tutor-style)
- Sub-graphs (P5+) use input/output schemas for state isolation
