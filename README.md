# Agentic AI Portfolio

A progressive learning portfolio with 11 projects spanning five agentic AI frameworks and one protocol: **LangGraph**, **CrewAI**, **AG2**, **BeeAI**, **LlamaIndex Workflows**, and **MCP**.

Each project introduces 3-4 new concepts while building on the previous ones.

## Learning Arc

```
FOUNDATION (done)                    INTERMEDIATE (done)              ADVANCED
-----------------                    -------------------              --------
P1  Chatbot (LangGraph)              P5  Support Bot (LangGraph)      P9  RAG Pipeline (LlamaIndex) [done]
P2  Research Agent (LangGraph)       P6  Intel Crew (CrewAI)          P10 Framework Showdown (All)
P3  Writing Team (LangGraph+CrewAI)  P7  Debate Arena (AG2)           P11 Job Search MCP Server (MCP)
P4  Data Analyst (LangGraph)         P8  Research Agent v2 (BeeAI)
```

## Projects

| # | Project | Framework | Key Concepts | Status |
|---|---------|-----------|-------------|--------|
| 1 | Chatbot | LangGraph | State, nodes, conditional edges, routing | Done |
| 2 | Research Agent | LangGraph | Tool calling (`@tool`), agent loop, `ToolNode`, RAG, citations | Done |
| 3 | Writing Team | LangGraph + CrewAI | Multi-agent coordination, pipeline edges, quality-control loop | Done |
| 4 | Data Analyst | LangGraph | Structured output, code execution, error-recovery loop, checkpointing | Done |
| 5 | Support Bot | LangGraph | Sub-graphs, `interrupt()` (human-in-the-loop), streaming | Done |
| 6 | Intel Crew | CrewAI | CrewAI Flows (`@start`/`@listen`), custom tools, Flow state | Done |
| 7 | Debate Arena | AG2 | `ConversableAgent`, `GroupChat`, speaker selection, message termination | Done |
| 8 | BeeAI Research | BeeAI | ReAct agent, `Tool` base class, event-driven observability, async execution | Done |
| 9 | RAG Pipeline | LlamaIndex | Event-driven workflows (`@step`), document ingestion, embedding, retrieval strategies (keyword, semantic, hybrid, reranking) | Done |
| 10 | Framework Showdown | All | Cross-framework comparison on the same task | Not started |
| 11 | Job Search MCP Server | MCP | MCP protocol (stdio/JSON-RPC), resources, tools, prompts, least-privilege boundaries | Not started |

## Project Structure

```
src/langgraph_portfolio/        -> all implementation code
  core/                         -> shared models, graph engine, knowledge base
  projects/                     -> one subpackage per project (each has main.py)
tests/                          -> all tests (shared + per-project smoke tests)
docs/                           -> planning & architecture docs
scripts/                        -> utility scripts
```

P1-P3 have both a **scaffold** version (custom `GraphBuilder`) and a **real LangGraph** version. P4+ are real-framework only.

## Tech Stack

- **Python** with **Poetry** for dependency management
- **LangGraph** / LangChain (P1-P5, P10)
- **CrewAI** (P3, P6)
- **AG2** (P7)
- **BeeAI Framework** (P8 — installed via pip due to dep conflict with CrewAI)
- **LlamaIndex Workflows** (P9 — installed via pip)
- **Ollama** for local LLM (`qwen3.5:35b` general; `qwen3.5:2b` for P9; `qwen3-embedding` for embeddings)
- **Pydantic** for structured output models

## Getting Started

```bash
# Install dependencies
poetry install

# Run a project (example)
poetry run project-1-chatbot
```

Make sure Ollama is running locally with the required models pulled.

## Docs

- [Architecture](docs/architecture.md)
- [Framework Comparison](docs/comparisons.md)
- [Expansion Plan](docs/portfolio_expansion_plan.md)
