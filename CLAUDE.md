# LangGraph Portfolio — Agentic AI Learning Project

## What this is

A progressive learning portfolio with 10 projects that teach agentic AI frameworks (LangGraph, CrewAI, AG2, BeeAI, LlamaIndex). Each project builds on concepts from the previous one. See `docs/portfolio_expansion_plan.md` for the full roadmap.

## Project Structure

```
project_1_chatbot/          → entry point (thin wrapper)
project_2_research_agent/   → entry point (thin wrapper)
project_3_writing_team_*/   → entry points (LangGraph + CrewAI versions)
project_4_data_analyst/     → entry point
project_5_support_bot/      → entry point
src/langgraph_portfolio/    → actual implementation code
  core/                     → shared models, graph engine, knowledge base, fixtures
  projects/                 → one subpackage per project
```

- Each `project_N_*/main.py` is a thin entry point that imports from `src/langgraph_portfolio/projects/...`
- The `core/` module provides shared infrastructure: `BaseWorkflowState`, `GraphBuilder`, `Document`, `KnowledgeBase`, etc.
- P1-P3 have both a **scaffold** version (custom GraphBuilder) and a **real LangGraph** version. P4+ are real LangGraph only.

## Progress

- **Project 1 (Chatbot)**: DONE. Has both scaffold (`graph.py`) and real LangGraph (`langgraph_chatbot.py`) versions. Concepts: state, nodes, conditional edges, routing.
- **Project 2 (Research Agent)**: DONE. Has both scaffold and real LangGraph (`langgraph_research.py`) versions. New concepts: tool calling (`@tool`), agent loop, `ToolNode`, `bind_tools()`, RAG, citations.
- **Project 3 (Writing Team)**: DONE. Has both scaffold and real LangGraph (`langgraph_writing_team.py`) versions. New concepts: multi-agent coordination (5 roles with different system prompts), pipeline edges, quality-control loop (editor→writer), revision tracking with MAX_REVISIONS safety cap.
- **Project 4 (Data Analyst)**: DONE. Real LangGraph only (`langgraph_data_analyst.py`). New concepts: structured output (`with_structured_output()`), code execution, error-recovery loop, checkpointing (`MemorySaver`).
- **Project 5 (Support Bot)**: DONE. Real LangGraph only (`langgraph_support_bot.py`). New concepts: sub-graphs (graphs inside graphs with input/output schemas), `interrupt()` (human-in-the-loop mid-execution), streaming (`stream_mode="updates"`). Has `subgraphs/` folder with billing, technical, returns departments.
- **Project 6 (Intel Crew)**: DONE. CrewAI only (`flow.py`, `research_crew.py`, `analysis_crew.py`, `report_crew.py`). New concepts: CrewAI Flows (`@start`, `@listen` decorators for multi-crew pipelines), custom tools (`BaseTool` classes), Flow state (Pydantic `BaseModel` with direct mutation vs LangGraph merge). Known constraints: `output_pydantic` disabled (Instructor/Ollama incompatibility), crews return freeform text instead.
- **Project 7 (Debate Arena)**: DONE. AG2 only (`debate.py`, `agents.py`, `models.py`). New concepts: AG2 `ConversableAgent` (conversational participants), `GroupChat` (shared conversation space), `GroupChatManager` (orchestrates turns), custom `speaker_selection_method` (structured debate flow), message-based termination (`is_termination_msg`). Uses Ollama via OpenAI-compatible API (`base_url`).
- **Project 8 (BeeAI Research)**: DONE. BeeAI only (`agent.py`, `tools.py`, `events.py`). New concepts: BeeAI `ReActAgent` (built-in Think→Act→Observe loop), `Tool` base class (explicit Pydantic input schemas vs LangGraph's `@tool` inference), event-driven observability (`emitter.on("update"/"success"/"error")` for fine-grained logging of every ReAct step), `ChatModel.from_name("ollama:...")` (provider-agnostic LLM config), async-native execution (`await agent.run()`). Dep conflict: BeeAI and CrewAI can't coexist in Poetry (json-repair version clash), installed via pip.
- **Project 9 (RAG Pipeline)**: DONE. LlamaIndex Workflows only (`workflow.py` → split into `ingestion.py`, `query.py`, `strategies.py`, `events.py`, `models.py`). New concepts: event-driven architecture (`@step` + typed `Event` classes vs LangGraph's explicit edges), document ingestion pipeline (PDF → chunk → embed → vector index), embedding models (`qwen3-embedding` via Ollama), retrieval strategies (keyword/BM25, semantic, hybrid, LLM-based reranking), workflow composition (IngestionWorkflow runs once, QueryWorkflow runs per question), conditional event routing (rerank step only fires for reranking strategy via `DocumentsReady` convergence event). Quality-control loop (evaluate_answer) designed but disabled due to local LLM timeout constraints — code commented with re-enable instructions. Dependencies installed via pip (llama-index-llms-ollama, llama-index-embeddings-ollama, llama-index-readers-file, llama-index-retrievers-bm25, pymupdf).
- **Project 10**: Not started yet.

## Tech Stack

- Python with Poetry for dependency management
- LangGraph (real framework) + custom scaffold for learning (P1-P3 only)
- LangChain (langchain_core, langchain_ollama)
- BeeAI Framework (beeai-framework, installed via pip due to dep conflict with CrewAI)
- LlamaIndex Workflows (llama-index-core, llama-index-workflows, plus Ollama/BM25/file-reader sub-packages via pip)
- Ollama for local LLM (model: qwen3.5:2b for P9, qwen3.5:35b for others; embedding: qwen3-embedding)
- Pydantic for structured output models

## How we work

- This is a **learning project** — Bruno is new to LangGraph/agentic frameworks
- We build step by step, explaining every concept before coding
- P1-P3: scaffold version first, then real LangGraph version alongside it
- P4+: real LangGraph only (no scaffold), tutor-style comments in the code

## Conventions

- Real LangGraph implementations use `TypedDict` for state (LangGraph's native approach)
- Scaffold versions (P1-P3) use Pydantic `BaseWorkflowState` from core
- Node functions are plain functions that take state and return update dicts
- Each file has section comments explaining concepts (tutor-style)
- Sub-graphs (P5+) use input/output schemas for state isolation
