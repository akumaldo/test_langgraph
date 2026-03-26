# Architecture

This repository is designed as a shared-core portfolio scaffold for building and comparing several agentic applications.

## Stack

- LangGraph for orchestration and branching workflows
- Pydantic for typed state and validation
- LlamaIndex for document ingestion and retrieval
- CrewAI for the multi-agent comparison project

## Layout

```text
langgraph-portfolio/
├── src/langgraph_portfolio/core/
├── src/langgraph_portfolio/projects/
├── project_1_chatbot/
├── project_2_research_agent/
├── project_3_writing_team_langgraph/
├── project_3_writing_team_crewai/
├── project_4_data_analyst/
├── project_5_capstone/
├── docs/
└── scripts/
```

## Design Goals

- Keep shared orchestration logic in one place
- Make each project runnable on its own
- Keep the framework choices visible and easy to compare
- Provide scaffold-first code that can be extended incrementally

