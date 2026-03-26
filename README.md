# LangGraph Portfolio Scaffold

This repository is a greenfield portfolio scaffold for an eight-week learning path:

1. Multi-intent chatbot with LangGraph + Pydantic
2. Document research agent with LangGraph + Pydantic + LlamaIndex
3. Writing team implemented twice for a LangGraph vs CrewAI comparison
4. Autonomous data analyst with LangGraph + Pydantic
5. Capstone personal AI assistant that composes the earlier patterns selectively

The code is organized as a shared-core monorepo:

- `src/langgraph_portfolio/core` holds reusable orchestration, state, and retrieval utilities
- `src/langgraph_portfolio/projects` contains the framework-facing project implementations
- `project_*` folders provide runnable wrappers, local docs, and per-project tests

## Getting started

The repository is configured for Poetry, but the scaffold is also runnable from the repo root because `sitecustomize.py` adds `src/` to `sys.path`.

```bash
python -m unittest discover -s tests
python project_1_chatbot/main.py
python scripts/run_project.py
```

If you have Poetry installed:

```bash
poetry install
poetry run project-1-chatbot
```

## Docs

- [Architecture](docs/architecture.md)
- [Framework comparison](docs/comparisons.md)

