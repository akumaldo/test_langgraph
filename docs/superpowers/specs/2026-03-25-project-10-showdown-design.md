# Project 10: Framework Showdown — Design Spec

## Overview

One research assistant problem solved with all 5 frameworks (LangGraph, CrewAI, AG2, BeeAI, LlamaIndex Workflows) side by side. Same input, same output format, different orchestration. The capstone "when to use what" study.

**Research question**: "What are the main uses for AI orchestration in industry today?"

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| LLM model | `qwen3.5:2b` for all 5 | Fair comparison, avoids timeout issues |
| Info source | Pre-loaded knowledge base | Controlled size, reproducible, no network dependency |
| Output format | Pydantic where supported, plain text otherwise | Tests each framework's native capability |
| Test scope | 1 research question | Plan called for 5; reduced to 1 for speed. Can extend later |
| Runner | Semi-automated (independent runs, shared comparison) | Sidesteps BeeAI/CrewAI dep conflict |
| Architecture | Shared tools, separate orchestration | Isolates the framework as the only variable |

## Project Structure

```
src/langgraph_portfolio/projects/project_10_showdown/
  problem.py              — shared: KB, question, output schema, search function, save/load
  langgraph_version.py    — LangGraph implementation
  crewai_version.py       — CrewAI implementation
  ag2_version.py          — AG2 implementation
  beeai_version.py        — BeeAI implementation
  llamaindex_version.py   — LlamaIndex Workflows implementation
  compare.py              — loads saved outputs, generates comparison report
  results/                — each framework saves its output here as JSON

project_10_showdown/
  main.py                 — thin entry point (run one or all, or --compare)
```

## Shared Components (`problem.py`)

### Knowledge Base

4 documents (~500-800 words each), hardcoded as a list of dicts:

```python
KNOWLEDGE_BASE = [
    {
        "id": "doc1",
        "title": "AI Orchestration in Financial Services",
        "source": "Industry Report 2024",
        "content": "..."  # fraud detection, trading, compliance
    },
    {
        "id": "doc2",
        "title": "AI Orchestration in Healthcare",
        "source": "Tech Review 2024",
        "content": "..."  # diagnostics, drug discovery, patient flow
    },
    {
        "id": "doc3",
        "title": "AI Orchestration in Manufacturing",
        "source": "Industry Analysis 2024",
        "content": "..."  # predictive maintenance, supply chain, QA
    },
    {
        "id": "doc4",
        "title": "AI Orchestration in Customer Service",
        "source": "Business Tech Report 2024",
        "content": "..."  # chatbots, ticket routing, sentiment analysis
    },
]
```

### Search Function

`search_knowledge_base(query: str) -> list[dict]` — simple keyword matching over document content. No embeddings, no vector store. Returns all documents where the content contains at least one query term (case-insensitive split on whitespace). Returns all documents if no terms match, to ensure the agent always has context to work with. All frameworks use this same function, wrapped in their own tool format.

### Output Schema

```python
class ResearchReport(BaseModel):
    summary: str          # 2-3 sentence overview
    findings: list[str]   # key findings (bullet points)
    citations: list[str]  # which documents were used (by title/source)
    framework: str        # which framework produced this
```

### Result Persistence

- `save_result(framework: str, output: dict, elapsed: float)` — saves to `results/{framework}.json`. Includes `structured_output_success: bool` (whether Pydantic parsing succeeded) and the raw text fallback if it didn't.
- `load_results() -> dict` — loads all saved results for comparison

### JSON Parsing Fallback

For frameworks that rely on prompt-based JSON (BeeAI, LlamaIndex, AG2, CrewAI): if JSON parsing fails, save the raw LLM text and mark `structured_output_success: false`. No retries — the failure itself is a comparison data point.

### Error Handling

Each framework run is wrapped in a try/except with a 120s timeout. Failures are saved as error results (framework name, error message, elapsed time) so `compare.py` can report which frameworks succeeded and which didn't.

## Framework Implementations

All use `qwen3.5:2b` via Ollama. Each follows: **receive question → search KB → analyze results → produce report**.

### LangGraph (`langgraph_version.py`)

- Graph with nodes: `research` (calls search tool), `analyze` (synthesizes), `report` (structured output)
- `@tool` wrapping `search_knowledge_base`
- `ToolNode` + `bind_tools()` for agent loop
- `with_structured_output(ResearchReport)` on report node
- **Structured output**: native Pydantic

### CrewAI (`crewai_version.py`)

- Single crew: researcher agent + analyst agent, sequential process
- Tool called from Python (NOT via agent) — Ollama/Instructor incompatibility
- Search results passed as text context to the crew
- **Structured output**: plain text (output_pydantic broken with Ollama), parsed afterward

### AG2 (`ag2_version.py`)

- 2 agents in GroupChat: researcher + analyst
- Researcher gets KB search results injected in initial message
- Analyst produces final report as a message
- Custom speaker selection: researcher → analyst
- **Structured output**: plain text, parse final message

### BeeAI (`beeai_version.py`)

- Single `ReActAgent` with search tool (Tool subclass, Pydantic input schema)
- Agent autonomously decides to search, then synthesizes
- System prompt asks for JSON output matching ResearchReport
- **Structured output**: prompt-based JSON, parsed afterward

### LlamaIndex Workflows (`llamaindex_version.py`)

- Workflow with steps: `search` → `analyze` → `report`
- Event-driven: `SearchDone` → `AnalysisDone` → `StopEvent`
- No vector store — uses shared keyword search function
- `Ollama.acomplete()` for synthesis
- **Structured output**: prompt-based JSON, parsed afterward

## Comparison Report (`compare.py`)

Loads all results from `results/` and generates `comparison_report.md`.

### Automated Metrics (computed at comparison time)

- **Execution time** — from saved results JSON (elapsed seconds)
- **Lines of code** — counted by reading each `*_version.py` source file directly (non-blank, non-comment lines)
- **Output completeness** — checks saved output for all fields (summary, findings, citations)
- **Structured output success** — from saved `structured_output_success` flag: `true` = Pydantic parsed, `false` = raw text fallback

### Qualitative Notes (filled in manually after running all frameworks)

The generated `comparison_report.md` includes template sections with placeholder text for you to fill in based on your experience building and running each version:

- **Control level** — how much you designed the flow vs framework decided
- **Debuggability** — how easy to trace what happened
- **Boilerplate** — setup code vs actual logic ratio
- **Constraints encountered** — what didn't work

## New Concepts

- **Cross-framework comparison** — same problem, different orchestration, measurable results
- **Automated benchmarking** — time, LOC, output completeness measured programmatically
- **Tool adapter pattern** — one search function wrapped in 5 framework-specific tool formats

## Running

```bash
# Run each framework independently
poetry run python project_10_showdown/main.py --framework langgraph
poetry run python project_10_showdown/main.py --framework crewai
poetry run python project_10_showdown/main.py --framework ag2
poetry run python project_10_showdown/main.py --framework beeai
poetry run python project_10_showdown/main.py --framework llamaindex

# Generate comparison report
poetry run python project_10_showdown/main.py --compare
```

### Dependency Handling

- LangGraph, CrewAI, AG2: work in base Poetry venv
- BeeAI: needs `poetry run pip install beeai-framework` first (dep conflict with CrewAI)
- LlamaIndex: pip packages already present from P9

### Execution Order

1. Run LangGraph, CrewAI, AG2 (base venv)
2. Install BeeAI deps, run BeeAI
3. Run LlamaIndex (deps present from P9)
4. Run `--compare` to generate report

## Known Limitations (to document in final report)

- **Model**: Using `qwen3.5:2b` for all — production would use larger models or API-based LLMs. Chosen for fair comparison and to avoid timeout issues experienced in earlier projects.
- **CrewAI tool use**: Broken with Ollama due to Instructor incompatibility. Tools called from Python instead of by agents. Would work fine with OpenAI API.
- **CrewAI structured output**: `output_pydantic` disabled for same reason. Returns plain text.
- **BeeAI/CrewAI dep conflict**: Can't coexist in same Poetry venv (json-repair version clash). Must install separately via pip.
- **Knowledge base**: Static keyword search, not vector/semantic retrieval. Chosen for simplicity and equal footing across frameworks.
- **Single question**: One research question for speed. More questions would give stronger comparison signal.
- **Structured output**: Only native in LangGraph (`with_structured_output`). Others rely on prompt-based JSON — a real framework difference worth noting.
- **Local LLM context limits**: Prompts kept under ~15k tokens to avoid empty responses from Ollama.
