# Project 6: Competitive Intelligence Crew — Design Spec

## Overview

A multi-crew intelligence system that researches, analyzes, and reports on competitors for any user-specified company. Three CrewAI crews are chained together using **CrewAI Flows**, each with its own agents, process type, and structured output.

**New concepts taught:**
- CrewAI Flows (`@start`, `@listen`) for multi-crew pipelines
- CrewAI custom tools (`BaseTool`) for real web scraping
- Mixed process types (sequential vs hierarchical) within one project
- CrewAI Memory (short-term, long-term, entity) for cross-run learning
- FlowState for sharing data between crews

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   IntelligenceFlow                       │
│                                                         │
│  @start()          @listen()           @listen()        │
│  ┌──────────────┐  ┌────────────────┐  ┌─────────────┐ │
│  │Research Crew  │─→│ Analysis Crew  │─→│ Report Crew │ │
│  │ (sequential)  │  │ (hierarchical) │  │ (sequential)│ │
│  │               │  │                │  │             │ │
│  │ • Scraper     │  │ • Analyst      │  │ • Writer    │ │
│  │ • Gatherer    │  │ • Comparator   │  │ • Editor    │ │
│  │               │  │ • Strategist   │  │             │ │
│  └──────────────┘  └────────────────┘  └─────────────┘ │
│         │                   │                  │        │
│    ScrapedPage +      CompetitiveAnalysis  IntelligenceReport
│    CompetitorProfile                      (console + .md file)
│                                                         │
│  FlowState carries data between phases                  │
│  memory=True on all crews (short-term + long-term)      │
└─────────────────────────────────────────────────────────┘
```

## File Structure

```
src/langgraph_portfolio/projects/project_6_intel_crew/
  __init__.py          — package init
  flow.py              — IntelligenceFlow class, chains the 3 crews
  research_crew.py     — Research Crew (sequential, 2 agents)
  analysis_crew.py     — Analysis Crew (hierarchical, 3 agents)
  report_crew.py       — Report Crew (sequential, 2 agents)
  tools.py             — WebScraperTool (requests + BeautifulSoup)
  models.py            — all Pydantic models for structured output
  main.py              — CLI entry point

project_6_intel_crew/
  main.py              — thin wrapper entry point (like P1-P5)

reports/               — generated markdown reports (gitignored)
```

## Data Models (`models.py`)

All models use Pydantic `BaseModel` for structured output via `output_pydantic=`.

### ScrapedPage
```python
class ScrapedPage(BaseModel):
    url: str
    title: str
    raw_content: str        # cleaned text from the page
    scraped_at: datetime
```

### CompetitorProfile
```python
class CompetitorProfile(BaseModel):
    name: str
    description: str        # what the company does
    products: list[str]     # key products/services
    pricing_info: str       # pricing details (if found)
    strengths: list[str]
    weaknesses: list[str]
```

### CompetitiveAnalysis
```python
class CompetitiveAnalysis(BaseModel):
    target_company: str
    competitors: list[CompetitorProfile]
    feature_comparison: str    # side-by-side comparison narrative
    market_positioning: str    # market position analysis
    key_insights: list[str]    # strategic insights
```

### CompetitorSummary
```python
class CompetitorSummary(BaseModel):
    name: str
    overview: str              # brief description
    key_strengths: list[str]
    key_weaknesses: list[str]
    competitive_position: str  # how they compare to target company
```

### IntelligenceReport
```python
class IntelligenceReport(BaseModel):
    title: str
    executive_summary: str
    detailed_analysis: str              # full analysis narrative
    competitor_summaries: list[CompetitorSummary]  # per-competitor breakdown
    recommendations: list[str]
    generated_at: datetime
```

## Custom Tool (`tools.py`)

### WebScraperTool

> **IMPLEMENTATION NOTE:** During development, we discovered that CrewAI agent
> tool use is broken with Ollama. CrewAI uses Instructor internally to parse
> ALL agent responses, and Instructor doesn't support the combination of
> tools + response_model with Ollama models. Additionally, when agents scrape
> multiple pages autonomously, context grows to 40k+ tokens, causing the model
> to return empty responses.
>
> **Resolution:** The WebScraperTool class exists and teaches the CrewAI BaseTool
> pattern, but it is called from Python code in the Flow (`flow.py`), not
> through an agent. The Flow pre-scrapes with controlled content size (2000
> chars/page) and passes the text to the Research Crew as context. This is a
> common production pattern: deterministic I/O in code, reasoning in the LLM.

Extends CrewAI's `BaseTool`. This is the first custom tool in the portfolio.

```python
class WebScraperTool(BaseTool):
    name: str = "Web Scraper"
    description: str = "Scrapes a website URL and returns clean text content"

    def _run(self, url: str) -> str:
        # 1. Fetch with requests (timeout=10s, User-Agent header set)
        # 2. Parse with BeautifulSoup
        # 3. Strip scripts, styles, nav elements
        # 4. Extract title + clean text
        # 5. Truncate to ~4000 chars to avoid overwhelming LLM context
        # 6. Return formatted string with title + content
        # 7. On error (timeout, bad status, invalid URL), return an error
        #    message string (not an exception) so the agent can try
        #    alternative URLs or proceed with partial data
```

**Concept comparison (tutor comment in file):**
- LangGraph P2: `@tool` decorated function, bound with `llm.bind_tools()`, executed by `ToolNode`
- CrewAI P6: `BaseTool` subclass, assigned to agent via `tools=[WebScraperTool()]`, agent decides when to use it

## Research Crew (`research_crew.py`)

**Process:** `Process.sequential`

### Agent 1: Web Scraper Specialist
- **Role:** "Web Scraper Specialist"
- **Goal:** Scrape provided URLs and extract clean, relevant content about companies
- **Tools:** `[WebScraperTool()]`
- **Task:** Given company names and optional URLs, scrape pages and return structured content
- **Task output:** Scraped content as text (intermediate — not the crew's final output)

### Agent 2: Intelligence Gatherer
- **Role:** "Intelligence Gatherer"
- **Goal:** Organize raw scraped content into structured competitor profiles
- **Tools:** None (works with Scraper's output)
- **Task:** Parse raw scraped data, identify key information, produce structured profiles
- **Task output:** `CompetitorProfile` list (this is the crew's final output, since it's the last task in sequential mode)

**Crew-level output:** In `Process.sequential`, the crew returns the last task's output. So the Research Crew returns the Gatherer's `CompetitorProfile` list. The `research_phase()` method stores this in `self.state.competitor_profiles`.

**Tutor focus:** Explain `Process.sequential` — tasks run in fixed order, output of task 1 feeds into task 2. The crew's output = the last task's output. Compare to P3 where everything was hierarchical.

## Analysis Crew (`analysis_crew.py`)

**Process:** `Process.hierarchical` — manager LLM coordinates agents
**Manager LLM:** Same `LLM_MODEL` passed as `manager_llm` parameter to `Crew()`

### Agent 1: Market Analyst
- **Role:** "Market Analyst"
- **Goal:** Analyze each competitor's market position, target audience, and business model
- **Task:** Take competitor profiles and produce market positioning analysis

### Agent 2: Feature Comparator
- **Role:** "Feature Comparator"
- **Goal:** Build side-by-side comparisons of products, features, and pricing
- **Task:** Create a structured comparison matrix across all competitors

### Agent 3: Strategic Advisor
- **Role:** "Strategic Advisor"
- **Goal:** Identify opportunities, threats, and strategic recommendations
- **Task:** Synthesize analyst's and comparator's findings into actionable insights

**Output:** `CompetitiveAnalysis` Pydantic model

**Tutor focus:** Explain why hierarchical fits here — the manager can adapt task delegation based on what the Analyst discovers. Compare sequential (predictable pipeline) vs hierarchical (adaptive coordination).

## Report Crew (`report_crew.py`)

**Process:** `Process.sequential`

### Agent 1: Intelligence Report Writer
- **Role:** "Intelligence Report Writer"
- **Goal:** Transform competitive analysis into a clear, actionable executive briefing
- **Task:** Write polished report with executive summary, detailed findings, recommendations

### Agent 2: Report Editor
- **Role:** "Report Editor"
- **Goal:** Review and polish the report for clarity, accuracy, and actionability
- **Task:** Review draft, check consistency with source analysis, improve structure
- **Output:** `IntelligenceReport` Pydantic model

**Tutor focus:** Show how structured output ensures the final deliverable has a predictable shape, regardless of LLM creativity.

## The Flow (`flow.py`)

### IntelligenceFlow

```python
class IntelFlowState(BaseModel):
    target_company: str = ""
    competitors: list[str] = []
    urls: list[str] = []
    # Each phase stores its output as a string (crew.kickoff() returns
    # a CrewOutput whose .raw is the text representation).
    # We pass this raw text into the next crew's task description,
    # so the LLM reads it as context — no manual deserialization needed.
    competitor_profiles: str = ""   # raw output from Research Crew
    analysis: str = ""              # raw output from Analysis Crew
    report: str = ""                # raw output from Report Crew

class IntelligenceFlow(Flow[IntelFlowState]):
    @start()
    def research_phase(self):
        # Build and kick off Research Crew
        result = research_crew.kickoff(inputs={...})
        self.state.competitor_profiles = result.raw
        return result.raw

    @listen(research_phase)
    def analysis_phase(self, research_output):
        # Build and kick off Analysis Crew
        # research_output = return value from research_phase
        result = analysis_crew.kickoff(inputs={...})
        self.state.analysis = result.raw
        return result.raw

    @listen(analysis_phase)
    def report_phase(self, analysis_output):
        # Build and kick off Report Crew
        result = report_crew.kickoff(inputs={...})
        self.state.report = result.raw
        return result.raw

    @listen(report_phase)
    def save_report(self, report_output):
        # Print formatted report to console
        # Save markdown file to reports/ directory
        # (This is a plain function, no crew — shows Flow flexibility)
```

**Data flow between crews:** Each crew's `kickoff()` returns a `CrewOutput` object. We store `result.raw` (the text output) in FlowState, and pass it into the next crew's task description as context. The `@listen` method also receives the return value of the previous step as a parameter. This avoids manual JSON serialization — the LLM reads the previous crew's output as natural text.

**Tutor focus:**
- `@start()` = entry point (like LangGraph's `START` → first node)
- `@listen(prev)` = run after prev (like `add_edge("prev", "next")`)
- `FlowState` = shared state (like LangGraph's `TypedDict` state)
- Plain function steps show Flows aren't limited to crew executions

## Memory Configuration

All three crews enable memory:

```python
crew = Crew(
    agents=[...],
    tasks=[...],
    memory=True,    # enables short-term + long-term + entity memory
    verbose=True,
)
```

**What this enables:**
- **Short-term:** Agents within a run share context (Comparator sees what Analyst found)
- **Long-term:** Across runs, crews remember past research (SQLite backend, managed by CrewAI)
- **Entity memory:** Remembers facts about specific companies across runs

**Tutor focus:** Explain each memory type, how CrewAI manages storage internally, and compare to LangGraph (where you'd build this yourself with a database + checkpointing).

## CLI (`main.py`)

Interactive prompt:

```
🔍 Competitive Intelligence System

Enter target company: <user input>
Enter competitors (comma-separated): <user input>
Enter URLs to research (optional, comma-separated, or press Enter to skip): <user input>

Starting intelligence gathering...
```

If no URLs provided, the Scraper agent uses LLM reasoning to construct reasonable URLs from company names (e.g., `https://company.com`, `https://en.wikipedia.org/wiki/Company`). The agent autonomously decides which URLs to try and calls `WebScraperTool` for each. If a URL fails, the tool returns an error message and the agent can try alternatives.

Output:
1. Formatted report printed to console
2. Markdown file saved to `reports/<target>_vs_<competitors>_<timestamp>.md`

## LLM Configuration

```python
LLM_MODEL = "ollama/qwen3.5:35b"
```

Single constant, easy to swap to a smaller model for faster execution. Used across all three crews and the hierarchical manager.

## Dependencies

New dependencies needed (add to `pyproject.toml`):
- `requests` — HTTP requests for web scraping (likely already available via other deps)
- `beautifulsoup4` — HTML parsing

CrewAI Flows should be available in `crewai ^1.11.0` (already installed).

## Entry Points

**Thin wrapper** (`project_6_intel_crew/main.py`):
```python
from langgraph_portfolio.projects.project_6_intel_crew.main import main
main()
```

**pyproject.toml** — add script entry:
```toml
project-6-intel-crew = "langgraph_portfolio.projects.project_6_intel_crew.main:main"
```

**reports/ directory** — add to `.gitignore` so generated reports aren't committed.

## Concept Progression from Previous Projects

| Concept | Where Introduced | P6 Evolution |
|---------|-----------------|--------------|
| CrewAI basics | P3 (single crew) | P6: multiple crews chained via Flows |
| Structured output | P3 (`output_pydantic`) | P6: same pattern, more complex models |
| Process types | P3 (hierarchical only) | P6: sequential + hierarchical side by side |
| Custom tools | P2 (LangGraph `@tool`) | P6: CrewAI `BaseTool` class pattern |
| Multi-component | P5 (LangGraph sub-graphs) | P6: CrewAI Flows (same idea, different framework) |
| Memory | New | P6: CrewAI's built-in memory system |

## Known Issues / Fixes Needed

### 1. Memory requires OpenAI API key (must fix)

**Error:** `Memory requires an LLM for analysis but initialization failed: OPENAI_API_KEY is required`

**Cause:** `memory=True` on `Crew(...)` defaults to using `gpt-4o-mini` for memory analysis (embedding + retrieval). Since we use Ollama, not OpenAI, this fails.

**Fix:** Pass an explicit `memory_config` or configure the memory embedder to use Ollama. Options:
- **Option A (recommended):** Configure memory with Ollama embedder:
  ```python
  from crewai.memory.storage import LTMSQLiteStorage
  crew = Crew(
      ...,
      memory=True,
      embedder={"provider": "ollama", "config": {"model": "nomic-embed-text"}},
  )
  ```
- **Option B:** Disable memory entirely (`memory=False`) if it's not essential for the demo — simplest fix, loses the learning concept.
- **Option C:** Use `depth="shallow"` for recall to skip LLM analysis.

**Affects:** All three crews (`research_crew.py`, `analysis_crew.py`, `report_crew.py`).

### 2. `output_pydantic` incompatible with Ollama/Instructor (must fix)

**Error:** `Instructor does not support multiple tool calls, use List[Model] instead` (AssertionError)

**Cause:** CrewAI uses Instructor to force structured output when a task has `output_pydantic=<Model>`. Instructor sends a tool-call schema to the LLM and expects exactly one tool call back. But Ollama/qwen3.5 returns `tool_calls=None` (empty content, no tool call), so Instructor's assertion `len(message.tool_calls or []) == 1` fails. This is the same fundamental Ollama + Instructor incompatibility we hit in P3.

**Failing tasks:**
- `profile_task` → `output_pydantic=CompetitorProfile` (research_crew.py)
- `strategy_task` → `output_pydantic=CompetitiveAnalysis` (analysis_crew.py)
- `edit_task` → `output_pydantic=IntelligenceReport` (report_crew.py)

**Fix options:**
- **Option A (recommended):** Remove `output_pydantic` from all tasks. Instead, add explicit JSON formatting instructions in the task description (tell the agent to output valid JSON matching the schema). Then parse `result.raw` with Pydantic in the flow step (`Model.model_validate_json(result.raw)`). This is the same "control structured output from Python" pattern we use in P3.
- **Option B:** Use `output_json` instead of `output_pydantic` — may or may not hit the same Instructor issue (needs testing).

**Affects:** All three crews — every task that uses `output_pydantic`.
