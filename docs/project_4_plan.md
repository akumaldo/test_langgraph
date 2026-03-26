# Project 4: Autonomous Data Analyst — Implementation Plan

## Goal

Build a data analyst agent that takes a question about a CSV dataset, decomposes it into analysis steps, executes Python/pandas code, generates matplotlib charts, handles errors with retry logic, and produces a final report — all using real LangGraph with Pydantic structured output.

## Why This Project Matters

In Projects 1-3, the LLM only produced TEXT (outlines, articles, reviews). Here the LLM produces CODE that gets EXECUTED. This is a fundamentally different kind of agent:

- P2 Research Agent: LLM calls a search tool (reads data)
- P4 Data Analyst: LLM writes and runs Python code (transforms data, creates artifacts)

The agent doesn't just think — it ACTS on real data and produces real files (charts).

## New Concepts (vs Project 3 Writing Team)

1. **Pydantic structured output with `with_structured_output()`** — instead of parsing freeform LLM text, we force the LLM to return typed Python objects. The LLM fills out a "form" (Pydantic model) instead of writing free text. We mentioned this in the CrewAI project — now we use it natively in LangGraph.

2. **Code execution as a tool** — the agent writes Python/pandas code, and a tool node EXECUTES it. This is different from P2's search tool (which just looked up text). Here the tool runs arbitrary code — much more powerful, but needs safety guardrails.

3. **Real data (CSV)** — the agent works with an actual CSV file you provide. It reads the data, understands the columns, and reasons about what analysis to perform. No more hardcoded fixtures.

4. **Artifact generation** — the agent produces FILES (matplotlib charts), not just text. The state tracks these artifacts (file paths) so the report can reference them.

5. **Retry/recovery loop** — when code execution fails (syntax error, wrong column name, etc.), the agent sees the error, fixes its code, and retries. Compare to P3's editor loop: there the EDITOR decided to loop. Here the ERROR decides — it's automatic recovery, not quality judgment.

6. **LangGraph checkpointing** — LangGraph's built-in `MemorySaver` checkpointer replaces our manual JSON save/load from the scaffold. The framework snapshots state after every node, so you can resume interrupted runs.

7. **Progress tracking** — the state tracks progress (0.0 → 1.0) through the analysis pipeline, so the user knows where the agent is.

## How It Compares to Previous Projects

| Concept | P2 Research Agent | P3 Writing Team | P4 Data Analyst |
|---------|------------------|-----------------|-----------------|
| LLMs | 1 | 5 roles | 1 (analyst) |
| Tools | search_documents | None | execute_code, plot_chart |
| Loop type | Tool-calling loop | Quality-control loop | **Error-recovery loop** |
| Data | Fixture documents | None (LLM knowledge) | **Real CSV file** |
| Output | Text + citations | Text (article) | **Text + chart files** |
| State validation | TypedDict (loose) | TypedDict (loose) | **Pydantic models (strict)** |
| LLM output | Raw text | Raw text | **Structured (Pydantic)** |
| Checkpointing | None | None | **LangGraph MemorySaver** |

## The Graph Shape

```
START → load_data → plan_analysis → execute_step → error?
                                         ↑            |
                                         |     YES → fix_and_retry ──┐
                                         |                           |
                                         └───────────────────────────┘
                                         |
                                    NO → more steps?
                                         |
                                    YES → execute_step (loop for each step)
                                    NO  → generate_charts → write_report → END
```

Three types of flow, building on everything we've learned:
- **Pipeline** (like P3): load → plan → execute → charts → report
- **Step loop** (new): execute_step runs once per analysis step from the plan
- **Error recovery** (new): if code fails, fix it and retry (with max retries)

## Pydantic Models (Structured Output)

These models define the "forms" the LLM fills out at each step:

### AnalysisPlan
```python
class AnalysisStep(BaseModel):
    description: str       # "Calculate monthly revenue trend"
    code: str              # "df.groupby('month')['revenue'].sum()"
    explanation: str       # "Groups by month and sums revenue"

class AnalysisPlan(BaseModel):
    steps: list[AnalysisStep]  # 3-5 steps to answer the question
    reasoning: str              # Why these steps answer the question
```

### ChartSpec
```python
class ChartSpec(BaseModel):
    chart_type: str        # "line", "bar", "scatter", "histogram"
    x_column: str          # Column name for x-axis
    y_column: str          # Column name for y-axis
    title: str             # Chart title
    code: str              # matplotlib code to generate the chart
```

### CodeFix
```python
class CodeFix(BaseModel):
    original_code: str     # The code that failed
    error_message: str     # What went wrong
    fixed_code: str        # The corrected code
    explanation: str       # What was wrong and how it was fixed
```

### AnalysisReport
```python
class AnalysisReport(BaseModel):
    title: str
    summary: str           # Key findings
    sections: list[...]    # Detailed analysis per step
    chart_references: list[str]  # Paths to generated charts
```

## Build Steps

### Step 1: State Definition
- `DataAnalystState` as TypedDict with `add_messages` reducer
- Fields: messages, question, csv_path, dataframe_info, plan, current_step_index, step_results, errors, retries, charts, report, progress
- This is our most complex state yet — tracks the full analysis lifecycle

### Step 2: The Tools
Two LangChain tools:
- **`execute_python`** — takes Python code as a string, runs it with exec() in a controlled namespace (with pandas/numpy loaded), returns the output or error
- **`save_chart`** — takes matplotlib code, executes it, saves the figure to a file, returns the path

### Step 3: Node Functions
- **load_data** — reads the CSV, stores column info and basic stats in state (so the LLM knows what data it has)
- **plan_analysis** — LLM with `with_structured_output(AnalysisPlan)` creates a step-by-step plan
- **execute_step** — runs the current step's code using the execute_python tool
- **fix_and_retry** — LLM with `with_structured_output(CodeFix)` reads the error and produces fixed code
- **generate_charts** — LLM with `with_structured_output(ChartSpec)` decides what charts to create, then runs matplotlib code
- **write_report** — LLM with `with_structured_output(AnalysisReport)` produces the final report

### Step 4: Routing Logic
Three routing decisions:
1. **after_execute_step**: error? → fix_and_retry / no error + more steps? → execute_step / done? → generate_charts
2. **after_fix_and_retry**: retries < max? → execute_step / give up → generate_charts (skip failed step)
3. Simple edges for the rest of the pipeline

### Step 5: Graph Assembly + Checkpointing
- StateGraph with all nodes
- Pipeline edges + conditional edges for the two decision points
- `MemorySaver` checkpointer so state is saved after every node
- Compare to scaffold's manual `save_checkpoint()` / `load_checkpoint()`

### Step 6: Interactive Loop (User Input)
- Simple `input()` loop OUTSIDE the graph
- User types a question → graph runs → prints report + charts → asks for next question
- Type "quit" to exit
- The graph itself is stateless per run — each question is a fresh invocation
- This is the simplest form of human-in-the-loop: the human controls WHAT to analyze, the agent controls HOW
- (LangGraph also supports `interrupt()` for mid-graph user input — a future enhancement)

## Files to Create

- `src/langgraph_portfolio/projects/project_4_data_analyst/langgraph_data_analyst.py` — the real LangGraph implementation
- `src/langgraph_portfolio/projects/project_4_data_analyst/models.py` — Pydantic structured output models
- `src/langgraph_portfolio/projects/project_4_data_analyst/tools.py` — execute_python and save_chart tools

## Key Takeaways for Study

1. **Structured output changes everything.** Instead of hoping the LLM writes valid code in freeform text, we FORCE it to return a typed object with a `code` field. Pydantic validates the structure before we execute anything.

2. **Code execution is the most powerful (and dangerous) tool.** The agent can do anything pandas/matplotlib can do — but bad code can crash. That's why the error-recovery loop exists.

3. **The retry loop is error-driven, not judgment-driven.** In P3, the editor DECIDED to loop based on quality. Here, the SYSTEM detects a failure (exception) and triggers the fix. It's mechanical, not subjective.

4. **Checkpointing is free in LangGraph.** The scaffold needed 15 lines of manual JSON save/load. LangGraph's `MemorySaver` does it in one line — and it snapshots after EVERY node, not just at the end.

5. **Real data forces real robustness.** With fixture data, everything is predictable. With a real CSV, the LLM might guess wrong column names, write bad pandas code, or produce ugly charts. The error-recovery loop handles this gracefully.
