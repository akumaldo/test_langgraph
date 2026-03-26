# Project 5: Customer Support Bot — Design Spec

## Overview

An intelligent customer support system built with real LangGraph that teaches three advanced patterns: **sub-graphs** (graphs inside graphs), **interrupt()** (human-in-the-loop mid-execution), and **streaming** (token-by-token output).

No scaffold version — this project goes straight to real LangGraph.

## Decisions

- **Sub-graph isolation**: Each department sub-graph has its own state type. Isolation is achieved via `input`/`output` schemas on each sub-graph's `StateGraph`, and the compiled sub-graph is added directly as a node in the parent. This is the production pattern that also supports `interrupt()` bubbling up.
- **LLM intensity gradient**: Technical sub-graph is fully LLM-driven, billing is medium (one LLM call), returns is rule-based. Keeps run time reasonable with local Ollama.
- **Directory rename**: Old `project_5_capstone` directories renamed to `project_5_support_bot`.

## State Architecture

### Parent State

```python
class SupportState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    customer_issue: str            # raw user input
    category: str                  # "billing" | "technical" | "returns" | "unknown"
    department_response: str       # final answer from whichever sub-graph ran
    satisfaction_rating: int       # 1-5 from feedback collection (initial: 0)
```

`messages` is used by parent-level nodes: `classify_issue` appends an AIMessage with the classification result, `collect_feedback` appends the feedback exchange, and `fallback_response` appends its response. This gives a full conversation trace at the parent level.

Initial values when invoking the graph:
- `messages`: `[HumanMessage(content=customer_issue)]`
- `customer_issue`: the user's input
- `category`: `""` (filled by classifier)
- `department_response`: `""` (filled by sub-graph)
- `satisfaction_rating`: `0` (filled by collect_feedback after interrupt)

### Department States (isolated via input/output schemas)

Each sub-graph defines three types:
- **Full state** — all internal fields the sub-graph needs
- **Input schema** — only the fields the parent passes in (subset of keys shared with parent)
- **Output schema** — only the fields the parent gets back (subset of keys shared with parent)

Keys shared between parent and sub-graph must have the **same name**. LangGraph automatically maps matching key names when using compiled sub-graph nodes.

**TechnicalState** (full internal state):
- `customer_issue`, `os_info`, `diagnosis`, `suggested_fix`, `department_response`, `messages`
- **Input**: `customer_issue` (from parent)
- **Output**: `department_response` (back to parent)

**BillingState** (full internal state):
- `customer_issue`, `account_status`, `refund_amount`, `billing_response`, `department_response`, `messages`
- **Input**: `customer_issue`
- **Output**: `department_response`

**ReturnsState** (full internal state):
- `customer_issue`, `order_id`, `return_reason`, `return_eligible`, `department_response`, `messages`
- **Input**: `customer_issue`
- **Output**: `department_response`

### How State Mapping Works

```python
# Sub-graph defines input/output schemas for isolation
class TechnicalInput(TypedDict):
    customer_issue: str

class TechnicalOutput(TypedDict):
    department_response: str

# StateGraph uses input/output to control what flows in and out
technical_graph = StateGraph(TechnicalState, input=TechnicalInput, output=TechnicalOutput)
# ... add nodes and edges ...
compiled_technical = technical_graph.compile()

# Parent adds the compiled sub-graph DIRECTLY as a node
# LangGraph maps matching key names automatically
parent_graph.add_node("technical_subgraph", compiled_technical)
```

The parent passes `customer_issue` into the sub-graph (matching key name). The sub-graph works with all its internal fields (`os_info`, `diagnosis`, etc.) which are invisible to the parent. Only `department_response` flows back out.

This pattern is required for `interrupt()` to work inside sub-graphs — compiled sub-graph nodes are first-class LangGraph nodes, so interrupt bubbles up correctly to the parent's checkpointer.

## Graph Architecture

### Parent Graph

```
START → classify_issue → route_to_department ──→ billing_subgraph ────→ collect_feedback → END
                                               → technical_subgraph ──→ collect_feedback → END
                                               → returns_subgraph ───→ collect_feedback → END
                                               → fallback_response ──→ collect_feedback → END
```

- **classify_issue**: LLM call with `with_structured_output(IssueClassification)`. Returns category + confidence. If confidence < 0.5, overrides category to `"unknown"`. Appends an AIMessage with the classification result.
- **route_to_department**: Conditional edge based on `state["category"]`.
- **collect_feedback**: Uses `interrupt()` to pause and ask the user for a 1-5 satisfaction rating.
- **fallback_response**: Simple node for unrecognized categories.

### Technical Sub-graph (rich, LLM-driven)

```
START → diagnose → [interrupt: ask OS] → suggest_fix → draft_response → END
```

- `diagnose`: LLM analyzes the issue, then calls `interrupt("What operating system are you using?")` to get more info. The interrupt pauses the entire parent graph via the checkpointer.
- `suggest_fix`: LLM uses the diagnosis + OS info to propose a solution.
- `draft_response`: LLM writes a friendly customer-facing response.

### Billing Sub-graph (medium)

```
START → check_account → calculate_resolution → draft_response → END
```

- `check_account`: Simulated account lookup (returns hardcoded data).
- `calculate_resolution`: Rule-based logic (refund if overcharged, etc.).
- `draft_response`: One LLM call to write a friendly response.

### Returns Sub-graph (simple)

```
START → validate_return → process_return → draft_response → END
```

- `validate_return`: Checks return window (simulated).
- `process_return`: Determines eligibility (rule-based).
- `draft_response`: Templates the response, no LLM call.

## New Concepts

### 1. Sub-graphs with Isolated State (input/output schemas)

Each sub-graph is built as its own `StateGraph` with `input` and `output` schema parameters. The compiled sub-graph is added **directly as a node** in the parent graph via `add_node("name", compiled_subgraph)`. LangGraph automatically maps matching key names between parent and sub-graph states.

This is different from the "wrapper function" pattern (where you write a Python function that calls `.invoke()`). The compiled-node pattern is required because:
1. `interrupt()` inside the sub-graph correctly bubbles up to the parent's checkpointer
2. Streaming events from sub-graph nodes are visible in the parent's stream
3. LangGraph handles state mapping automatically via the input/output schemas

### 2. interrupt() — Human-in-the-Loop

Two interrupt points:
- **Inside technical sub-graph**: `diagnose` node pauses to ask for OS info. Teaches interrupt inside a sub-graph.
- **In parent graph**: `collect_feedback` pauses to ask for satisfaction rating. Teaches interrupt at the parent level.

Both require `MemorySaver` checkpointer on the parent graph. When a sub-graph calls `interrupt()`, execution bubbles up and pauses the entire parent graph. When the user responds via `Command(resume=value)`, execution resumes inside the sub-graph exactly where it left off.

### 3. Streaming

We use **one stream mode at a time** (each `.stream()` call picks one mode):

- `stream_mode="updates"`: Used for the main loop. Shows node-by-node updates — which node ran and what it produced. Good for progress feedback.
- `stream_mode="messages"`: Demonstrated separately to show token-by-token LLM output. Good for chat UX where you want text to appear as it's generated.

The interactive loop handles resume-after-interrupt:

```python
# Each .stream() call uses ONE mode
for chunk in app.stream(state, config=config, stream_mode="updates"):
    for node_name, update in chunk.items():
        print(f"[{node_name}] completed", flush=True)

# If interrupted, resume with Command:
for chunk in app.stream(Command(resume=user_input), config=config, stream_mode="updates"):
    for node_name, update in chunk.items():
        print(f"[{node_name}] completed", flush=True)
```

## Structured Output

**IssueClassification** (used by `classify_issue`):

```python
class IssueClassification(BaseModel):
    category: Literal["billing", "technical", "returns", "unknown"]
    confidence: float   # logged for observability; low confidence routes to "unknown"
    reasoning: str
```

## File Structure

```
src/langgraph_portfolio/projects/project_5_support_bot/
  __init__.py                      # exports build_support_bot, main
  models.py                        # Pydantic: IssueClassification
  langgraph_support_bot.py         # parent graph + classifier + wrapper nodes + streaming loop
  subgraphs/
    __init__.py
    billing.py                     # BillingState + input/output schemas + billing sub-graph
    technical.py                   # TechnicalState + input/output schemas + technical sub-graph (interrupt)
    returns.py                     # ReturnsState + input/output schemas + returns sub-graph
  main.py                          # implementation runner: interactive_loop + single_issue

project_5_support_bot/             # entry point (thin sys.path wrapper, imports from src)
  __init__.py
  main.py
  tests/
    __init__.py
    test_smoke.py
```

- `langgraph_support_bot.py` owns the parent graph, LLM setup, and the interactive loop
- `main.py` (in src) is the implementation runner with `interactive_loop()` and `single_issue()`
- `main.py` (entry point) is the thin wrapper that fixes `sys.path` and calls `main()`

## Testing

- **Smoke tests**: Verify each sub-graph compiles, parent graph compiles, classifier routing works with hardcoded categories (no LLM needed).
- **Manual testing**: Run with `--issue "I was charged twice"` for single-issue mode, or interactive mode for the full streaming + interrupt experience.

## Running

```bash
# Interactive mode (streaming + interrupt)
python project_5_support_bot/main.py

# Single issue (non-interactive)
python project_5_support_bot/main.py --issue "My internet keeps disconnecting"
```

## Rename Plan

Rename all `project_5_capstone` references to `project_5_support_bot`:
- `project_5_capstone/` → `project_5_support_bot/` (entry point directory)
- `project_5_capstone/tests/` → `project_5_support_bot/tests/` (entry point tests)
- `src/langgraph_portfolio/projects/project_5_capstone/` → `src/langgraph_portfolio/projects/project_5_support_bot/` (implementation)
- Update imports in entry point `main.py`
- Update `tests/test_repo_structure.py` line 18: change `project_5_capstone` to `project_5_support_bot`
- Delete old scaffold code (state.py, graph.py with GraphBuilder)
