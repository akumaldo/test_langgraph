# Project 3 (CrewAI): Writing Team — Implementation Plan

## Goal

Build the SAME writing team (5 roles producing an article) using **CrewAI** instead of LangGraph. Same problem, different framework — so you can compare approaches side by side.

## Why This Matters

You already built this with LangGraph. You know what's involved: state, nodes, edges, routing logic. Now you'll see how CrewAI solves the same problem with a completely different philosophy.

The core question: **who manages the workflow?**
- LangGraph: **you** manage it (you draw the graph, write the routing function)
- CrewAI: **an AI manager** manages it (you describe the team, the manager coordinates)

## New Concepts (vs LangGraph Writing Team)

1. **Agents as objects** — in LangGraph, an "agent" was just a function with a prompt inside. In CrewAI, agents are objects with `role`, `goal`, and `backstory` — the framework builds prompts from these.

2. **Tasks (separate from agents)** — in LangGraph, what an agent does was baked into its function. In CrewAI, you define Tasks separately and ASSIGN them to Agents. WHO and WHAT are decoupled.

3. **Process types** — instead of wiring edges, you pick a process:
   - `Process.sequential`: tasks run in order (like LangGraph's linear edges, but no loops)
   - `Process.hierarchical`: a manager agent decides who works when, including sending work back

4. **Hierarchical delegation** — the manager agent is an "invisible 6th member" that reads your task list and coordinates the team. It can re-assign tasks, request revisions, and decide when quality is met. This replaces our hand-coded `after_editor()` routing function.

5. **Hidden state** — in LangGraph, you defined WritingState with explicit fields (outline, draft, etc.) and could inspect them after running. In CrewAI, intermediate results flow between tasks internally — you see the final output, not the plumbing.

## How It Compares

| Aspect | LangGraph Writing Team | CrewAI Writing Team |
|--------|----------------------|---------------------|
| Agent definition | Function with SystemMessage | Object with role/goal/backstory |
| Task definition | Baked into the function | Separate Task objects |
| Flow control | Explicit edges + routing function | Process type (hierarchical) |
| Revision loop | `after_editor()` + `MAX_REVISIONS` | Manager agent's judgment |
| Loop safety | Hard cap in YOUR code | `max_iter` config parameter |
| State | TypedDict you define and inspect | Hidden, managed by framework |
| Intermediate results | `state["outline"]`, `state["draft"]` | verbose console logs |
| LLM calls | You write `llm.invoke([...])` | Framework calls LLM for you |
| Predictability | Deterministic (you control every path) | Non-deterministic (AI decides) |
| Boilerplate | More (edges, state, routing) | Less (declarative setup) |

## The "Graph" Shape

There IS no graph in CrewAI. Instead:

```
        ┌──────────────────────────────┐
        │     MANAGER AGENT            │
        │  (created by CrewAI)         │
        │                              │
        │  Reads the task list.        │
        │  Delegates to agents.        │
        │  Reviews results.            │
        │  Decides: revise or move on. │
        └──────────┬───────────────────┘
                   │ delegates
       ┌───────────┼───────────────────────┐
       ▼           ▼           ▼           ▼           ▼
   ┌────────┐ ┌──────────┐ ┌────────┐ ┌────────┐ ┌───────────┐
   │Planner │ │Researcher│ │ Writer │ │ Editor │ │ Publisher │
   └────────┘ └──────────┘ └────────┘ └────────┘ └───────────┘
```

The manager decides the order. It will likely follow the natural task sequence (plan → research → write → edit → publish), but it CAN deviate — for example, sending the writer's draft back after the editor gives feedback.

Compare this to LangGraph's explicit graph:
```
START → planner → researcher → writer → editor → [conditional] → publisher → END
                                  ↑                              ↓
                                  └──────── revision ────────────┘
```

## Build Steps

### Step 1: Define the LLM

- CrewAI uses LiteLLM format: `"ollama/qwen3.5:35b"`
- One string, shared across all agents
- Compare: LangGraph used `ChatOllama(model="qwen3.5:35b")` and you called `.invoke()` yourself

### Step 2: Define the Agents (5)

For each of the 5 roles, create an `Agent` object:
- `role` — their job title (replaces the SystemMessage content)
- `goal` — what they're trying to achieve
- `backstory` — extra personality/behavior context
- `llm` — which model to use
- `verbose=True` — so you can watch them think

Agents DON'T define what to do — that's the Tasks.

### Step 3: Define the Tasks (5)

For each step of the workflow, create a `Task` object:
- `description` — what needs to be done (with `{topic}` placeholder)
- `expected_output` — what the result should look like
- `agent` — who is assigned to it (suggestion in hierarchical mode)

Tasks are the WHAT. In LangGraph, the what and who were combined in each node function.

### Step 4: Assemble the Crew

Create a `Crew` with:
- `agents` — the list of 5 agents
- `tasks` — the list of 5 tasks
- `process=Process.hierarchical` — lets the manager coordinate
- `manager_llm` — which model the manager uses
- `verbose=True` — watch the delegation happen

Compare: LangGraph needed `add_node()` x5, `add_edge()` x5, `add_conditional_edges()`, `set_entry_point()`, `compile()`.

### Step 5: Run It

- `crew.kickoff(inputs={"topic": "..."})` — fills `{topic}` placeholders and starts
- Returns a `CrewOutput` object with `.raw` (the final text)
- Compare: LangGraph's `app.invoke({...})` returned the full state dict

## Key Takeaways for Study

1. **Trade-off: control vs convenience.** LangGraph makes you build everything but you control everything. CrewAI handles orchestration but you lose visibility and determinism.

2. **The manager is the graph.** In LangGraph, YOUR routing function was the decision-maker. In CrewAI hierarchical, a MANAGER LLM is the decision-maker. Same role, different executor.

3. **State is a spectrum.** LangGraph: fully explicit state you can inspect. CrewAI: hidden state flowing between tasks. Both work, but debugging is very different.

4. **Neither is "better."** LangGraph for complex, custom flows where you need predictability. CrewAI for quick team setups where approximate coordination is fine.

## Files to Create

- `src/langgraph_portfolio/projects/project_3_writing_team_crewai/crewai_writing_team.py` — the real CrewAI implementation
