# Project 3: Writing Team — Implementation Plan

## Goal

Build a multi-agent writing team where different LLM "agents" (each with a specialized role) collaborate to produce an article, using real LangGraph.

## New Concepts (vs Project 2 Research Agent)

1. **Multi-agent coordination** — multiple LLMs with different system prompts, each playing a role (planner, researcher, writer, editor, publisher)
2. **Quality-control loop** — the editor can reject work and send it back to the writer (conditional loop based on LLM judgment, not tool calls)
3. **Shared state across agents** — all agents read from and write to the same state, building on each other's work
4. **Revision tracking** — counting how many edit rounds happened, with a max to prevent infinite loops

## How It Compares to Previous Projects

| Concept | P1 Chatbot | P2 Research Agent | P3 Writing Team |
|---------|-----------|-------------------|-----------------|
| LLMs | 1 | 1 | Multiple (5 roles) |
| Tools | None | search_documents | search (reuse from P2) |
| Loop type | None | Tool-calling loop | Quality-control loop |
| Flow shape | Fork | Roundabout | Pipeline + feedback loop |
| Routing | Confidence check | Tool call check | Editor judgment |

## The Graph Shape

```
START → planner → researcher → writer → editor → good enough?
                                  ↑                    |
                                  └──── NO ────────────┘
                                        YES → publisher → END
```

- The planner → researcher → writer path is a straight pipeline
- The editor → writer loop is the new pattern: a feedback/revision loop
- We cap revisions (e.g., max 2 rounds) to prevent infinite loops

## Build Steps

### Step 1: State Definition
- `WritingState` TypedDict with `add_messages` reducer
- Fields: messages, topic, outline, draft, editor_feedback, revision_count, final_article

### Step 2: The Nodes (5 Agents)
Each node is a function that calls the LLM with a different system prompt:
- **planner** — given a topic, produces an outline
- **researcher** — given an outline, searches for info (reuses search_documents tool from P2)
- **writer** — given outline + research + feedback (if any), writes/revises the draft
- **editor** — reads the draft, decides: approve or request revisions
- **publisher** — assembles the final article with citations

### Step 3: Routing Logic
- After editor: check if approved or needs revision
  - Approved → publisher
  - Needs revision (and revision_count < max) → writer
  - Max revisions reached → publisher anyway (ship it)

### Step 4: Graph Assembly
- StateGraph with all 5 nodes
- Linear edges: planner → researcher → writer → editor
- Conditional edge: editor → (writer OR publisher)
- Linear edge: publisher → END

## Files to Create

- `src/langgraph_portfolio/projects/project_3_writing_team_langgraph/langgraph_writing_team.py` — the real LangGraph implementation
