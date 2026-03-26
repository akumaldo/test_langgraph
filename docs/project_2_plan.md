# Project 2: Research Agent — Implementation Plan

## Goal

Build a research agent that can search documents and synthesize answers with citations, using real LangGraph (not the scaffold).

## New Concepts (vs Project 1 Chatbot)

1. **Tools** — giving the LLM a search function it can call (tool calling / function calling)
2. **The Agent Loop** — LLM calls tool, gets results, decides if it needs more info or is done (loop pattern)
3. **Citations / Source Tracking** — tracking which documents were used to produce the answer (RAG pattern)

## The Graph Shape

```
START
  |
agent (LLM thinks)  <---------+
  |                            |
should_continue?               |
  +-- tool call? --> tools ----+
  +-- final answer? --> END
```

Two nodes + a conditional edge that creates the loop.

## Build Steps

### Step 1: State Definition
- Define `ResearchState` as a `TypedDict`
- Fields: messages (with add_messages reducer), documents found, citations, summary

### Step 2: The Search Tool
- Create a `search_documents` function decorated as a LangChain tool
- Uses the existing `KnowledgeBase` + `sample_documents` from core
- Takes a query string, returns matching document content

### Step 3: Node Functions
- **Agent node**: the LLM with the search tool bound to it (via `.bind_tools()`)
- **Tool node**: LangGraph's built-in `ToolNode` that executes tool calls

### Step 4: Graph Assembly
- `StateGraph(ResearchState)`
- Add agent node and tool node
- Conditional edge after agent: tool call → tools node, final answer → END
- Edge from tools → back to agent (creates the loop)
- Compile and run

## Key Differences from Chatbot

| Chatbot | Research Agent |
|---------|---------------|
| LLM just talks | LLM can call tools |
| Linear flow (no loops) | Loop: agent → tool → agent → ... |
| Single response | Gathers info iteratively, then synthesizes |
| No external data | Searches a document knowledge base |
| No citations | Tracks sources for every claim |

## Files to Create

- `src/langgraph_portfolio/projects/project_2_research_agent/langgraph_research.py` — the real LangGraph implementation
