# Portfolio Expansion Plan — Agentic AI Deep Dive

## Overview

This plan extends the original 5-project portfolio into an 11-project journey across multiple agentic AI frameworks and protocols. The first 4 projects built your foundation with LangGraph and CrewAI. Projects 5-10 go deeper and wider — advanced LangGraph patterns, CrewAI Flows, AG2, BeeAI, LlamaIndex Workflows, and a capstone that compares them all. Project 11 adds MCP (Model Context Protocol) — the standard for connecting AI clients to external services.

## Learning Arc

```
FOUNDATION (done)                    INTERMEDIATE                     ADVANCED
─────────────────                    ────────────                     ────────
P1 Chatbot (LangGraph)               P5 Support Bot (LangGraph)       P9  RAG Pipeline (LlamaIndex)
P2 Research Agent (LangGraph)         P6 Intel Crew (CrewAI)           P10 Framework Showdown (All)
P3 Writing Team (LangGraph + CrewAI)  P7 Debate Arena (AG2)            P11 Job Search MCP Server (MCP)
P4 Data Analyst (LangGraph)           P8 Research Agent v2 (BeeAI)
```

Each project introduces 3-4 new concepts while reusing what you already know.

## Concept Coverage Map

| Concept | P1 | P2 | P3 | P4 | P5 | P6 | P7 | P8 | P9 | P10 | P11 |
|---------|----|----|----|----|----|----|----|----|----|----|-----|
| State + nodes + edges | ✅ | ✅ | ✅ | ✅ | ✅ | | | | | ✅ | |
| Conditional routing | ✅ | ✅ | ✅ | ✅ | ✅ | | | | | ✅ | |
| Tool calling | | ✅ | | ✅ | ✅ | ✅ | | ✅ | | ✅ | |
| Agent loop | | ✅ | | | | | | ✅ | | ✅ | |
| Multi-agent | | | ✅ | | ✅ | ✅ | ✅ | | | ✅ | |
| Quality-control loop | | | ✅ | | | | | | | | |
| Error-recovery loop | | | | ✅ | | | | | | | |
| Structured output (Pydantic) | | | ✅ | ✅ | ✅ | | | | | | ✅ |
| Checkpointing | | | | ✅ | ✅ | | | | | | |
| **Sub-graphs** | | | | | ✅ | | | | | | |
| **interrupt() / HITL** | | | | | ✅ | | | | | | |
| **Streaming** | | | | | ✅ | | | | | | |
| **CrewAI Flows** | | | | | | ✅ | | | | | |
| **CrewAI custom tools** | | | | | | ✅ | | | | | |
| **CrewAI memory** | | | | | | ✅ | | | | | |
| **AG2 conversations** | | | | | | | ✅ | | | | |
| **AG2 group chat** | | | | | | | ✅ | | | | |
| **AG2 speaker selection** | | | | | | | ✅ | | | | |
| **BeeAI agents** | | | | | | | | ✅ | | | |
| **BeeAI observability** | | | | | | | | ✅ | | | |
| **Event-driven workflows** | | | | | | | | | ✅ | | |
| **RAG strategies** | | | | | | | | | ✅ | | |
| **Cross-framework comparison** | | | | | | | | | | ✅ | |
| **MCP protocol (stdio/JSON-RPC)** | | | | | | | | | | | ✅ |
| **MCP resources (read-only)** | | | | | | | | | | | ✅ |
| **MCP tools (write actions)** | | | | | | | | | | | ✅ |
| **MCP prompts (templates)** | | | | | | | | | | | ✅ |
| **Least-privilege boundaries** | | | | | | | | | | | ✅ |
| **Server-side input validation** | | | | | | | | | | | ✅ |

---

## Project 5: Customer Support Bot (LangGraph — Advanced)

### What You're Building

An intelligent customer support system that:
1. Classifies the customer's issue (billing, technical, returns)
2. Routes to a specialized sub-graph for that department
3. Each sub-graph has its own flow (e.g., billing checks account status, technical runs diagnostics)
4. The agent can PAUSE mid-conversation to ask the user for more info (`interrupt()`)
5. Responses stream token-by-token (the user sees text appearing, not a blank screen)

### Why This Project

Projects 1-4 used LangGraph in its simplest form: flat graphs with nodes at one level. Real applications need COMPOSITION — graphs inside graphs, like functions calling functions. This project teaches the three most important advanced LangGraph patterns.

### New Concepts

#### 1. Sub-graphs (Graphs Inside Graphs)

In P1-P4, all nodes lived in ONE graph:

```
START → node_A → node_B → node_C → END
```

But what happens when different departments need different flows? You don't want 20 nodes in one giant graph. Instead, you build a SMALL graph for each department and nest them inside a parent graph:

```
Parent graph:
  START → classify → [conditional] → billing_subgraph → collect_feedback → END
                                    → technical_subgraph → collect_feedback → END
                                    → returns_subgraph → collect_feedback → END

Billing sub-graph (its own StateGraph):
  START → check_account → calculate_refund → draft_response → END

Technical sub-graph (its own StateGraph):
  START → diagnose → check_logs → suggest_fix → END
```

Each sub-graph is a COMPILED graph that acts like a single node in the parent. The parent passes state IN, the sub-graph processes it, and returns state OUT. It's like function composition — each sub-graph is a self-contained unit.

This matters because:
- Each department team can develop their sub-graph independently
- You can test sub-graphs in isolation
- The parent graph stays clean and readable
- You can swap or update a department flow without touching the others

In LangGraph, you do this by compiling a sub-graph and adding it as a node:

```python
# Build the billing sub-graph
billing_graph = StateGraph(BillingState)
billing_graph.add_node(...)
billing_compiled = billing_graph.compile()

# Use it as a node in the parent
parent_graph = StateGraph(SupportState)
parent_graph.add_node("billing", billing_compiled)  # <-- sub-graph as a node
```

#### 2. interrupt() — Human-in-the-Loop Mid-Execution

In P4, the human interacted OUTSIDE the graph (the input() loop). The graph ran to completion, then the human asked a new question. The human controlled WHAT to analyze, but couldn't intervene DURING analysis.

With `interrupt()`, the graph PAUSES mid-execution and waits for human input:

```
classify → route to technical → diagnose → interrupt("I need your OS version")
                                                    ↓
                                            [USER TYPES: "Windows 11"]
                                                    ↓
                                            suggest_fix → draft_response → END
```

The graph literally stops, saves its state (thanks to checkpointing), and resumes when the human responds. This is powerful because:
- The agent can ask clarifying questions mid-flow
- A human supervisor can approve/reject actions before they happen
- Long-running workflows can wait for external input

How it works in code:

```python
from langgraph.types import interrupt

def diagnose(state):
    # ... run diagnostics ...
    if need_more_info:
        # This PAUSES the graph and returns to the caller
        user_answer = interrupt("What operating system are you using?")
        # When the graph resumes, user_answer has the human's response
        return {"os_info": user_answer}
```

The key insight: `interrupt()` requires a CHECKPOINTER (like MemorySaver from P4). The state is saved when the graph pauses and restored when it resumes. Without checkpointing, the graph couldn't "remember" where it was.

Compare to other frameworks:
- CrewAI: has `human_input=True` on agents, but less granular (whole agent pauses, not a specific point)
- AG2: has `human_input_mode` on agents (similar to CrewAI)
- LangGraph: most flexible — you choose EXACTLY where to pause

#### 3. Streaming Responses

In P1-P4, we used `app.invoke()` which waits for the ENTIRE graph to finish, then returns the result. The user sees nothing until it's done.

With streaming, the user sees output AS IT'S GENERATED:

```python
# Before (P1-P4): wait for everything
result = app.invoke(state)
print(result["messages"][-1].content)  # all at once

# Now (P5): stream tokens as they arrive
for chunk in app.stream(state):
    print(chunk, end="", flush=True)  # appears character by character
```

LangGraph supports multiple streaming modes:
- `stream_mode="values"` — streams complete state after each node
- `stream_mode="updates"` — streams only the changes each node makes
- `stream_mode="messages"` — streams individual LLM tokens (what you want for chat UX)

This matters for user experience. A support bot that sits silent for 10 seconds feels broken. One that starts typing immediately feels responsive.

### The Graph Shape

```
                    ┌──────────────────────┐
                    │    Parent Graph      │
                    │                      │
START → classify ──→│→ billing_subgraph   │──→ collect_feedback → END
                    │→ technical_subgraph  │
                    │→ returns_subgraph    │
                    │                      │
                    └──────────────────────┘
                              │
                    (each sub-graph can interrupt()
                     to ask the user for more info)
```

### Build Steps

1. **State definition** — `SupportState` with customer issue, category, department-specific fields
2. **Classifier node** — LLM classifies the issue into a department
3. **Three sub-graphs** — billing, technical, returns (each with 2-3 nodes and their own flows)
4. **interrupt() integration** — technical sub-graph pauses to ask for system info
5. **Parent graph assembly** — conditional routing to sub-graphs
6. **Streaming** — stream the final response token by token
7. **Interactive loop** — chat-style interface (like P4 but with streaming output)

### Files to Create

```
src/langgraph_portfolio/projects/project_5_support_bot/
  langgraph_support_bot.py   — parent graph + classifier + streaming loop
  subgraphs/
    billing.py               — billing department sub-graph
    technical.py             — technical support sub-graph
    returns.py               — returns department sub-graph
  models.py                  — Pydantic models (IssueClassification, etc.)
```

---

## Project 6: Competitive Intelligence Crew (CrewAI — Advanced)

### What You're Building

A multi-crew intelligence system that:
1. **Research Crew**: agents scrape websites and gather raw data about competitors
2. **Analysis Crew**: agents process the raw data — compare pricing, features, market position
3. **Report Crew**: agents synthesize everything into an executive briefing

These three crews are chained together using CrewAI **Flows** — a pipeline that connects multiple crews into a larger workflow.

### Why This Project

In P3, you built a single Crew with 5 agents. That's fine for one task, but real applications need multiple crews working in sequence or parallel — a research crew feeds an analysis crew feeds a report crew. CrewAI Flows is the framework for composing crews, just like LangGraph sub-graphs compose graphs.

### New Concepts

#### 1. CrewAI Flows (Multi-Crew Pipelines)

In P3, you had ONE crew that did everything:

```python
crew = Crew(agents=[planner, researcher, writer, editor, publisher], ...)
result = crew.kickoff(inputs={"topic": "..."})
```

But what if the research phase and the analysis phase need DIFFERENT crews with different agents, tools, and processes? You can't easily mix sequential research with hierarchical analysis in one crew.

CrewAI Flows solve this by chaining crews:

```python
from crewai.flow.flow import Flow, start, listen

class IntelligenceFlow(Flow):
    @start()
    def research_phase(self):
        # Run the research crew
        result = research_crew.kickoff(inputs=self.state)
        return result

    @listen(research_phase)
    def analysis_phase(self, research_result):
        # Research is done → run the analysis crew
        result = analysis_crew.kickoff(inputs={"raw_data": research_result})
        return result

    @listen(analysis_phase)
    def report_phase(self, analysis_result):
        # Analysis is done → run the report crew
        result = report_crew.kickoff(inputs={"analysis": analysis_result})
        return result
```

The `@start()` decorator marks the entry point. `@listen(previous_step)` says "run this after that step completes." It's declarative — you describe the pipeline, CrewAI handles the execution.

Compare to LangGraph:
- LangGraph: you wire edges manually (`graph.add_edge("research", "analysis")`)
- CrewAI Flows: you use decorators (`@listen(research_phase)`)
- Same idea (step A → step B), different syntax and philosophy

Flows also support:
- **Conditional routing** with `@router()` — like LangGraph's conditional edges
- **Parallel execution** — multiple `@listen()` on the same source
- **State management** — a shared `FlowState` across all steps

#### 2. CrewAI Custom Tools

In P3, agents had no tools — they relied on LLM knowledge. In P2 (LangGraph), we built a search tool with `@tool`. CrewAI has its own tool system.

CrewAI tools are classes that extend `BaseTool`:

```python
from crewai.tools import BaseTool

class WebScraperTool(BaseTool):
    name: str = "Web Scraper"
    description: str = "Scrapes a website URL and returns the text content"

    def _run(self, url: str) -> str:
        # Scrape the URL and return content
        ...
```

Then you assign tools to agents:

```python
researcher = Agent(
    role="Web Researcher",
    tools=[WebScraperTool()],  # agent can use this tool
    ...
)
```

Compare to LangGraph:
- LangGraph: tools are `@tool` decorated functions, bound with `llm.bind_tools()`
- CrewAI: tools are classes assigned to agents via the `tools` parameter
- LangGraph gives you more control (you decide when tools run via ToolNode)
- CrewAI is more declarative (the agent decides when to use its tools)

#### 3. CrewAI Memory

In P1-P4, every run started fresh — no memory of previous conversations. CrewAI has built-in memory types:

- **Short-term memory**: within a single crew execution (agents remember what happened earlier)
- **Long-term memory**: across multiple runs (the crew learns from past executions)
- **Entity memory**: remembers facts about specific entities (companies, people, products)

```python
crew = Crew(
    agents=[...],
    tasks=[...],
    memory=True,  # enables short-term + long-term memory
)
```

This means the intelligence crew can LEARN from previous research runs. If you researched Company X last week, the crew remembers key facts. Compare to LangGraph where you'd need to build this yourself (save to a database, load on startup).

### The Flow Shape

```
┌─────────────────────────────────────────────────────┐
│                 Intelligence Flow                     │
│                                                       │
│  ┌──────────────┐   ┌──────────────┐   ┌───────────┐│
│  │ Research Crew │──→│ Analysis Crew│──→│Report Crew││
│  │              │   │              │   │           ││
│  │ • Scraper    │   │ • Analyst    │   │ • Writer  ││
│  │ • Gatherer   │   │ • Comparator │   │ • Editor  ││
│  │              │   │ • Strategist │   │           ││
│  └──────────────┘   └──────────────┘   └───────────┘│
│                                                       │
│  Flow state passes between crews automatically        │
└─────────────────────────────────────────────────────┘
```

### Build Steps

1. **Research Crew** — 2 agents (scraper + gatherer) with WebScraperTool
2. **Analysis Crew** — 3 agents (analyst, comparator, strategist) in hierarchical mode
3. **Report Crew** — 2 agents (writer, editor) with structured output (Pydantic)
4. **Flow assembly** — chain the three crews with `@start()` and `@listen()`
5. **Memory setup** — enable long-term memory so repeat runs are smarter
6. **CLI interface** — ask about a company, get an intelligence report

### Files to Create

```
src/langgraph_portfolio/projects/project_6_intel_crew/
  flow.py            — the CrewAI Flow connecting all three crews
  research_crew.py   — research crew with scraping tools
  analysis_crew.py   — analysis crew (hierarchical)
  report_crew.py     — report crew with structured output
  tools.py           — WebScraperTool and other custom tools
  models.py          — Pydantic models for structured output
```

### Known Issues

1. **Memory requires Ollama embedder config** — `memory=True` defaults to OpenAI's `gpt-4o-mini` for memory analysis/embedding, which fails without `OPENAI_API_KEY`. Fix: pass `embedder={"provider": "ollama", "config": {"model": "nomic-embed-text"}}` to each `Crew(...)`, or set `memory=False` to skip memory. Affects all three crews.

2. **`output_pydantic` fails with Ollama via Instructor** — CrewAI uses the Instructor library to force structured output from tasks with `output_pydantic=<Model>`. Instructor expects the LLM to return a single tool call, but Ollama/qwen3.5 returns `tool_calls=None` (empty content), causing `AssertionError: Instructor does not support multiple tool calls`. This breaks the `profile_task` (CompetitorProfile), `strategy_task` (CompetitiveAnalysis), and `edit_task` (IntelligenceReport). Fix options:
   - **Option A:** Remove `output_pydantic` from tasks and parse the LLM's text output manually in the flow (same workaround pattern as P3's Ollama constraints — control structured output from Python, not through agents).
   - **Option B:** Use `output_json` instead of `output_pydantic` (may have the same Instructor issue).
   - **Option C:** Add explicit JSON formatting instructions in the task description and parse `result.raw` with Pydantic in the flow step.

---

## Project 7: Multi-Agent Debate Arena (AG2)

### What You're Building

A debate system where:
1. Two or more AI agents argue different sides of a topic
2. A moderator agent controls turn-taking and keeps the debate on track
3. A judge agent scores arguments and declares a winner
4. The debate runs as a GROUP CHAT where agents respond to each other's messages

### Why This Project

AG2 (formerly AutoGen, by Microsoft) has a fundamentally different model from LangGraph and CrewAI. While LangGraph is about graphs and CrewAI is about crews, AG2 is about **conversations between agents**. Agents send messages to each other, and a group chat manager decides who speaks next.

This is the most natural model for a debate — agents literally talk to each other.

### New Concepts

#### 1. AG2 Conversational Agents

In LangGraph, "agents" are functions. In CrewAI, agents are objects with roles. In AG2, agents are **conversational participants** — they have a name, a system message, and they respond to messages from other agents.

```python
from ag2 import ConversableAgent

debater_pro = ConversableAgent(
    name="Debater_Pro",
    system_message="You argue IN FAVOR of the topic. Be persuasive and use evidence.",
    llm_config={"model": "ollama/qwen3.5:35b"},
)

debater_con = ConversableAgent(
    name="Debater_Con",
    system_message="You argue AGAINST the topic. Be critical and find weaknesses.",
    llm_config={"model": "ollama/qwen3.5:35b"},
)
```

The key difference: in LangGraph, you write the flow (who talks when). In AG2, agents talk TO EACH OTHER — the framework manages the conversation.

```python
# Two agents having a conversation — AG2 manages the back-and-forth
debater_pro.initiate_chat(
    debater_con,
    message="I believe AI will create more jobs than it destroys. Here's why..."
)
```

AG2 handles: sending messages, collecting responses, managing turns, and stopping when a termination condition is met.

#### 2. Group Chat + Group Chat Manager

A two-agent chat is simple. But a debate needs a MODERATOR. AG2's GroupChat puts multiple agents in a shared conversation, and a GroupChatManager controls who speaks next:

```python
from ag2 import GroupChat, GroupChatManager

group_chat = GroupChat(
    agents=[debater_pro, debater_con, moderator, judge],
    messages=[],
    max_round=10,  # max turns (like P3's MAX_REVISIONS, but for the whole debate)
)

manager = GroupChatManager(
    groupchat=group_chat,
    llm_config={"model": "ollama/qwen3.5:35b"},
)
```

The manager is like CrewAI's hierarchical manager — it's an LLM that decides who should speak next. But instead of delegating TASKS, it's managing a CONVERSATION.

Compare across frameworks:
- LangGraph: YOU decide the order (graph edges)
- CrewAI hierarchical: a MANAGER agent delegates tasks
- AG2 GroupChat: a MANAGER agent selects the next speaker in a conversation

#### 3. Speaker Selection Strategies

The GroupChatManager can pick the next speaker in different ways:

- **auto** (default): the manager LLM decides based on the conversation ("the pro debater just spoke, the con debater should respond")
- **round_robin**: fixed rotation (pro → con → pro → con → ...)
- **random**: random selection
- **manual**: the human picks who speaks next
- **custom function**: you write a function that selects the speaker

This is a spectrum of control:
- round_robin = fully deterministic (like LangGraph edges)
- auto = fully LLM-driven (like CrewAI hierarchical)
- custom function = you control the logic (like LangGraph conditional edges)

#### 4. Termination Conditions

How does the debate end? In LangGraph, you route to END explicitly. In AG2, you define termination conditions:

```python
debater_pro = ConversableAgent(
    name="Debater_Pro",
    is_termination_msg=lambda msg: "DEBATE_OVER" in msg["content"],
    ...
)
```

Or you set `max_round` on the GroupChat. Or the judge can send a special message that triggers termination. Multiple ways to stop — you pick what fits.

### The Conversation Shape

```
Round 1:  Moderator → "Topic: Should AI replace teachers?"
Round 2:  Debater_Pro → "AI can personalize learning..."
Round 3:  Debater_Con → "But human connection is irreplaceable..."
Round 4:  Moderator → "Good points. Let's go deeper on evidence..."
Round 5:  Debater_Pro → "Studies show that..."
Round 6:  Debater_Con → "Those studies have limitations..."
...
Round 9:  Moderator → "Final statements, please."
Round 10: Judge → "Scoring... Pro: 7/10, Con: 8/10. Winner: Con."
```

No graph. No tasks. Just agents talking — AG2's core model.

### Build Steps

1. **Setup AG2** — install ag2, configure with Ollama
2. **Create debater agents** — pro and con with different system prompts
3. **Create moderator** — manages the debate flow, keeps things on topic
4. **Create judge** — scores arguments, declares winner
5. **GroupChat assembly** — put all agents together with speaker selection
6. **Termination logic** — end after the judge scores
7. **CLI interface** — user provides a topic, watches the debate unfold

### Files to Create

```
src/langgraph_portfolio/projects/project_7_debate_arena/
  debate.py          — main AG2 group chat setup and execution
  agents.py          — agent definitions (debaters, moderator, judge)
  models.py          — Pydantic models for judge scoring
```

---

## Project 8: Research Agent v2 (BeeAI)

### What You're Building

A research agent (similar to P2) but built with BeeAI, IBM's framework. The agent:
1. Takes a research question
2. Uses multiple tools (search, calculator, code interpreter)
3. Iterates until it has enough information
4. Produces a cited research report

Same problem as P2, but you'll see how BeeAI approaches agent design differently — with a strong focus on observability, structured tool use, and event-driven architecture.

### Why This Project

You built a research agent with LangGraph (P2). Now you build the SAME thing with BeeAI. This gives you a direct comparison: same problem, different framework. You'll see what BeeAI makes easier, harder, and different.

BeeAI's philosophy:
- **Observability first** — every agent action is logged, traced, and inspectable
- **Tool-centric** — tools are first-class citizens with schemas, validation, and error handling
- **Event-driven** — agents emit events (thinking, tool_call, tool_result, final_answer) that you can hook into

### New Concepts

#### 1. BeeAI Agent Architecture

BeeAI agents are built around a **ReAct loop** (Reason + Act):

```
Think → Act → Observe → Think → Act → Observe → ... → Answer
```

This is similar to P2's agent loop, but BeeAI structures it explicitly:

```python
from beeai_framework.agents.react import ReActAgent
from beeai_framework.backend import ChatModel, UserMessage
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool

llm = ChatModel.from_name("ollama:qwen3.5:35b")

agent = ReActAgent(
    llm=llm,
    tools=[DuckDuckGoSearchTool()],
    memory=UnconstrainedMemory(),
)

result = await agent.run(prompt=UserMessage("What factors affect student math performance?"))
```

Note: `pip install beeai-framework` (requires Python >= 3.11). The top-level module is `beeai_framework` (underscores).
Available agent classes: `ReActAgent` (ReAct loop), `RequirementAgent` (declarative rules, recommended for production), `LiteAgent` (minimal).

Compare:
- LangGraph: you BUILD the loop (agent node → tools node → back to agent)
- BeeAI: the loop is BUILT-IN (the agent reasons, acts, and loops automatically)
- CrewAI: similar to BeeAI (built-in loop), but less visibility into the process

#### 2. Observability and Event System

This is BeeAI's killer feature. Every agent action emits an EVENT that you can listen to:

```python
from beeai_framework.agents.react import ReActAgent

# Hook into ALL events with a wildcard pattern
@agent.emitter.on("*.*")
async def handle_event(data, event):
    if event.name == "update":
        # data contains parsed values: thought, tool_name, tool_input, tool_output, final_answer
        print(f"Agent update: {data}")

# Or hook into specific events
@agent.emitter.on("success")
async def on_success(data, event):
    print(f"Agent finished: {data}")
```

Event types: `start`, `update` (state changes with parsed values), `partialUpdate` (streaming), `success`, `error`, `retry`, `final_answer`.

You get a real-time stream of what the agent is doing and WHY. Compare:
- LangGraph: you see state changes after each node (coarse-grained)
- CrewAI: verbose mode shows delegation decisions (medium-grained)
- BeeAI: event system shows EVERY thought and action (fine-grained)

This matters for debugging. When an agent gives a wrong answer, you need to know: did it search for the wrong thing? Did it misinterpret a result? Did it reason incorrectly? BeeAI's events tell you exactly where things went wrong.

#### 3. Structured Tool Definitions

BeeAI tools have explicit input/output schemas via the `Tool` base class:

```python
from beeai_framework.tools import Tool, StringToolOutput
from pydantic import BaseModel, Field

class CalculatorInput(BaseModel):
    expression: str = Field(description="Math expression to evaluate")

class CalculatorTool(Tool[CalculatorInput, None, StringToolOutput]):
    name = "Calculator"
    description = "Evaluates a mathematical expression"
    input_schema = CalculatorInput

    async def _run(self, input: CalculatorInput, options=None):
        result = eval(input.expression)  # simplified
        return StringToolOutput(f"Result: {result}")
```

Compare to LangGraph's `@tool` decorator:
- LangGraph/LangChain: `@tool` infers schema from the function signature
- BeeAI: explicit Pydantic schema definition (more verbose, but clearer, with type safety)

Both approaches work. BeeAI's is more explicit — you always know exactly what the tool expects and returns.

### The Agent Loop

```
User: "What factors affect student math grades?"
  │
  ├→ 🧠 Think: "I need to analyze the student dataset"
  ├→ 🔧 Act: PythonTool("df.corr()['G3'].sort_values()")
  ├→ 👁 Observe: "Highest correlations: G2 (0.91), G1 (0.80), failures (-0.36)..."
  ├→ 🧠 Think: "Previous grades matter most. Let me check study time."
  ├→ 🔧 Act: PythonTool("df.groupby('studytime')['G3'].mean()")
  ├→ 👁 Observe: "Study time 1: 9.2, 2: 10.3, 3: 11.4, 4: 11.8"
  ├→ 🧠 Think: "Study time helps but isn't the strongest factor. I have enough."
  └→ ✅ Answer: "The strongest predictors of math grades are..."
```

Same loop as P2, but with full observability at each step.

### Build Steps

1. **Setup BeeAI** — `poetry add beeai-framework` (requires Python >= 3.11), configure with Ollama via `ChatModel.from_name("ollama:qwen3.5:35b")`
2. **Define tools** — custom `Tool` subclasses for data analysis and knowledge base search
3. **Create agent** — `ReActAgent` (or `RequirementAgent`) with tools and `UnconstrainedMemory`
4. **Event handlers** — hook into `agent.emitter` events (`update`, `success`, `error`) for logging
5. **Run against the student CSV** — same dataset as P4 for comparison
6. **Compare to P2** — document the differences in control, observability, code size

### Files to Create

```
src/langgraph_portfolio/projects/project_8_beeai_research/
  agent.py           — BeeAI agent setup and execution
  tools.py           — custom tools (data analysis, search)
  events.py          — event handlers for observability logging
```

---

## Project 9: RAG Pipeline (LlamaIndex Workflows)

### What You're Building

A Retrieval-Augmented Generation (RAG) pipeline that:
1. Ingests documents (PDFs, text files, web pages)
2. Builds a searchable index (embeddings + vector store)
3. Takes user questions and retrieves relevant passages
4. Synthesizes an answer from the retrieved context
5. Uses different retrieval strategies (keyword, semantic, hybrid)

All built using LlamaIndex Workflows — an EVENT-DRIVEN architecture that's fundamentally different from LangGraph's graph-based and CrewAI's crew-based approaches.

### Why This Project

You've built RAG before (P2 used a simple knowledge base). But P2's RAG was basic — a list of documents with keyword matching. Real RAG needs embeddings, vector stores, chunking strategies, and retrieval ranking.

LlamaIndex is the leading framework for RAG. Its Workflow system provides a new mental model: instead of graphs or crews, you think in EVENTS.

### New Concepts

#### 1. Event-Driven Architecture

In LangGraph, nodes connect via EDGES:
```
node_A --edge--> node_B --edge--> node_C
```

In LlamaIndex Workflows, steps connect via EVENTS:
```
step_A emits EventX → step_B listens for EventX → step_B emits EventY → step_C listens for EventY
```

```python
from llama_index.core.workflow import Workflow, StartEvent, StopEvent, step, Event

class RetrievalDone(Event):
    """Fired when document retrieval is complete."""
    documents: list[str]
    query: str

class RAGWorkflow(Workflow):
    @step
    async def retrieve(self, ev: StartEvent) -> RetrievalDone:
        # Search the index
        docs = self.index.search(ev.query)
        return RetrievalDone(documents=docs, query=ev.query)

    @step
    async def synthesize(self, ev: RetrievalDone) -> StopEvent:
        # Generate answer from retrieved docs
        answer = self.llm.generate(ev.documents, ev.query)
        return StopEvent(result=answer)
```

The key difference:
- LangGraph: steps are EXPLICITLY connected (you draw the edge)
- LlamaIndex: steps are IMPLICITLY connected (step B runs because step A emitted an event B listens for)

This is like the difference between:
- Calling a function directly: `result = process(data)` → LangGraph
- Firing an event: `emit("data_ready", data)` and whoever listens handles it → LlamaIndex

Event-driven is more DECOUPLED — steps don't know about each other, they just know about events. This makes it easy to add new steps without changing existing ones.

#### 2. Document Ingestion and Chunking

Raw documents (PDFs, web pages) are too large to pass to an LLM. They need to be:
1. **Loaded** — read the file/URL
2. **Chunked** — split into smaller pieces (paragraphs, sections)
3. **Embedded** — convert each chunk to a vector (numerical representation)
4. **Indexed** — store vectors in a searchable structure

LlamaIndex provides all of this out of the box:

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# Load documents from a folder
documents = SimpleDirectoryReader("./data").load_data()

# Build index (chunks + embeds + stores automatically)
index = VectorStoreIndex.from_documents(documents)
```

Compare to P2 where we manually searched through a list of document objects. LlamaIndex handles the entire pipeline — chunking strategy, embedding model, vector storage.

#### 3. Retrieval Strategies

Not all searches are created equal:

- **Keyword** — traditional text matching (like P2's search)
- **Semantic** — embedding similarity (finds conceptually related passages even if words differ)
- **Hybrid** — combines keyword + semantic for best of both
- **Reranking** — retrieve many, then rerank by relevance

LlamaIndex lets you swap strategies easily:

```python
# Semantic search (default)
query_engine = index.as_query_engine()

# Hybrid search
query_engine = index.as_query_engine(
    similarity_top_k=10,       # get 10 semantic matches
    sparse_top_k=10,           # get 10 keyword matches
    mode="hybrid",             # combine them
)
```

This project will implement multiple strategies and let you compare their results on the same questions.

### The Workflow Shape

```
StartEvent(query="...")
     │
     ├→ retrieve (searches the index)
     │      │
     │      └→ RetrievalDone(documents=[...])
     │              │
     │              ├→ rerank (optional: rerank by relevance)
     │              │      │
     │              │      └→ RerankDone(documents=[...])
     │              │              │
     └──────────────┴──────────────┘
                    │
                    └→ synthesize (LLM generates answer from docs)
                           │
                           └→ StopEvent(result="The answer is...")
```

Events flow through the system. Each step does ONE thing and emits an event for the next step. You can insert new steps (like reranking) just by adding a new event listener — no graph rewiring needed.

### Build Steps

1. **Setup LlamaIndex** — install llama-index, configure with Ollama embeddings
2. **Document ingestion** — load sample documents (could reuse the student dataset docs or add PDFs)
3. **Index building** — VectorStoreIndex with chunking
4. **Workflow definition** — RAGWorkflow with retrieve + synthesize steps
5. **Multiple retrieval strategies** — keyword, semantic, hybrid
6. **Comparison mode** — run the same question with different strategies, compare results
7. **Interactive loop** — ask questions, see retrieved passages + synthesized answer

### Files to Create

```
src/langgraph_portfolio/projects/project_9_rag_pipeline/
  workflow.py        — LlamaIndex Workflow (event-driven RAG)
  ingestion.py       — document loading and index building
  strategies.py      — different retrieval strategies
  models.py          — Pydantic models for results
```

---

## Project 10: Framework Showdown (Capstone)

### What You're Building

ONE problem solved with ALL frameworks — a side-by-side comparison:

**The problem**: An AI assistant that takes a research question, searches for information, analyzes it, and produces a structured report with citations.

**The implementations**:
1. LangGraph version (graph-based)
2. CrewAI version (crew-based)
3. AG2 version (conversation-based)
4. BeeAI version (ReAct loop)
5. LlamaIndex Workflows version (event-driven)

Same input, same output format, different orchestration. This is the ultimate "when to use what" study.

### Why This Project

You'll have used each framework individually by now. But the real understanding comes from COMPARING them on the same problem. This project forces you to think about:
- Which framework makes this task easiest?
- Which gives the most control?
- Which is most debuggable?
- Which produces the best output?
- Which has the least boilerplate?

### The Comparison Dimensions

#### 1. Lines of Code

How much code does each implementation need? Expect:
- LangGraph: most code (explicit graph + state + routing)
- CrewAI: least code (declarative agents + tasks)
- AG2: medium (agents + group chat setup)
- BeeAI: medium (agent + tools + events)
- LlamaIndex: medium (workflow + steps + events)

#### 2. Control Level

How much do YOU control vs the framework?
- LangGraph: maximum control (you design every path)
- BeeAI: high control (you define tools, hook into events)
- LlamaIndex: medium control (event system is flexible, but retrieval is abstracted)
- AG2: medium control (speaker selection, but conversation flow is emergent)
- CrewAI: least control (the manager decides)

#### 3. Debuggability

When something goes wrong, how easy is it to find the problem?
- BeeAI: best (event system shows every thought and action)
- LangGraph: good (explicit state, you can inspect after each node)
- LlamaIndex: good (events are traceable)
- AG2: medium (conversation log helps, but group chat decisions are opaque)
- CrewAI: challenging (hierarchical manager's decisions are hard to trace)

#### 4. Output Quality

Does the framework's approach affect the quality of the result?
This depends on the task. We'll measure by running the same 5 questions through all implementations and comparing the reports.

### The Structure

```
src/langgraph_portfolio/projects/project_10_showdown/
  problem.py              — shared problem definition + evaluation criteria
  langgraph_version.py    — LangGraph implementation
  crewai_version.py       — CrewAI implementation
  ag2_version.py          — AG2 implementation
  beeai_version.py        — BeeAI implementation
  llamaindex_version.py   — LlamaIndex Workflows implementation
  compare.py              — runs all implementations, compares results
  report.py               — generates the comparison report
```

### Build Steps

1. **Define the problem** — research assistant spec, input format, expected output format
2. **Implement each version** — reuse patterns from P2-P9
3. **Create evaluation criteria** — what makes a good research report?
4. **Build the comparator** — runs all 5, measures time, code size, output quality
5. **Generate comparison report** — markdown report with side-by-side analysis

### The Key Insight

After building this, you'll be able to answer the question every engineering team asks: **"Which agent framework should we use?"**

Your answer won't be "use X because it's popular." It will be: "For THIS type of problem, use X because of Y, and here's the evidence from my comparison."

That's the real value of this portfolio — not just knowing how to USE the frameworks, but knowing WHEN to use each one and WHY.

---

## Implementation Order

| Phase | Projects | Timeline |
|-------|----------|----------|
| Foundation (DONE) | P1, P2, P3, P4 | Completed |
| LangGraph Advanced (DONE) | P5 (support bot) | Completed |
| CrewAI Advanced (DONE) | P6 (intel crew) | Completed |
| New Frameworks (DONE) | P7 (AG2), P8 (BeeAI), P9 (LlamaIndex) | Completed |
| Capstone | P10 (showdown) | Next |
| MCP Protocol | P11 (job search MCP server) | Independent — can be done anytime |

P7, P8, and P9 are independent of each other — you can tackle them in any order based on interest. P10 must come last since it compares all frameworks. P11 is fully independent — it teaches MCP, not agentic graph frameworks — and can be built at any point.

## Dependencies to Add

```toml
# Already have:
langgraph = "^0.2.0"
crewai = {version = "^1.11.0", extras = ["litellm"]}

# To add for new projects:
ag2 = "^0.6"                    # P7 — formerly autogen
bee-agent-framework = "^0.1"    # P8 — IBM's BeeAI (check latest version)
llama-index = "^0.12.0"         # P9 — already in pyproject.toml
mcp = "^1.0"                    # P11 — MCP Python SDK (FastMCP)
```

---

## Project 11: Job Search MCP Server (MCP — Protocol)

### What You're Building

An MCP (Model Context Protocol) server that exposes job search data through the three MCP primitives:
1. **Resource** (`jobs://listings`) — read-only access to job listings
2. **Tool** (`mark_as_applied`) — narrowly-scoped write action to mark a job as applied
3. **Prompt** (`cover_letter`) — reusable prompt template for generating cover letters in Portuguese

The server enforces **least-privilege boundaries**: the tool can ONLY mark one job as applied (no deletes, no bulk updates, no arbitrary SQL). The resource is strictly read-only. The prompt has no data access at all.

### Why This Project

Projects 1-10 teach agentic AI frameworks (how agents think and collaborate). Project 11 teaches the **protocol layer** — how AI clients (Claude Code, Claude Desktop, custom apps) connect to external services. MCP is to AI what REST is to web apps: the standard interface. Understanding MCP completes the picture from "how agents work" to "how agents connect to the world."

### New Concepts

#### 1. MCP Protocol (stdio + JSON-RPC)

MCP uses JSON-RPC messages over stdio (standard input/output). The AI client runs the MCP server as a subprocess and communicates via stdin/stdout. No HTTP, no ports, no networking.

This is fundamentally different from a REST API:
- REST: client sends HTTP request → server returns HTTP response
- MCP: client spawns server process → exchanges JSON-RPC messages over stdio

The `mcp` Python SDK (`FastMCP`) handles all the protocol plumbing — you just decorate functions.

#### 2. MCP Primitives — Resources, Tools, Prompts

Three primitives, each with a specific purpose:

| Primitive | Direction | Purpose | Analogy |
|-----------|-----------|---------|---------|
| **Resource** | Server → Client | Client pulls read-only data | GET endpoint |
| **Tool** | Client → Server | Client asks server to do something | POST endpoint |
| **Prompt** | Server → Client | Server offers reusable instructions | Stored procedure |

The key insight: resources and prompts are **read-only** (the client pulls data or templates). Only tools can **write** (modify state). This separation is the foundation of least-privilege design.

#### 3. Least-Privilege Tool Boundaries

The tool exposes `mark_as_applied(job_id, notes)`, NOT `execute_sql(query)`. This is least privilege:
- **Narrow scope**: one specific action, not a general-purpose operation
- **Server-side validation**: every input checked (job exists? already applied? notes too long?)
- **No escape hatches**: the tool literally cannot delete data or modify listings

#### 4. Server-Side Input Validation

Even though the MCP schema defines parameter types, the server validates EVERYTHING:
- `job_id`: must be positive integer, must exist in database
- `notes`: max 500 characters, whitespace-trimmed
- Duplicate check: can't apply to the same job twice

### Architecture

**Data store**: SQLite (`jobs.db`) with two tables:
- `jobs` — listings (title, company, location, platform, score, status)
- `applications` — application log (job_id, applied_at, notes)

**Security boundaries**:
- Resource: `SELECT` on `jobs` only. No writes, no `applications` access.
- Tool: `INSERT` into `applications` + `UPDATE jobs.status` for ONE `job_id`. No deletes, no bulk ops.
- Prompt: No database access. Pure text template.

**Transport**: stdio (default for Claude Code integration).

### The Three Primitives

**Resource — `jobs://listings`**:
```python
@server.resource("jobs://listings")
async def get_listings() -> str:
    """Read-only. Returns all job listings as JSON."""
```

**Tool — `mark_as_applied`**:
```python
@server.tool()
async def mark_as_applied(job_id: int, notes: str = "") -> str:
    """Mark ONE job as applied. Validates input, checks duplicates."""
```

**Prompt — `cover_letter`**:
```python
@server.prompt()
async def cover_letter(job_title: str, company: str) -> list[Message]:
    """Cover letter template in Brazilian Portuguese."""
```

### Build Steps

1. **Database setup** — SQLite schema, seed data, query functions
2. **Pydantic models** — `JobListing`, `ApplicationRecord` for structured data
3. **Input validation** — validation helpers with clear error messages
4. **MCP server** — `FastMCP` with resource, tool, and prompt decorators
5. **Security boundaries writeup** — what it can/can't do
6. **Tests** — smoke, validation (invalid inputs), security (boundary enforcement)
7. **Claude Code integration** — register with `claude mcp add`, test live

### Files to Create

```
src/langgraph_portfolio/projects/project_11_mcp_server/
    server.py          — FastMCP server (resource + tool + prompt)
    db.py              — SQLite setup, seed data, query functions
    validation.py      — Input validation helpers
    models.py          — Pydantic models for job data
    main.py            — entry point

project_11_mcp_server/
    __init__.py
    main.py            — thin wrapper
    tests/
        __init__.py
        test_smoke.py       — server initializes, DB creates tables
        test_validation.py  — invalid inputs rejected properly
        test_security.py    — boundary enforcement
    security_boundaries.md  — writeup of what the server can/can't do
```
