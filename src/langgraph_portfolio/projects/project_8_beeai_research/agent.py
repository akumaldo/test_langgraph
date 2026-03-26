"""
Project 8 — BeeAI Research Agent.

---------------------------------------------------------------------------
THE BIG PICTURE: LangGraph vs BeeAI
---------------------------------------------------------------------------

In P2 (LangGraph), building a research agent required wiring 5 things:

    1. State (TypedDict with messages, documents, citations)
    2. Tool (function with @tool decorator)
    3. Agent node (function that calls llm.bind_tools().invoke())
    4. Tool node (ToolNode that executes tool calls)
    5. Graph (StateGraph with nodes, edges, conditional routing)

You built the loop YOURSELF:
    agent → should_continue() → "tools" → agent → should_continue() → END

In BeeAI, you configure 3 things:

    1. Tools (classes that subclass Tool)
    2. Agent (ReActAgent with LLM, tools, memory)
    3. Event handlers (optional, for observability)

The loop is BUILT IN. The ReActAgent does Think → Act → Observe
automatically until it reaches a final answer. You don't wire edges
or routing — BeeAI handles that.

TRADEOFF:
  - LangGraph: more work to set up, but you control EVERYTHING
    (you could add custom logic between steps, short-circuit the loop,
    branch into sub-graphs, etc.)
  - BeeAI: less setup, built-in ReAct loop, but less control
    (you can't easily change the loop structure itself — it's fixed
    as Think → Act → Observe → repeat)

For a standard research agent, BeeAI is simpler. For complex multi-step
workflows with branching, LangGraph gives you more power.
---------------------------------------------------------------------------
"""

import asyncio

from beeai_framework.agents.react import ReActAgent
from beeai_framework.backend import ChatModel
from beeai_framework.memory import UnconstrainedMemory

from .tools import SearchDocumentsTool, CalculatorTool, create_knowledge_base
from .events import attach_event_handlers


# ---------------------------------------------------------------------------
# 1. LLM SETUP
#
# BeeAI uses ChatModel.from_name() to connect to any LLM backend.
# The format is "provider:model_name".
#
# Compare to LangGraph (P2):
#   LangGraph:  ChatOllama(model="qwen3.5:35b", temperature=0)
#   BeeAI:      ChatModel.from_name("ollama:qwen3.5:35b")
#
# Same Ollama backend, different wrapper. BeeAI's approach is more
# generic — you can switch to "openai:gpt-4" or "watsonx:granite"
# just by changing the string. LangGraph uses different classes for
# each provider (ChatOllama, ChatOpenAI, etc.).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 2. AGENT CREATION
#
# ReActAgent is BeeAI's implementation of the ReAct pattern:
#   Think → Act → Observe → Think → Act → Observe → ... → Answer
#
# Constructor needs:
#   - llm:    the language model
#   - tools:  list of Tool instances (not functions — remember, BeeAI
#             tools are classes, not decorated functions)
#   - memory: how the agent tracks conversation history
#
# UnconstrainedMemory keeps ALL messages — no truncation. For a simple
# demo this is fine. In production, you'd use TokenMemory or
# SlidingMemory to cap context size (same concern as P6's Ollama
# context overflow).
#
# Optional params we're using:
#   - execution: AgentExecutionConfig with max_iterations
#     This is like our MAX_REVISIONS safety cap from P3 — prevents the
#     agent from looping forever.
# ---------------------------------------------------------------------------

def create_agent() -> ReActAgent:
    """Create and configure the BeeAI research agent.

    Returns a ReActAgent with:
      - Ollama LLM (same model as P2)
      - Search + Calculator tools
      - Unconstrained memory
      - Event handlers for observability
    """
    from beeai_framework.agents.types import AgentExecutionConfig

    # LLM — same Ollama model we've used throughout the portfolio
    llm = ChatModel.from_name("ollama:qwen3.5:35b")

    # Tools — create instances (BeeAI tools are objects, not functions)
    kb = create_knowledge_base()
    tools = [
        SearchDocumentsTool(kb),
        CalculatorTool(),
    ]

    # Memory — keep everything (fine for demo, watch out for long sessions)
    memory = UnconstrainedMemory()

    # Execution config — safety cap on iterations
    # Same idea as MAX_REVISIONS in P3: prevent infinite loops.
    # max_iterations=10 means the agent can do at most 10 Think→Act→Observe
    # cycles before BeeAI forces it to stop.
    execution = AgentExecutionConfig(
        max_iterations=10,
        max_retries_per_step=3,
        total_max_retries=10,
    )

    # Create the agent
    agent = ReActAgent(
        llm=llm,
        tools=tools,
        memory=memory,
        execution=execution,
    )

    # Attach event handlers — this is what gives us observability.
    # Without this, the agent would still work, but you'd only see
    # the final answer. With events, you see every step.
    attach_event_handlers(agent.emitter)

    return agent


# ---------------------------------------------------------------------------
# 3. RUNNING THE AGENT
#
# BeeAI agents are ASYNC. This is different from LangGraph (P2), where
# we called app.invoke() synchronously.
#
# Why async?
#   - Tool calls might involve network requests (API calls, web searches)
#   - Multiple tools could theoretically run in parallel
#   - Event handlers run as async callbacks
#
# The agent.run() method returns a Run object. We use `await` with
# .result() or iterate with `async for` to get the output.
#
# The input is just a string (the user's question). Compare:
#   LangGraph (P2):  app.invoke({"messages": [("user", question)], ...})
#   BeeAI:           await agent.run(question).result()
#
# BeeAI is simpler — you pass a string, it handles message formatting.
# LangGraph requires you to structure the initial state yourself.
# ---------------------------------------------------------------------------

async def run_research(question: str) -> str:
    """Run the research agent on a question and return the answer.

    This is the main entry point. It:
      1. Creates the agent
      2. Runs the ReAct loop (automatically!)
      3. Returns the final answer

    The ReAct loop looks like:
      🧠 Think: "I need to search for..."
      🔧 Act:   search_documents("LangGraph vs CrewAI")
      👁 Observe: "[Source: LangGraph Overview] LangGraph provides..."
      🧠 Think: "Now I know about LangGraph. Let me search CrewAI..."
      🔧 Act:   search_documents("CrewAI")
      👁 Observe: "[Source: CrewAI Overview] CrewAI focuses on..."
      🧠 Think: "I have enough information to answer."
      ✅ Answer: "LangGraph and CrewAI differ in..."
    """
    agent = create_agent()

    print(f"\n{'='*60}")
    print(f"🔬 Research Question: {question}")
    print(f"{'='*60}\n")

    # Run the agent — this starts the ReAct loop.
    # The agent will Think → Act → Observe until it has a final answer.
    # Event handlers (from events.py) will print each step in real time.
    #
    # agent.run() returns a Run object, which is AWAITABLE — you just
    # use `await` directly on it. No .result() or .invoke() needed.
    output = await agent.run(question)

    # The output contains:
    #   - output.output: list of final messages (AssistantMessage objects)
    #   - output.iterations: list of all ReAct iterations
    #   - output.memory: the conversation memory
    #   - output.usage: token usage statistics

    # output.output is a list of Message objects. Each message's .content
    # can be a string or a list of MessageTextContent parts. We extract
    # the text from the last message.
    if output.output:
        content = output.output[-1].content
        if isinstance(content, list):
            # List of MessageTextContent — join their text fields
            final_answer = "\n".join(part.text for part in content)
        else:
            final_answer = str(content)
    else:
        final_answer = "No answer produced."

    print(f"\n{'='*60}")
    print(f"📋 Final Answer:\n{final_answer}")
    print(f"{'='*60}")
    print(f"\n📊 Stats: {len(output.iterations)} iterations, "
          f"{output.usage.prompt_tokens} prompt tokens, "
          f"{output.usage.completion_tokens} completion tokens")

    return final_answer


# ---------------------------------------------------------------------------
# 4. DIRECT EXECUTION
#
# BeeAI is async, so we need asyncio.run() to start the event loop.
# In P2, we just called app.invoke() — synchronous and simple.
# The async requirement is a small inconvenience, but it's standard
# for modern Python frameworks.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(run_research("How do LangGraph, CrewAI, and LlamaIndex differ?"))
