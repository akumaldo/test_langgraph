"""
Project 2 — Research Agent using real LangGraph.

We build this in layers (same approach as the chatbot):
  1. State definition
  2. The search tool (new concept!)
  3. Node functions (agent + tool executor)
  4. Routing logic (the agent loop)
  5. Graph assembly
"""

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langgraph.graph import add_messages


# ---------------------------------------------------------------------------
# 1. STATE
#
# Same idea as the chatbot: a TypedDict that holds all the data flowing
# through the graph.
#
# What's different?
#
# In the chatbot, we had: messages, intent, confidence
# Here we have: messages, plus fields to track what the agent found.
#
# "messages" does double duty here — it holds:
#   - The user's original question
#   - The LLM's "thinking" (including tool call requests)
#   - Tool results (search results come back as messages too!)
#   - The final answer
#
# Everything flows through messages. This is how LangGraph's tool-calling
# loop works: the LLM says "call this tool" as a message, the tool node
# puts results back as a message, and the LLM reads those results.
#
# The extra fields (documents, citations) are for US — so we can inspect
# what the agent found and where it got its information, outside of the
# message stream.
# ---------------------------------------------------------------------------

class ResearchState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # conversation + tool calls
    documents: list[str]    # document contents found by searching
    citations: list[str]    # source references ("Title (source)")


# ---------------------------------------------------------------------------
# 2. THE SEARCH TOOL
#
# This is the new concept! In the chatbot, the LLM could only produce text.
# Now we give it a tool — a Python function it can ask to run.
#
# How the @tool decorator works:
#   - It reads the function NAME → becomes the tool name the LLM sees
#   - It reads the DOCSTRING → becomes the description the LLM uses to
#     decide when to call it
#   - It reads the PARAMETERS (type hints) → becomes the input schema
#
# So the LLM literally sees something like:
#   "You have a tool called 'search_documents'. It takes a 'query' (string).
#    It searches a knowledge base of documents about AI frameworks..."
#
# The LLM NEVER runs this function directly. It says:
#   "I'd like to call search_documents with query='LangGraph routing'"
# Then LangGraph intercepts that, runs the function, and passes results back.
#
# IMPORTANT: The docstring matters a LOT. It's the LLM's only guide for
# understanding when and how to use the tool. Be clear and specific.
# ---------------------------------------------------------------------------

# First, we set up our knowledge base with sample documents.
# In a real app, this would connect to a vector database, LlamaIndex, etc.
# For learning, we use the in-memory knowledge base from our core module.

from ...core.knowledge import Document, KnowledgeBase

_knowledge_base = KnowledgeBase()
_knowledge_base.index([
    Document(
        title="LangGraph Overview",
        content="LangGraph provides stateful orchestration, routing, and persistence for agentic workflows.",
        source="docs/langgraph.md",
    ),
    Document(
        title="CrewAI Overview",
        content="CrewAI focuses on role-based multi-agent coordination and task-oriented collaboration.",
        source="docs/crewai.md",
    ),
    Document(
        title="LlamaIndex Connectors",
        content="LlamaIndex provides data connectors, document loading, and retrieval pipelines for RAG systems.",
        source="docs/llamaindex.md",
    ),
])


@tool
def search_documents(query: str) -> str:
    """Search a knowledge base of documents about AI frameworks.

    Use this tool to find information about LangGraph, CrewAI, LlamaIndex,
    and other agentic AI frameworks. Pass a search query and get back
    matching document excerpts with their sources.
    """
    hits = _knowledge_base.search(query, limit=3)

    if not hits:
        return "No documents found matching your query."

    # Format results so the LLM can read them easily
    results = []
    for hit in hits:
        results.append(
            f"[Source: {hit.document.title} ({hit.document.source})] "
            f"{hit.document.content}"
        )
    return "\n\n".join(results)


# We collect all tools in a list — this gets passed to the LLM later
# so it knows what tools are available, and to the ToolNode so it knows
# how to execute them.
tools = [search_documents]


# ---------------------------------------------------------------------------
# 3. NODES
#
# In the chatbot we had 5 nodes (greeting, classifier, router, respond,
# clarify). The research agent has only 2, but they're more powerful:
#
#   a) "agent" — the LLM, with tools bound to it
#   b) "tools" — the tool executor
#
# Here's the mental model:
#
#   The AGENT node is like a person sitting at a desk. They read the
#   conversation so far, think about it, and then do ONE of two things:
#     - Say "I need to look something up" → produces a tool call message
#     - Say "I have enough info, here's my answer" → produces a text message
#
#   The TOOLS node is like a librarian. When the agent asks for a search,
#   the librarian runs it and brings back results. The agent then reads
#   those results and decides what to do next.
#
# How .bind_tools() works:
#
#   In the chatbot, we used the LLM directly: llm.invoke(messages)
#   Now we do: llm.bind_tools(tools).invoke(messages)
#
#   bind_tools() tells the LLM: "Hey, you have these tools available."
#   It modifies the LLM so that when it responds, it can EITHER:
#     - Return a normal text message (AIMessage with content)
#     - Return a tool call request (AIMessage with tool_calls field)
#
#   The LLM decides on its own which to do, based on the conversation.
#
# What is ToolNode?
#
#   LangGraph provides a built-in ToolNode class. You give it the list
#   of tool functions, and it knows how to:
#     - Read tool call requests from the last message
#     - Actually execute the matching Python function
#     - Package the results as a ToolMessage (which goes back into state)
#
#   So we don't have to write any tool execution logic ourselves.
# ---------------------------------------------------------------------------

from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama
from langgraph.prebuilt import ToolNode

# Set up the LLM with tools bound to it.
# Same ChatOllama as the chatbot, but now with .bind_tools().
llm = ChatOllama(model="qwen3.5:35b", temperature=0)
llm_with_tools = llm.bind_tools(tools)


def agent(state: ResearchState) -> dict:
    """The 'thinking' node — the LLM decides what to do.

    This is called every time the loop comes back to the agent.
    The LLM sees the full message history (including any previous
    tool results) and decides: search more, or give a final answer.
    """
    system_prompt = SystemMessage(content=(
        "You are a research assistant. Your job is to answer the user's "
        "question by searching through available documents.\n\n"
        "INSTRUCTIONS:\n"
        "1. Use the search_documents tool to find relevant information.\n"
        "2. You may search multiple times with different queries if needed.\n"
        "3. Once you have enough information, provide a clear answer with "
        "citations in the format [Source: title].\n"
        "4. If no relevant documents are found, say so honestly."
    ))

    # Invoke the LLM with the system prompt + full conversation history.
    # The LLM will EITHER return a text response OR a tool call request.
    # We don't control which — the LLM decides based on the conversation.
    response = llm_with_tools.invoke([system_prompt] + state["messages"])

    # Return the response as a message update.
    # The add_messages reducer will APPEND it to the messages list.
    # If it's a tool call, LangGraph will see that in the next routing step.
    return {"messages": [response]}


# The tool executor node — LangGraph's built-in.
# We just pass it our tools list and it handles everything.
tool_node = ToolNode(tools)


# ---------------------------------------------------------------------------
# 4. ROUTING — THE AGENT LOOP
#
# In the chatbot, our router checked confidence:
#   confidence >= 0.7  →  respond
#   otherwise          →  ask_clarification
#
# Here, our router checks something different: did the LLM's last message
# contain a tool call?
#
#   tool call present  →  go to "tools" node (execute the search)
#   no tool call       →  go to END (the LLM gave its final answer)
#
# This is what creates the LOOP:
#
#   agent → has tool call? → YES → tools → back to agent → has tool call?...
#                           → NO  → END
#
# The loop keeps going until the LLM decides it has enough information
# and responds with plain text instead of a tool call.
#
# How do we detect a tool call?
#
# When the LLM wants to use a tool, its response (an AIMessage) has a
# field called `tool_calls` — a list of tool call requests. If that list
# is non-empty, the LLM is asking for a tool. If it's empty (or missing),
# the LLM is giving a final text answer.
# ---------------------------------------------------------------------------

from langgraph.graph import END


def should_continue(state: ResearchState) -> str:
    """Decide: did the LLM call a tool, or give a final answer?

    This function is used as a conditional edge (same concept as
    route_by_confidence in the chatbot). LangGraph calls it after
    the agent node runs, and uses the returned string to pick
    which node to go to next.
    """
    # Get the last message (the one the agent just produced)
    last_message = state["messages"][-1]

    # Check if it has tool calls
    if last_message.tool_calls:
        # The LLM wants to search → go to the tools node
        return "tools"
    else:
        # The LLM gave a final answer → we're done
        return END


# ---------------------------------------------------------------------------
# 5. GRAPH ASSEMBLY
#
# Now we connect everything. Let's compare with the chatbot:
#
# CHATBOT graph:
#   Nodes: classify_intent, respond, ask_clarification
#   Flow:  START → classify → (confidence check) → respond OR clarify → END
#   Shape: a fork (two possible paths, no loops)
#
# RESEARCH AGENT graph:
#   Nodes: agent, tools
#   Flow:  START → agent → (tool call check) → tools → agent → ... → END
#   Shape: a loop (keeps cycling until the LLM is satisfied)
#
# The chatbot was like a hallway with a fork: go left or right, then exit.
# The research agent is like a roundabout: keep circling until you find
# your exit.
#
# Key wiring:
#   1. START → agent                          (always start with the LLM)
#   2. agent → should_continue (conditional)  (check: tool call or done?)
#   3. tools → agent                          (after running tool, ask LLM again)
#
# That's it. Three connections create the full agent loop.
# ---------------------------------------------------------------------------

from langgraph.graph import StateGraph


def build_research_agent():
    """Build and compile the research agent graph."""

    # Create the graph with our state type
    graph = StateGraph(ResearchState)

    # Add our two nodes
    graph.add_node("agent", agent)       # the LLM "thinker"
    graph.add_node("tools", tool_node)   # the tool executor

    # Set where the graph starts — always the agent first
    graph.set_entry_point("agent")

    # THE LOOP — this is the core pattern:
    #
    # After the agent runs, call should_continue to decide what's next.
    # If should_continue returns "tools" → go to tools node
    # If should_continue returns END    → stop the graph
    graph.add_conditional_edges("agent", should_continue)

    # After tools run, ALWAYS go back to the agent.
    # This is what closes the loop — the agent gets to see the tool
    # results and decide again.
    graph.add_edge("tools", "agent")

    # Compile turns the builder into a runnable application
    return graph.compile()


# ---------------------------------------------------------------------------
# 6. RUN IT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = build_research_agent()

    # Start with a user question — same as the chatbot, we pass initial state
    result = app.invoke({
        "messages": [("user", "How do LangGraph, CrewAI, and LlamaIndex differ?")],
        "documents": [],
        "citations": [],
    })

    # Print the conversation (including tool calls and results)
    print("\n=== Research Agent Result ===\n")
    for msg in result["messages"]:
        # Each message has a .type: "human", "ai", or "tool"
        role = msg.type
        if role == "ai" and msg.tool_calls:
            # This is a tool call request (not a text response)
            for tc in msg.tool_calls:
                print(f"[agent → tool call] {tc['name']}({tc['args']})")
        elif role == "tool":
            # This is a tool result coming back
            print(f"[tool result] {msg.content[:100]}...")
        else:
            # Human message or final AI response
            print(f"[{role}] {msg.content}")
        print()  # blank line between messages
