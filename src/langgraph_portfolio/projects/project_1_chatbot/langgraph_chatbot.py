"""
Project 1 — Multi-intent chatbot using real LangGraph.

We build this in layers:
  1. State definition (TypedDict + reducer)
  2. Node functions (each one does one job)
  3. Routing logic (conditional edges)
  4. Graph assembly (wiring it all together)
"""

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph, add_messages


# ---------------------------------------------------------------------------
# 1. STATE
#
# This is the "product on the assembly line." Every node can read it and
# return updates to it.
#
# The Annotated[..., add_messages] part is a *reducer*: it tells LangGraph
# "append new messages to the list" instead of replacing the whole list.
# Without it, each node would overwrite the conversation history.
# ---------------------------------------------------------------------------

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # conversation history
    intent: str       # detected intent (e.g. "comparison", "research")
    confidence: float  # how confident the classifier is (0.0 to 1.0)


# ---------------------------------------------------------------------------
# LLM setup
#
# ChatOllama is langchain's wrapper around Ollama. It speaks the same
# "messages in, message out" interface that all langchain chat models use.
# This means if you later switch to Claude or GPT, you only change this line.
# ---------------------------------------------------------------------------

llm = ChatOllama(model="qwen3.5:35b", temperature=0)


# ---------------------------------------------------------------------------
# 2. NODES
#
# Each node is a plain function:  state in → dict of updates out.
#
# The dict you return gets *merged* into the state. So if you return
# {"intent": "research"}, only the intent field is updated — everything
# else stays the same.
#
# For the "messages" field specifically, because we used the add_messages
# reducer, returning {"messages": [new_msg]} will APPEND, not replace.
# ---------------------------------------------------------------------------

def classify_intent(state: ChatState) -> dict:
    """Ask the LLM to classify the user's intent.

    We send a system prompt that constrains the LLM to output just the
    intent label and a confidence score. This is a simple approach —
    later you could use structured output or tool calling for more
    reliability.
    """
    classification_prompt = SystemMessage(content=(
        "You are an intent classifier. Analyze the user's last message and "
        "respond with EXACTLY two lines:\n"
        "INTENT: <one of: comparison, research, analysis, general>\n"
        "CONFIDENCE: <a number between 0.0 and 1.0>\n"
        "Do not say anything else."
    ))

    # We send the system prompt + the full conversation so far
    response = llm.invoke([classification_prompt] + state["messages"])

    # Parse the LLM's response to extract intent and confidence
    intent = "general"
    confidence = 0.5
    for line in response.content.strip().splitlines():
        line = line.strip().upper()
        if line.startswith("INTENT:"):
            intent = line.split(":", 1)[1].strip().lower()
        elif line.startswith("CONFIDENCE:"):
            try:
                confidence = float(line.split(":", 1)[1].strip())
            except ValueError:
                confidence = 0.5

    # Return ONLY the fields we want to update
    # Note: we do NOT return "messages" here because the classification
    # is internal work — we don't want to show it to the user
    return {"intent": intent, "confidence": confidence}


def respond(state: ChatState) -> dict:
    """Generate a response based on the classified intent.

    Now the LLM knows the intent, so we give it a persona and let it
    answer naturally. The response message gets appended to the
    conversation via the add_messages reducer.
    """
    system_prompt = SystemMessage(content=(
        f"You are a helpful AI assistant. The user's intent has been "
        f"classified as '{state['intent']}' with confidence "
        f"{state['confidence']:.0%}.\n\n"
        f"Respond helpfully based on this intent. Keep your answer concise."
    ))

    response = llm.invoke([system_prompt] + state["messages"])

    # Returning messages here — the reducer will APPEND this to the list
    return {"messages": [response]}


def ask_clarification(state: ChatState) -> dict:
    """When confidence is low, ask the user to clarify.

    This is the 'else' branch of our router — if we're not sure what
    the user wants, we ask instead of guessing wrong.
    """
    msg = AIMessage(content=(
        "I'm not quite sure what you're looking for. Could you clarify? "
        "For example, are you interested in:\n"
        "- **Comparing** frameworks (LangGraph vs CrewAI)\n"
        "- **Researching** a topic with document retrieval\n"
        "- **Analyzing** data\n"
        "- Or something else?"
    ))
    return {"messages": [msg]}


# ---------------------------------------------------------------------------
# 3. ROUTING
#
# In your old scaffold, routing was a node (the "router" function).
# In real LangGraph, routing is an *edge* — specifically a
# "conditional edge." You provide a function that looks at the state
# and returns the NAME of the next node to go to.
#
# This is a cleaner separation: nodes do WORK, edges decide WHERE to go.
# ---------------------------------------------------------------------------

def route_by_confidence(state: ChatState) -> str:
    """Decide next node based on classification confidence.

    Returns a node name (string). LangGraph uses this to pick the edge.
    """
    if state["confidence"] >= 0.7:
        return "respond"
    else:
        return "ask_clarification"


# ---------------------------------------------------------------------------
# 4. GRAPH ASSEMBLY
#
# This is where we wire everything together. Think of it as drawing
# a flowchart:
#
#   START → classify_intent →(confidence >= 0.7?)→ respond → END
#                             (otherwise)→ ask_clarification → END
#
# StateGraph takes our state type so it knows the shape of the data.
# Then we add nodes, edges, set the entry point, and compile.
# ---------------------------------------------------------------------------

def build_chatbot():
    """Build and compile the chatbot graph."""

    # Create the graph, telling it what our state looks like
    graph = StateGraph(ChatState)

    # Add nodes — each gets a name and a function
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("respond", respond)
    graph.add_node("ask_clarification", ask_clarification)

    # Set where the graph starts
    graph.set_entry_point("classify_intent")

    # Add the conditional edge: after classify_intent, call route_by_confidence
    # to decide which node runs next
    graph.add_conditional_edges("classify_intent", route_by_confidence)

    # Both respond and ask_clarification lead to END (graph stops)
    graph.add_edge("respond", END)
    graph.add_edge("ask_clarification", END)

    # Compile turns the builder into a runnable application
    return graph.compile()


# ---------------------------------------------------------------------------
# 5. RUN IT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = build_chatbot()

    # invoke() takes an initial state dict and returns the final state
    result = app.invoke({
        "messages": [("user", "Can you compare LangGraph and CrewAI?")],
        "intent": "",
        "confidence": 0.0,
    })

    # Print the conversation
    print("\n=== Chatbot Result ===")
    for msg in result["messages"]:
        role = msg.type  # "human" or "ai"
        print(f"\n[{role}]: {msg.content}")
    print(f"\n--- Intent: {result['intent']} | Confidence: {result['confidence']:.0%} ---")
