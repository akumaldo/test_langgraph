"""
Technical support sub-graph — the MOST COMPLEX of the three.

Same sub-graph pattern (three types for isolation), but two new things:
  1. All three nodes use the LLM (fully LLM-driven)
  2. The diagnose node uses interrupt() to PAUSE and ask the user a question

NEW CONCEPT: interrupt() — Human-in-the-Loop mid-execution
==========================================================

In P4, the human interacted OUTSIDE the graph:
    while True:
        question = input("Ask: ")       # human chooses WHAT to analyze
        result = app.invoke(state)      # graph runs to completion
        print(result["report"])         # human sees result
    The graph ran start-to-finish without stopping.

With interrupt(), the graph PAUSES in the middle:
    START → diagnose → [PAUSE: "What OS are you using?"]
                              ↓
                       [user types "Windows 11"]
                              ↓
                       suggest_fix → draft_response → END

How it works:
    1. diagnose() calls interrupt("What OS are you using?")
    2. The graph STOPS. State is saved by the checkpointer.
    3. The caller (parent graph) gets an Interrupt event.
    4. The user provides input.
    5. The caller resumes with Command(resume="Windows 11")
    6. The interrupt() call RETURNS "Windows 11" and diagnose() continues.

The key insight: interrupt() is like input() but for graphs.
  - input() pauses a Python program and waits for terminal input
  - interrupt() pauses a LangGraph and waits for the caller to resume

And because we're inside a SUB-GRAPH, the interrupt bubbles up to the
PARENT graph. The parent's checkpointer saves everything, and when the
user responds, execution resumes right here in the technical sub-graph.

This only works because we use the compiled sub-graph as a node
(parent_graph.add_node("tech", compiled_subgraph)). If we used a
wrapper function instead, interrupt wouldn't bubble up correctly.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph, add_messages
from langgraph.types import interrupt


# ---------------------------------------------------------------------------
# 1. STATE TYPES
#
# TechnicalState has MORE internal fields than billing/returns because
# the diagnostic flow needs to track: diagnosis, OS info, suggested fix.
# ---------------------------------------------------------------------------

class TechnicalState(TypedDict):
    """Full internal state for the technical sub-graph."""
    messages: Annotated[list[BaseMessage], add_messages]
    customer_issue: str        # from parent
    os_info: str               # filled by interrupt() — user provides this mid-flow
    diagnosis: str             # LLM's initial diagnosis (internal)
    suggested_fix: str         # LLM's suggested solution (internal)
    department_response: str   # final answer (back to parent)


class TechnicalInput(TypedDict):
    """What the parent passes in."""
    customer_issue: str


class TechnicalOutput(TypedDict):
    """What flows back to the parent."""
    department_response: str


# ---------------------------------------------------------------------------
# 2. LLM SETUP
# ---------------------------------------------------------------------------

llm = ChatOllama(model="qwen3.5:35b", temperature=0)


# ---------------------------------------------------------------------------
# 3. NODE FUNCTIONS
#
# Three nodes, ALL use the LLM. The first one also uses interrupt().
# ---------------------------------------------------------------------------

def diagnose(state: TechnicalState) -> dict:
    """Node 1: LLM analyzes the issue, then interrupt() asks for OS info.

    This node does TWO things:
      1. LLM call — get an initial diagnosis based on the issue description
      2. interrupt() — pause to ask the user what OS they're on

    The interrupt() call:
      - PAUSES the graph (state is checkpointed)
      - Returns a value to the parent indicating what question to ask
      - When the user responds and the graph resumes, interrupt() RETURNS
        the user's answer as a string
      - Execution continues from the line AFTER the interrupt() call

    Think of it like:
        diagnosis = llm.invoke(...)      # step 1: analyze
        os_info = interrupt("question")  # step 2: ask user, PAUSE here
        return {"diagnosis": ..., "os_info": os_info}  # step 3: continue after resume
    """
    # Step 1: LLM analyzes the issue
    system = SystemMessage(content=(
        "You are a technical support specialist. Analyze the customer's "
        "technical issue and provide an initial diagnosis. Be concise — "
        "1-2 sentences about what you think the problem is."
    ))

    human = HumanMessage(content=f"CUSTOMER ISSUE: {state['customer_issue']}")

    diagnosis_response = llm.invoke([system, human])
    diagnosis = diagnosis_response.content

    # Step 2: interrupt() to ask for more info
    #
    # This is the KEY MOMENT — the graph STOPS here.
    # The string we pass to interrupt() is the "question" shown to the user.
    # When the user responds, the interrupt() call returns their answer.
    os_info = interrupt(
        "To help resolve your technical issue, could you please tell me "
        "what operating system you're using? (e.g., Windows 11, macOS, Ubuntu)"
    )

    # Step 3: We only get here AFTER the user responds
    return {
        "diagnosis": diagnosis,
        "os_info": os_info,
        "messages": [AIMessage(content=f"[Technical] Diagnosis: {diagnosis} | OS: {os_info}")],
    }


def suggest_fix(state: TechnicalState) -> dict:
    """Node 2: LLM suggests a fix using the diagnosis + OS info.

    This runs AFTER the user has provided their OS (after interrupt resume).
    Now the LLM has everything it needs: the issue, the diagnosis, and
    the user's operating system.
    """
    system = SystemMessage(content=(
        "You are a technical support specialist. Based on the diagnosis and "
        "the user's operating system, suggest a specific fix. Include step-by-step "
        "instructions. Keep it to 3-5 steps."
    ))

    human = HumanMessage(content=(
        f"CUSTOMER ISSUE: {state['customer_issue']}\n\n"
        f"DIAGNOSIS: {state['diagnosis']}\n\n"
        f"OPERATING SYSTEM: {state['os_info']}"
    ))

    fix_response = llm.invoke([system, human])

    return {
        "suggested_fix": fix_response.content,
        "messages": [AIMessage(content=f"[Technical] Fix suggested.")],
    }


def draft_response(state: TechnicalState) -> dict:
    """Node 3: LLM writes the final customer-facing response.

    Combines the diagnosis and fix into a friendly, complete response.
    """
    system = SystemMessage(content=(
        "You are a friendly technical support agent. Write a helpful response "
        "to the customer that includes your diagnosis and the suggested fix. "
        "Be empathetic and clear. Keep it concise but complete."
    ))

    human = HumanMessage(content=(
        f"CUSTOMER ISSUE: {state['customer_issue']}\n\n"
        f"DIAGNOSIS: {state['diagnosis']}\n\n"
        f"OPERATING SYSTEM: {state['os_info']}\n\n"
        f"SUGGESTED FIX:\n{state['suggested_fix']}"
    ))

    response = llm.invoke([system, human])

    return {
        "department_response": response.content,
        "messages": [AIMessage(content="[Technical] Response drafted via LLM.")],
    }


# ---------------------------------------------------------------------------
# 4. GRAPH ASSEMBLY
#
# Same pattern as returns and billing. Linear pipeline:
#   diagnose → suggest_fix → draft_response → END
#
# The interrupt() inside diagnose is invisible to the graph structure.
# LangGraph doesn't need special edges for it — the interrupt happens
# INSIDE the node function, not between nodes.
# ---------------------------------------------------------------------------

def build_technical_subgraph():
    """Build and compile the technical support sub-graph."""
    graph = StateGraph(TechnicalState, input=TechnicalInput, output=TechnicalOutput)

    graph.add_node("diagnose", diagnose)
    graph.add_node("suggest_fix", suggest_fix)
    graph.add_node("draft_response", draft_response)

    graph.set_entry_point("diagnose")
    graph.add_edge("diagnose", "suggest_fix")
    graph.add_edge("suggest_fix", "draft_response")
    graph.add_edge("draft_response", END)

    return graph.compile()
