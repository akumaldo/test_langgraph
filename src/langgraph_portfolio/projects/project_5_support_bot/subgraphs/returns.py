"""
Returns department sub-graph — the SIMPLEST of the three.

This is your first sub-graph. Let's learn the pattern here where it's easy,
then apply it to billing (medium) and technical (complex).

NEW CONCEPT: Sub-graphs with input/output schemas
================================================

In P1-P4, you had ONE StateGraph:
    graph = StateGraph(MyState)
    graph.add_node(...)
    app = graph.compile()

Now each department is its OWN StateGraph with its OWN state type.
But the parent graph needs to talk to it. How?

LangGraph uses KEY NAME MATCHING. If the parent state has a field called
"customer_issue" and the sub-graph's INPUT schema also has "customer_issue",
LangGraph automatically passes that value in.

Same for output: if the sub-graph's OUTPUT schema has "department_response"
and the parent state also has "department_response", the value flows back.

You control what flows in/out with THREE types:

    ReturnsState  — full internal state (all fields the sub-graph needs)
    ReturnsInput  — what the parent passes in (just customer_issue)
    ReturnsOutput — what flows back to the parent (just department_response)

    StateGraph(ReturnsState, input=ReturnsInput, output=ReturnsOutput)

The sub-graph can use internal fields (order_id, return_reason, etc.)
freely — the parent NEVER sees them. Only the output schema keys go back.

This is like function parameters and return values:
    def process_return(customer_issue: str) -> str:  # department_response
        order_id = extract_order(customer_issue)     # internal variable
        ...
        return response                              # only this goes back
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph import END, StateGraph, add_messages


# ---------------------------------------------------------------------------
# 1. STATE TYPES — three types for isolation
#
# Compare to P1-P4 where you had just ONE state type per project.
# Now each sub-graph defines its own trio: full state + input + output.
# ---------------------------------------------------------------------------

class ReturnsState(TypedDict):
    """Full internal state — everything the returns sub-graph needs."""
    messages: Annotated[list[BaseMessage], add_messages]
    customer_issue: str        # what the customer said (comes from parent)
    order_id: str              # extracted from customer message (internal)
    return_reason: str         # why they want to return (internal)
    return_eligible: bool      # are they within the return window? (internal)
    department_response: str   # final answer (goes back to parent)


class ReturnsInput(TypedDict):
    """What the parent passes IN. Only keys shared with parent state."""
    customer_issue: str


class ReturnsOutput(TypedDict):
    """What flows back OUT to the parent. Only keys shared with parent state."""
    department_response: str


# ---------------------------------------------------------------------------
# 2. NODE FUNCTIONS
#
# Three simple nodes, all rule-based (no LLM calls).
# This keeps the focus on learning the sub-graph pattern itself.
# ---------------------------------------------------------------------------

# Simulated data — in a real app, this would come from a database
RETURN_WINDOW_DAYS = 30


def validate_return(state: ReturnsState) -> dict:
    """Node 1: Extract order info and check return eligibility.

    In a real app, this would query an order database. Here we simulate
    it by looking for an order ID pattern in the customer message and
    always granting a 30-day return window.
    """
    issue = state["customer_issue"].lower()

    # Simple extraction — look for "order" followed by something
    order_id = "ORD-UNKNOWN"
    if "order" in issue:
        # Try to grab the word after "order"
        parts = issue.split("order")
        if len(parts) > 1:
            after = parts[1].strip().split()[0] if parts[1].strip() else ""
            if after:
                order_id = f"ORD-{after.upper().strip('#:- ')}"

    # Simulated: return is eligible if they mention it within the window
    # In reality, you'd check order date vs. current date
    eligible = True
    reason = "not specified"
    for keyword in ("broken", "defective", "wrong", "damaged"):
        if keyword in issue:
            reason = keyword
            break
    for keyword in ("changed my mind", "don't want", "don't need"):
        if keyword in issue:
            reason = keyword
            break

    return {
        "order_id": order_id,
        "return_reason": reason,
        "return_eligible": eligible,
        "messages": [AIMessage(content=f"[Returns] Validated order {order_id}. Eligible: {eligible}")],
    }


def process_return(state: ReturnsState) -> dict:
    """Node 2: Process the return based on eligibility.

    Determines the return type (refund, exchange, or denied)
    based on the validation results.
    """
    if not state["return_eligible"]:
        decision = "Return denied — outside the return window."
    elif state["return_reason"] in ("broken", "defective", "damaged"):
        decision = f"Full refund approved for {state['order_id']}. Defective item — no return shipping needed."
    elif state["return_reason"] == "wrong":
        decision = f"Exchange approved for {state['order_id']}. We'll send the correct item."
    else:
        decision = f"Return approved for {state['order_id']}. Refund will be processed after we receive the item."

    return {
        "messages": [AIMessage(content=f"[Returns] Decision: {decision}")],
        "department_response": "",  # will be set by draft_response
    }


def draft_response(state: ReturnsState) -> dict:
    """Node 3: Template a customer-friendly response.

    No LLM call — this is a simple template. Compare to the billing and
    technical sub-graphs where the response is LLM-generated.
    """
    if not state["return_eligible"]:
        response = (
            f"I'm sorry, but your return request for order {state['order_id']} "
            f"is outside our {RETURN_WINDOW_DAYS}-day return window. "
            "Please contact us if you believe this is an error."
        )
    elif state["return_reason"] in ("broken", "defective", "damaged"):
        response = (
            f"I'm sorry about the defective item in order {state['order_id']}! "
            "We've approved a full refund — no need to ship it back. "
            "The refund will appear in 3-5 business days."
        )
    elif state["return_reason"] == "wrong":
        response = (
            f"Sorry about the mix-up with order {state['order_id']}! "
            "We'll send the correct item right away. "
            "Please use the prepaid label we're emailing you to return the wrong one."
        )
    else:
        response = (
            f"Your return for order {state['order_id']} has been approved. "
            "Please ship the item back using the prepaid label we're emailing you. "
            "Your refund will be processed within 5 business days of receiving the item."
        )

    return {
        "department_response": response,
        "messages": [AIMessage(content="[Returns] Response drafted.")],
    }


# ---------------------------------------------------------------------------
# 3. GRAPH ASSEMBLY
#
# This is just like P1-P4's graph assembly, but:
#   - StateGraph takes THREE arguments: full state, input schema, output schema
#   - The compiled graph will be added as a NODE in the parent graph
#
# When the parent does:
#   parent_graph.add_node("returns_subgraph", returns_graph)
#
# LangGraph will:
#   1. Read ReturnsInput to know which parent keys to pass in
#   2. Run the sub-graph's full flow (validate → process → draft)
#   3. Read ReturnsOutput to know which keys to send back to the parent
# ---------------------------------------------------------------------------

def build_returns_subgraph():
    """Build and compile the returns department sub-graph.

    Returns a compiled graph ready to be used as a node in the parent.
    """
    graph = StateGraph(ReturnsState, input=ReturnsInput, output=ReturnsOutput)

    graph.add_node("validate_return", validate_return)
    graph.add_node("process_return", process_return)
    graph.add_node("draft_response", draft_response)

    graph.set_entry_point("validate_return")
    graph.add_edge("validate_return", "process_return")
    graph.add_edge("process_return", "draft_response")
    graph.add_edge("draft_response", END)

    # No checkpointer here — the PARENT's checkpointer handles everything.
    # Sub-graphs inherit the parent's checkpointer when used as nodes.
    return graph.compile()
