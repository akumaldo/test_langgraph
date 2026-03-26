"""
Billing department sub-graph — MEDIUM complexity.

Same sub-graph pattern as returns.py (three types for isolation), but now
the draft_response node makes an LLM call to write a natural response.

Compare:
  returns.py  → all rule-based, template response (simplest)
  billing.py  → rule-based logic + ONE LLM call for the response (medium)
  technical.py → full LLM-driven + interrupt() (complex)

This gradient lets you see the same sub-graph pattern at different
complexity levels. The INPUT/OUTPUT schema wiring is identical in all three.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph, add_messages


# ---------------------------------------------------------------------------
# 1. STATE TYPES — same trio pattern as returns.py
#
# The only difference: billing has its own internal fields
# (account_status, refund_amount) that the parent never sees.
# ---------------------------------------------------------------------------

class BillingState(TypedDict):
    """Full internal state for the billing sub-graph."""
    messages: Annotated[list[BaseMessage], add_messages]
    customer_issue: str        # from parent (via input schema)
    account_status: str        # simulated account lookup (internal)
    refund_amount: float       # calculated refund, if any (internal)
    billing_response: str      # internal draft before finalizing
    department_response: str   # final answer (back to parent via output schema)


class BillingInput(TypedDict):
    """What the parent passes in."""
    customer_issue: str


class BillingOutput(TypedDict):
    """What flows back to the parent."""
    department_response: str


# ---------------------------------------------------------------------------
# 2. LLM SETUP
#
# Same model as P1-P4. Only used by draft_response — the other two
# nodes are rule-based (no LLM needed for account lookup or calculation).
# ---------------------------------------------------------------------------

llm = ChatOllama(model="qwen3.5:35b", temperature=0)


# ---------------------------------------------------------------------------
# 3. SIMULATED DATA
#
# In a real app, check_account would query a billing database.
# Here we simulate a few account scenarios so the demo is interesting.
# ---------------------------------------------------------------------------

SIMULATED_ACCOUNTS = {
    "default": {
        "status": "active",
        "plan": "Premium",
        "monthly_charge": 29.99,
        "last_payment": "2026-03-01",
        "overcharged": False,
    },
    "overcharged": {
        "status": "active",
        "plan": "Basic",
        "monthly_charge": 9.99,
        "last_payment": "2026-03-01",
        "overcharged": True,
        "overcharge_amount": 20.00,
    },
    "cancelled": {
        "status": "cancelled",
        "plan": "None",
        "monthly_charge": 0.0,
        "last_payment": "2026-02-15",
        "overcharged": False,
    },
}


# ---------------------------------------------------------------------------
# 4. NODE FUNCTIONS
# ---------------------------------------------------------------------------

def check_account(state: BillingState) -> dict:
    """Node 1: Look up the customer's account status.

    Simulated — picks an account scenario based on keywords in the issue.
    In a real app, you'd query a database by customer ID.
    """
    issue = state["customer_issue"].lower()

    # Pick a simulated scenario based on keywords
    if any(word in issue for word in ("overcharged", "charged twice", "double charge", "wrong amount")):
        account = SIMULATED_ACCOUNTS["overcharged"]
    elif any(word in issue for word in ("cancelled", "canceled", "closed")):
        account = SIMULATED_ACCOUNTS["cancelled"]
    else:
        account = SIMULATED_ACCOUNTS["default"]

    status_summary = (
        f"Account status: {account['status']}, "
        f"Plan: {account['plan']}, "
        f"Monthly charge: ${account['monthly_charge']:.2f}, "
        f"Last payment: {account['last_payment']}"
    )

    return {
        "account_status": status_summary,
        "messages": [AIMessage(content=f"[Billing] Account lookup: {status_summary}")],
    }


def calculate_resolution(state: BillingState) -> dict:
    """Node 2: Determine the billing resolution.

    Rule-based — looks at account status and calculates refund if applicable.
    This is like returns' process_return but for billing scenarios.
    """
    issue = state["customer_issue"].lower()
    account_status = state["account_status"]

    refund = 0.0
    resolution = ""

    if "overcharged" in account_status.lower() or any(
        word in issue for word in ("overcharged", "charged twice", "double charge")
    ):
        # Overcharge scenario — the simulated account has an overcharge_amount
        refund = 20.00
        resolution = f"Overcharge confirmed. Refund of ${refund:.2f} will be issued."
    elif "cancelled" in account_status.lower():
        resolution = "Account is already cancelled. No active charges found."
    elif any(word in issue for word in ("upgrade", "downgrade", "change plan")):
        resolution = "Plan change request noted. Will be effective next billing cycle."
    else:
        resolution = "Account is in good standing. No billing issues detected."

    return {
        "refund_amount": refund,
        "billing_response": resolution,
        "messages": [AIMessage(content=f"[Billing] Resolution: {resolution}")],
    }


def draft_response(state: BillingState) -> dict:
    """Node 3: LLM writes a customer-friendly billing response.

    THIS is the difference from returns.py — instead of a template,
    we ask the LLM to write a natural, empathetic response.

    Compare:
      returns draft_response: f-string template → fast, predictable
      billing draft_response: LLM call → slower, but more natural

    The LLM gets context about the account status and resolution so it
    can write something specific and helpful.
    """
    system = SystemMessage(content=(
        "You are a friendly billing support agent. Write a brief, empathetic "
        "response to the customer based on the account information and resolution. "
        "Keep it to 2-3 sentences. Be specific about amounts and next steps."
    ))

    human = HumanMessage(content=(
        f"CUSTOMER ISSUE: {state['customer_issue']}\n\n"
        f"ACCOUNT INFO: {state['account_status']}\n\n"
        f"RESOLUTION: {state['billing_response']}\n\n"
        f"REFUND AMOUNT: ${state['refund_amount']:.2f}"
    ))

    response = llm.invoke([system, human])

    return {
        "department_response": response.content,
        "messages": [AIMessage(content="[Billing] Response drafted via LLM.")],
    }


# ---------------------------------------------------------------------------
# 5. GRAPH ASSEMBLY — identical pattern to returns.py
#
# StateGraph(full_state, input=InputSchema, output=OutputSchema)
# Same wiring, same compile(), same "no checkpointer on sub-graph" rule.
# ---------------------------------------------------------------------------

def build_billing_subgraph():
    """Build and compile the billing department sub-graph."""
    graph = StateGraph(BillingState, input=BillingInput, output=BillingOutput)

    graph.add_node("check_account", check_account)
    graph.add_node("calculate_resolution", calculate_resolution)
    graph.add_node("draft_response", draft_response)

    graph.set_entry_point("check_account")
    graph.add_edge("check_account", "calculate_resolution")
    graph.add_edge("calculate_resolution", "draft_response")
    graph.add_edge("draft_response", END)

    return graph.compile()
