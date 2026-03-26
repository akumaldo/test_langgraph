"""
Project 5 — Customer Support Bot using real LangGraph (Advanced).

Building on everything from Projects 1-4:
  P1: state + nodes + conditional edges (fork shape)
  P2: tools + agent loop (roundabout shape)
  P3: multi-agent pipeline + quality-control loop
  P4: structured output + code execution + error recovery + checkpointing

New things in this project:
  1. Sub-graphs — graphs inside graphs (each department is its own StateGraph)
  2. interrupt() — the graph PAUSES mid-execution to ask the user a question
  3. Streaming — see output as it's generated, not all at once
  4. Input/output schemas — sub-graphs have isolated state (production pattern)

The big picture:

    PARENT GRAPH (this file)
    ├── classify_issue          (LLM + structured output → picks department)
    ├── route_to_department     (conditional edge → one of the sub-graphs)
    │   ├── billing_subgraph    (compiled sub-graph as a node)
    │   ├── technical_subgraph  (compiled sub-graph as a node, uses interrupt)
    │   ├── returns_subgraph    (compiled sub-graph as a node)
    │   └── fallback_response   (simple node for unknown category)
    └── collect_feedback        (interrupt → asks for satisfaction rating)

Compare to P3 (Writing Team) which had 5 agents in ONE graph:
  planner → researcher → writer ←→ editor → publisher

P5's parent graph is SIMPLER (fewer nodes), but the nodes themselves are
MORE POWERFUL because some of them are entire sub-graphs with their own
internal flows.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, add_messages
from langgraph.types import Command, interrupt

from .models import IssueClassification
from .subgraphs.billing import build_billing_subgraph
from .subgraphs.returns import build_returns_subgraph
from .subgraphs.technical import build_technical_subgraph


# ---------------------------------------------------------------------------
# 1. PARENT STATE
#
# Compare to previous projects:
#   P1: messages, intent, confidence
#   P2: messages, documents, citations
#   P3: messages, topic, outline, draft, editor_feedback, ...
#   P4: messages, question, csv data, plan, step tracking, charts, report
#   P5: messages, customer_issue, category, department_response, rating
#
# This is actually SIMPLER than P3 and P4 because the complexity lives
# INSIDE the sub-graphs, not in the parent state. The parent only tracks:
#   - What the customer said (customer_issue)
#   - Which department handles it (category)
#   - What the department answered (department_response)
#   - How satisfied the customer is (satisfaction_rating)
# ---------------------------------------------------------------------------

class SupportState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    customer_issue: str          # raw user input
    category: str                # "billing" | "technical" | "returns" | "unknown"
    department_response: str     # final answer from whichever sub-graph ran
    satisfaction_rating: int     # 1-5 from feedback collection (initial: 0)


# ---------------------------------------------------------------------------
# 2. LLM SETUP
#
# Same as P4: one base LLM, and a structured-output variant for the
# classifier. The sub-graphs have their own LLM instances.
# ---------------------------------------------------------------------------

llm = ChatOllama(model="qwen3.5:35b", temperature=0)
llm_classifier = llm.with_structured_output(IssueClassification)


# ---------------------------------------------------------------------------
# 3. PARENT NODE FUNCTIONS
# ---------------------------------------------------------------------------

# Confidence threshold — below this, we route to "unknown"
MIN_CONFIDENCE = 0.5


def classify_issue(state: SupportState) -> dict:
    """Node 1: LLM classifies the customer's issue into a department.

    Uses with_structured_output() — same pattern as P4's plan_analysis.
    The LLM returns an IssueClassification object, not a string.

    If confidence is below MIN_CONFIDENCE, we override to "unknown"
    so the customer gets a helpful fallback instead of a wrong department.
    """
    system = SystemMessage(content=(
        "You are a customer support classifier. Categorize the customer's "
        "issue into one of these departments:\n"
        "- billing: payment issues, charges, refunds, subscriptions, pricing\n"
        "- technical: software bugs, connectivity, errors, performance\n"
        "- returns: product returns, exchanges, damaged items, wrong items\n"
        "- unknown: anything that doesn't clearly fit the above\n\n"
        "Be precise. If unsure, choose 'unknown'."
    ))

    human = HumanMessage(content=f"CUSTOMER ISSUE: {state['customer_issue']}")

    classification: IssueClassification = llm_classifier.invoke([system, human])

    # Override to unknown if confidence is too low
    category = classification.category
    if classification.confidence < MIN_CONFIDENCE:
        category = "unknown"

    return {
        "category": category,
        "messages": [AIMessage(content=(
            f"[Classifier] Category: {category} "
            f"(confidence: {classification.confidence:.0%}, "
            f"reasoning: {classification.reasoning})"
        ))],
    }


def fallback_response(state: SupportState) -> dict:
    """Fallback node for when the category is "unknown".

    Simple, no LLM call — just a friendly message asking the customer
    to rephrase or connect with a human agent.
    """
    response = (
        "I'm not quite sure which department can help with this. "
        "Could you rephrase your issue, or would you like me to "
        "connect you with a human agent?"
    )
    return {
        "department_response": response,
        "messages": [AIMessage(content="[Fallback] Unknown category — generic response.")],
    }


def collect_feedback(state: SupportState) -> dict:
    """Node (final): Ask the customer for a satisfaction rating.

    THIS is interrupt() at the PARENT level (compare to the technical
    sub-graph which uses interrupt() inside a sub-graph).

    The flow:
      1. We show the department's response to the customer
      2. interrupt() PAUSES the graph
      3. The user provides a 1-5 rating
      4. interrupt() RETURNS the rating
      5. We store it and the graph ends

    Two interrupt() points in this project:
      - Here (parent level): asks for satisfaction rating
      - Inside technical sub-graph: asks for OS info

    Both work the same way — the graph pauses, state is checkpointed,
    and resumes when the user provides input via Command(resume=value).
    """
    # Show the response before asking for feedback
    rating = interrupt(
        f"Here's what our team found:\n\n"
        f"{state['department_response']}\n\n"
        f"How would you rate this response? (1-5, where 5 is excellent)"
    )

    # Parse the rating — be forgiving with input
    try:
        parsed_rating = int(str(rating).strip())
        parsed_rating = max(1, min(5, parsed_rating))  # clamp to 1-5
    except (ValueError, TypeError):
        parsed_rating = 3  # default if they type something weird

    return {
        "satisfaction_rating": parsed_rating,
        "messages": [AIMessage(content=f"[Feedback] Rating: {parsed_rating}/5")],
    }


# ---------------------------------------------------------------------------
# 4. ROUTING LOGIC
#
# Compare to previous projects:
#   P1: confidence >= 0.7? → respond or clarify (two-way fork)
#   P2: tool calls? → tools or END (two-way loop)
#   P3: editor approved? → publisher or writer (two-way loop)
#   P4: error? more steps? done? (three-way fork)
#   P5: billing? technical? returns? unknown? (FOUR-way fork)
#
# This is our first four-way conditional edge. But each path leads to
# the same place (collect_feedback → END), so it's simpler than P4's
# step loop despite having more branches.
# ---------------------------------------------------------------------------

def route_to_department(state: SupportState) -> str:
    """Four-way routing based on the category from classify_issue.

    Returns the name of the node to go to next.
    """
    category = state["category"]
    if category == "billing":
        return "billing_subgraph"
    elif category == "technical":
        return "technical_subgraph"
    elif category == "returns":
        return "returns_subgraph"
    else:
        return "fallback_response"


# ---------------------------------------------------------------------------
# 5. GRAPH ASSEMBLY
#
# This is where it all comes together. The parent graph:
#   1. Has its own nodes (classify_issue, fallback_response, collect_feedback)
#   2. Has COMPILED SUB-GRAPHS as nodes (billing, technical, returns)
#   3. Uses conditional edges to route between them
#
# The compiled sub-graphs are added with add_node() just like regular
# nodes. LangGraph treats them as first-class nodes:
#   - State mapping happens automatically via input/output schemas
#   - interrupt() inside sub-graphs bubbles up correctly
#   - Streaming events from sub-graph nodes are visible
#   - The parent's checkpointer covers everything
#
# Compare graph assembly across all projects:
#   P1: 3 nodes, 1 conditional edge (simple fork)
#   P2: 2 nodes, 1 conditional edge (agent loop)
#   P3: 5 nodes, 1 conditional edge (pipeline + loop)
#   P4: 7 nodes, 2 conditional edges (pipeline + step loop + error recovery)
#   P5: 7 nodes (4 own + 3 sub-graphs), 1 conditional edge (four-way fork)
#       But each sub-graph has 3 internal nodes, so total: 7 + 9 = 16 nodes!
# ---------------------------------------------------------------------------

def build_support_bot(checkpointer=None):
    """Build and compile the customer support bot graph.

    Args:
        checkpointer: required for interrupt() to work. Use MemorySaver()
                      for in-memory checkpointing.
    """
    graph = StateGraph(SupportState)

    # --- Parent's own nodes ---
    graph.add_node("classify_issue", classify_issue)
    graph.add_node("fallback_response", fallback_response)
    graph.add_node("collect_feedback", collect_feedback)

    # --- Sub-graphs as nodes ---
    # Each compiled sub-graph acts as a single node in the parent.
    # LangGraph reads their input/output schemas to know what to pass in/out.
    graph.add_node("billing_subgraph", build_billing_subgraph())
    graph.add_node("technical_subgraph", build_technical_subgraph())
    graph.add_node("returns_subgraph", build_returns_subgraph())

    # --- Entry point ---
    graph.set_entry_point("classify_issue")

    # --- Conditional edge: classify → department ---
    graph.add_conditional_edges("classify_issue", route_to_department)

    # --- All departments converge to collect_feedback ---
    graph.add_edge("billing_subgraph", "collect_feedback")
    graph.add_edge("technical_subgraph", "collect_feedback")
    graph.add_edge("returns_subgraph", "collect_feedback")
    graph.add_edge("fallback_response", "collect_feedback")

    # --- Feedback → END ---
    graph.add_edge("collect_feedback", END)

    return graph.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# 6. STREAMING LOOP
#
# NEW CONCEPT: Streaming
#
# In P1-P4, we used app.invoke() — wait for EVERYTHING to finish, then
# print the result. The user sees nothing until the graph is done.
#
# With app.stream(), we get updates AS THEY HAPPEN:
#   - stream_mode="updates": shows which node just ran and what it produced
#   - stream_mode="messages": shows individual LLM tokens (chat UX)
#
# We use "updates" mode because it works well with sub-graphs — you see
# each sub-graph node completing, plus the parent nodes.
#
# The tricky part: interrupt() handling.
#
# When the graph hits an interrupt(), stream() yields an Interrupt event
# and stops. We need to:
#   1. Detect the interrupt
#   2. Show the question to the user
#   3. Get their input
#   4. Resume with Command(resume=answer)
#   5. Continue streaming from where we left off
#
# This is fundamentally different from P4's input() loop, which was
# OUTSIDE the graph. Here, the graph itself controls when to ask.
# ---------------------------------------------------------------------------

def handle_support_issue(customer_issue: str) -> dict:
    """Run the support bot for a single customer issue.

    Handles streaming and interrupt() resume automatically.
    Returns the final state.
    """
    checkpointer = MemorySaver()
    app = build_support_bot(checkpointer=checkpointer)

    initial_state: SupportState = {
        "messages": [HumanMessage(content=customer_issue)],
        "customer_issue": customer_issue,
        "category": "",
        "department_response": "",
        "satisfaction_rating": 0,
    }

    # thread_id is required by MemorySaver (same as P4)
    config = {"configurable": {"thread_id": "support-1"}}

    # --- Streaming with interrupt handling ---
    #
    # We use stream_mode="updates" to see node-by-node progress.
    # Each chunk is a dict: {node_name: state_update}
    #
    # When an interrupt occurs, the stream ends and app.get_state()
    # shows the pending interrupt. We then resume with Command(resume=...).

    final_state = _stream_with_interrupts(app, initial_state, config)
    return final_state


def _stream_with_interrupts(app, input_value, config) -> dict:
    """Stream the graph, handling any interrupt() calls along the way.

    This is a loop because there can be MULTIPLE interrupts:
      1. Technical sub-graph interrupts for OS info
      2. collect_feedback interrupts for satisfaction rating

    Each time we hit an interrupt:
      - Show the interrupt message (the question)
      - Get user input
      - Resume with Command(resume=answer)
      - Continue streaming

    The loop ends when the graph reaches END with no more interrupts.
    """
    current_input = input_value
    final_state = None

    while True:
        # Stream until we hit an interrupt or the graph ends
        print()
        for chunk in app.stream(current_input, config=config, stream_mode="updates"):
            # Each chunk: {node_name: state_update_dict}
            for node_name, update in chunk.items():
                # Show progress — which node just completed
                if node_name == "__interrupt__":
                    # This is the interrupt signal, handled below
                    continue
                print(f"  [{node_name}] completed")

        # Check if there's a pending interrupt
        state_snapshot = app.get_state(config)

        if state_snapshot.next:
            # Graph is paused at an interrupt — there are still nodes to run
            # The interrupt value is in the state snapshot's tasks
            for task in state_snapshot.tasks:
                if hasattr(task, "interrupts") and task.interrupts:
                    interrupt_value = task.interrupts[0].value
                    print(f"\n  >> {interrupt_value}")
                    user_input = input("  >> Your answer: ").strip()

                    # Resume the graph with the user's answer
                    current_input = Command(resume=user_input)
                    break
        else:
            # Graph is done — no more nodes to run
            final_state = state_snapshot.values
            break

    return final_state


def interactive_loop() -> None:
    """Run the support bot in interactive mode.

    The user types issues, the bot classifies, routes, and responds.
    Type 'quit' to exit.
    """
    print(f"\n{'='*60}")
    print("  Customer Support Bot — Project 5")
    print(f"{'='*60}")
    print("\nDescribe your issue and I'll route you to the right department.")
    print("Type 'quit' to exit.\n")

    while True:
        issue = input("Your issue: ").strip()

        if not issue:
            continue
        if issue.lower() in ("quit", "exit", "q"):
            print("\nThank you for contacting support. Goodbye!")
            break

        try:
            result = handle_support_issue(issue)

            # Show final summary
            print(f"\n{'─'*40}")
            print(f"  Category: {result.get('category', 'N/A')}")
            print(f"  Rating: {result.get('satisfaction_rating', 'N/A')}/5")
            print(f"{'─'*40}\n")

        except Exception as e:
            print(f"\nError: {e}\n")


def single_issue(issue: str) -> None:
    """Run a single support issue (non-interactive mode).

    Useful for testing from environments that don't support input().
    """
    print(f"\nIssue: {issue}\n")

    result = handle_support_issue(issue)

    print(f"\n{'─'*40}")
    print(f"  Category: {result.get('category', 'N/A')}")
    print(f"  Response: {result.get('department_response', 'N/A')}")
    print(f"  Rating: {result.get('satisfaction_rating', 'N/A')}/5")
    print(f"{'─'*40}")
