"""
Project 3 — Writing Team using real LangGraph.

We build this in layers (same approach as projects 1 and 2):
  1. State definition
  2. Node functions (5 agents, each with a role)
  3. Routing logic (editor quality-control loop)
  4. Graph assembly
"""

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import add_messages


# ---------------------------------------------------------------------------
# 1. STATE
#
# In P1 (chatbot), the state held: messages, intent, confidence
# In P2 (research agent), the state held: messages, documents, citations
#
# Here we need to hold the WORK PRODUCTS of each agent. Think of it like
# a shared desk that everyone on the team can see:
#
#   - The planner puts an outline on the desk
#   - The researcher adds research notes next to it
#   - The writer reads both and puts a draft on the desk
#   - The editor reads the draft and puts feedback on the desk
#   - The writer reads the feedback and revises (loop!)
#   - The publisher takes the final draft and formats it
#
# Each field below represents one of those work products.
#
# New field: revision_count
#   This prevents infinite loops. Without it, the editor could keep
#   rejecting and the writer could keep rewriting forever. We cap it
#   (e.g., max 2 revision rounds) and then ship whatever we have.
#
# New field: editor_approved
#   A boolean flag. The editor sets this to True ("looks good, publish it")
#   or False ("needs more work, send it back to the writer").
#   The routing function reads this to decide where to go next.
# ---------------------------------------------------------------------------

class WritingState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # full conversation history
    topic: str                # what we're writing about
    outline: str              # planner's output — structured outline
    research_notes: str       # researcher's output — gathered information
    draft: str                # writer's output — the article draft
    editor_feedback: str      # editor's output — review comments
    editor_approved: bool     # editor's decision — ready to publish?
    revision_count: int       # how many times the editor sent it back
    final_article: str        # publisher's output — the finished piece


# ---------------------------------------------------------------------------
# 2. NODES — THE FIVE AGENTS
#
# Each node follows the same pattern:
#   1. Create a SystemMessage with the agent's role and instructions
#   2. Create a HumanMessage with the actual work to do (pulled from state)
#   3. Call the LLM
#   4. Return state updates with the agent's output
#
# Why SystemMessage + HumanMessage?
#
#   The SystemMessage is like a job description pinned to the wall:
#     "You are a planner. Your job is to..."
#   The HumanMessage is like the actual task handed to that person:
#     "Here's the topic: LangGraph vs CrewAI. Create an outline."
#
#   We build these fresh in each node (not using state["messages"]) because
#   each agent needs a FOCUSED prompt — not the full conversation history.
#   A writer doesn't need to see the planner's internal thinking.
#   They just need the outline and research notes.
#
# Why not use state["messages"] for everything?
#
#   In P2, we passed the full message history to the agent because it was
#   ONE LLM making decisions across the whole conversation. Here, each
#   agent is a SPECIALIST. If we passed the full history, the writer would
#   see the planner's system prompt saying "you are a planner" — confusing!
#   Instead, each node builds its own focused prompt.
#
#   But we still SAVE important outputs to state["messages"] so there's
#   a record of what happened at each step.
# ---------------------------------------------------------------------------

# Same LLM as previous projects, shared by all agents.
# Each agent gets a different personality through its system prompt.
llm = ChatOllama(model="qwen3.5:35b", temperature=0)


def planner(state: WritingState) -> dict:
    """Agent 1: The Planner — creates an outline for the article.

    Reads: topic (from initial input)
    Produces: outline (a structured plan for the article)
    """
    # The system message defines WHO this agent is
    system = SystemMessage(content=(
        "You are an article planner. Given a topic, create a clear, structured "
        "outline with 3-4 sections. Each section should have a title and a "
        "one-sentence description of what it will cover.\n\n"
        "Output ONLY the outline, nothing else."
    ))

    # The human message defines WHAT to do right now
    human = HumanMessage(content=f"Create an outline for an article about: {state['topic']}")

    response = llm.invoke([system, human])

    # Save the outline to state AND record it in messages
    return {
        "outline": response.content,
        "messages": [AIMessage(content=f"[Planner] Outline created:\n{response.content}")],
    }


def researcher(state: WritingState) -> dict:
    """Agent 2: The Researcher — gathers information for each section.

    Reads: outline (from planner)
    Produces: research_notes (background info for the writer)

    Note: In a real app, this agent would use tools (like search_documents
    from P2) to find actual information. For simplicity, we're having it
    generate research notes from its training knowledge. You could add
    tool calling here later — you already know how from P2!
    """
    system = SystemMessage(content=(
        "You are a researcher. Given an article outline, provide brief "
        "research notes for each section — key facts, important points, "
        "and any relevant comparisons.\n\n"
        "Keep notes concise: 2-3 bullet points per section."
    ))

    human = HumanMessage(content=f"Research the following outline:\n\n{state['outline']}")

    response = llm.invoke([system, human])

    return {
        "research_notes": response.content,
        "messages": [AIMessage(content=f"[Researcher] Notes gathered:\n{response.content}")],
    }


def writer(state: WritingState) -> dict:
    """Agent 3: The Writer — drafts (or revises) the article.

    Reads: outline, research_notes, and optionally editor_feedback
    Produces: draft

    This node runs at LEAST once (initial draft) and possibly more times
    (revisions). On the first pass, there's no feedback. On revision
    passes, the writer reads the editor's feedback and incorporates it.

    This is the node that the editor loop sends work BACK to.
    """
    # Build the prompt differently depending on whether this is
    # the first draft or a revision
    if state.get("editor_feedback"):
        # REVISION — the editor already gave feedback
        system = SystemMessage(content=(
            "You are a writer revising an article. You previously wrote a draft "
            "and received feedback from an editor. Incorporate the feedback to "
            "improve the article.\n\n"
            "Output ONLY the revised article, nothing else."
        ))
        human = HumanMessage(content=(
            f"OUTLINE:\n{state['outline']}\n\n"
            f"RESEARCH NOTES:\n{state['research_notes']}\n\n"
            f"YOUR PREVIOUS DRAFT:\n{state['draft']}\n\n"
            f"EDITOR FEEDBACK:\n{state['editor_feedback']}\n\n"
            f"Please revise the draft based on the feedback."
        ))
    else:
        # FIRST DRAFT — no feedback yet
        system = SystemMessage(content=(
            "You are a writer creating an article. Use the outline as your "
            "structure and the research notes as your source material.\n\n"
            "Write a clear, well-structured article. Output ONLY the article."
        ))
        human = HumanMessage(content=(
            f"OUTLINE:\n{state['outline']}\n\n"
            f"RESEARCH NOTES:\n{state['research_notes']}\n\n"
            f"Write the article."
        ))

    response = llm.invoke([system, human])

    revision = state.get("revision_count", 0)

    # If this is a revision (editor_feedback exists), bump the counter.
    # First draft → revision_count stays at 0
    # First revision → revision_count becomes 1
    # Second revision → revision_count becomes 2 (will hit MAX_REVISIONS)
    if state.get("editor_feedback"):
        revision += 1

    label = "Draft" if revision == 0 else f"Revision {revision}"

    return {
        "draft": response.content,
        "revision_count": revision,
        "messages": [AIMessage(content=f"[Writer] {label} complete.")],
    }


def editor(state: WritingState) -> dict:
    """Agent 4: The Editor — reviews the draft and decides: approve or revise.

    Reads: draft, outline (to check if the draft follows the plan)
    Produces: editor_feedback, editor_approved

    This is the DECISION MAKER in the loop. The editor reads the draft
    and makes a judgment call:
      - "This is good enough" → editor_approved = True → publish
      - "This needs work" → editor_approved = False → back to writer

    To make this machine-parseable, we ask the editor to start their
    response with either APPROVED or REVISION_NEEDED. This is similar
    to how we parsed intent/confidence in the chatbot classifier.
    """
    system = SystemMessage(content=(
        "You are an editor reviewing an article draft. Check if it:\n"
        "1. Follows the outline structure\n"
        "2. Is clear and well-written\n"
        "3. Covers the key points\n\n"
        "Start your response with EXACTLY one of these words:\n"
        "APPROVED — if the article is ready to publish\n"
        "REVISION_NEEDED — if the article needs improvements\n\n"
        "Then provide your feedback (what's good, what needs fixing)."
    ))

    human = HumanMessage(content=(
        f"ORIGINAL OUTLINE:\n{state['outline']}\n\n"
        f"DRAFT TO REVIEW:\n{state['draft']}"
    ))

    response = llm.invoke([system, human])

    # Parse the editor's decision from the first word
    text = response.content.strip()
    approved = text.upper().startswith("APPROVED")

    return {
        "editor_feedback": text,
        "editor_approved": approved,
        "messages": [AIMessage(content=f"[Editor] {'Approved' if approved else 'Revision needed'}.")],
    }


def publisher(state: WritingState) -> dict:
    """Agent 5: The Publisher — formats the final article.

    Reads: draft (the editor-approved version)
    Produces: final_article

    This is the simplest agent. It takes the approved draft and does
    final formatting — adding a title, cleaning up structure, etc.
    It always runs exactly once, right before END.
    """
    system = SystemMessage(content=(
        "You are a publisher. Take the article draft and produce a clean, "
        "final version with proper formatting. Add a title if missing. "
        "Do not change the content, only improve formatting.\n\n"
        "Output ONLY the final article."
    ))

    human = HumanMessage(content=f"Format this article for publication:\n\n{state['draft']}")

    response = llm.invoke([system, human])

    return {
        "final_article": response.content,
        "messages": [AIMessage(content=f"[Publisher] Article published.")],
    }


# ---------------------------------------------------------------------------
# 3. ROUTING — THE EDITOR LOOP
#
# Comparing the routing across all three projects:
#
# P1 Chatbot:
#   Check: confidence >= 0.7?
#   YES → respond    NO → clarify
#   Shape: a fork (two paths, no loops)
#
# P2 Research Agent:
#   Check: tool_calls present?
#   YES → tools → agent (loop)    NO → END
#   Shape: a loop that the SAME agent controls
#
# P3 Writing Team:
#   Check: editor_approved? AND revision_count < max?
#   APPROVED → publisher    NOT APPROVED + under limit → writer (loop)
#                           NOT APPROVED + over limit → publisher (force ship)
#   Shape: a loop controlled by a DIFFERENT agent than the one doing the work
#
# That last point is key: in P2, the agent decided for ITSELF when to stop
# looping. Here, the EDITOR decides whether the WRITER needs to redo work.
# One agent judging another agent's work — that's multi-agent coordination.
#
# MAX_REVISIONS is our safety net. Without it, a picky editor could send
# work back forever. In real apps, you'd also track things like time spent
# or quality scores to decide when to stop.
# ---------------------------------------------------------------------------

MAX_REVISIONS = 2


def after_editor(state: WritingState) -> str:
    """Decide: publish the article, or send it back for revision?

    This is called after the editor node runs. Three possible outcomes:

    1. Editor approved → go to publisher (we're done!)
    2. Editor rejected + under revision limit → go back to writer (revise)
    3. Editor rejected + hit revision limit → go to publisher anyway (ship it)
    """
    if state["editor_approved"]:
        # Editor says it's good — move to publishing
        return "publisher"

    if state["revision_count"] < MAX_REVISIONS:
        # Editor wants changes and we haven't hit the limit — revise
        return "writer"

    # Editor still not happy, but we've revised enough — ship it
    return "publisher"


# ---------------------------------------------------------------------------
# 4. GRAPH ASSEMBLY
#
# Let's compare the wiring across all three projects:
#
# P1 Chatbot (fork shape):
#   START → classify → [conditional] → respond → END
#                                     → clarify → END
#   3 nodes, 1 conditional edge, 2 regular edges
#
# P2 Research Agent (loop shape):
#   START → agent → [conditional] → tools → agent (loop back)
#                                 → END
#   2 nodes, 1 conditional edge, 1 regular edge
#
# P3 Writing Team (pipeline + loop shape):
#   START → planner → researcher → writer → editor → [conditional] → publisher → END
#                                    ↑                              → writer (loop back)
#   5 nodes, 1 conditional edge, 5 regular edges
#
# The new thing here is the PIPELINE portion. In P1 and P2, we had at most
# 2-3 nodes in sequence. Here we have 4 nodes in a straight line before
# hitting the conditional edge. Each regular edge (add_edge) is just saying
# "after this node, always go to that node."
#
# The conditional edge is only on the editor — that's the only decision
# point in the graph. Everything else flows in a fixed order.
# ---------------------------------------------------------------------------

from langgraph.graph import END, StateGraph


def build_writing_team():
    """Build and compile the writing team graph."""

    graph = StateGraph(WritingState)

    # --- Add all 5 nodes ---
    graph.add_node("planner", planner)
    graph.add_node("researcher", researcher)
    graph.add_node("writer", writer)
    graph.add_node("editor", editor)
    graph.add_node("publisher", publisher)

    # --- Set the starting point ---
    graph.set_entry_point("planner")

    # --- The pipeline: planner → researcher → writer → editor ---
    # These are regular edges: "after A, always go to B"
    # No decisions, no conditions — just a straight line.
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "writer")
    graph.add_edge("writer", "editor")

    # --- The decision point: after editor, call after_editor ---
    # This is the only conditional edge in the whole graph.
    # after_editor returns "publisher" or "writer" (the loop).
    graph.add_conditional_edges("editor", after_editor)

    # --- After publisher, we're done ---
    graph.add_edge("publisher", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# 5. RUN IT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = build_writing_team()

    result = app.invoke({
        "messages": [],
        "topic": "LangGraph versus CrewAI for building AI agent teams",
        "outline": "",
        "research_notes": "",
        "draft": "",
        "editor_feedback": "",
        "editor_approved": False,
        "revision_count": 0,
        "final_article": "",
    })

    # Show what happened at each step
    print("\n=== Writing Team Trace ===\n")
    for msg in result["messages"]:
        print(msg.content)
        print()

    # Show the final article
    print("=== Final Article ===\n")
    print(result["final_article"])
