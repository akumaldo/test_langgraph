"""
Project 7 — Debate Arena (AG2 GroupChat).

This is the CORE of P7 — the GroupChat that brings all agents together.

FRAMEWORK COMPARISON — How multi-agent coordination works:

  LangGraph (P3, P5):
    - You build a GRAPH with nodes and edges
    - YOU decide the execution order (deterministic)
    - Conditional edges let you branch based on state
    - Example: graph.add_edge("writer", "editor")

  CrewAI (P3, P6):
    - You build a CREW with agents and tasks
    - Process.sequential: tasks run in order
    - Process.hierarchical: a manager LLM delegates tasks
    - Example: Crew(agents=[...], tasks=[...], process=Process.sequential)

  AG2 (P7 — this project):
    - You build a GROUP CHAT with agents
    - A GroupChatManager LLM decides who speaks next
    - Agents respond to each other's messages
    - Example: GroupChat(agents=[...], max_round=12)

The key insight: LangGraph is GRAPH-first, CrewAI is TASK-first,
AG2 is CONVERSATION-first. Each model fits different use cases:
    - Graph: complex workflows with branches and loops
    - Task: divide-and-conquer with clear deliverables
    - Conversation: dynamic discussions where the flow emerges

A debate is NATURALLY a conversation, so AG2 fits perfectly.

GROUP CHAT MECHANICS:

When you call manager.initiate_chat(), here's what happens internally:
  1. The initial message is sent to the group
  2. The GroupChatManager looks at ALL agents and the conversation so far
  3. The manager's LLM decides: "who should speak next?"
  4. That agent generates a response (using its system_message + conversation history)
  5. The response is added to the shared message list
  6. Back to step 2 — until max_round or termination

This is fundamentally different from LangGraph where YOU wire the edges,
or CrewAI where tasks have a fixed order. Here, the MANAGER LLM decides
the flow dynamically, based on the conversation content.

SPEAKER SELECTION STRATEGIES:

The GroupChatManager can pick the next speaker in several ways:
  - "auto" (default): the manager LLM chooses based on conversation context
  - "round_robin": fixed rotation (A → B → C → A → B → C)
  - "random": random selection each turn
  - custom function: you write the selection logic

We use a CUSTOM FUNCTION for this debate. Why?

  "auto" (LLM-driven) is unpredictable — the manager might let one debater
  speak twice in a row, or skip the moderator entirely. With Ollama/qwen,
  the selection is even less reliable than with GPT-4.

  A custom function gives us the STRUCTURE of a real debate:
    Moderator → Pro → Con → Moderator → Pro → Con → ... → Judge

  This is like LangGraph's explicit edges, but expressed as a function
  instead of graph.add_edge() calls. Same control, different syntax.

  Think of it as a spectrum:
    round_robin = fully deterministic (no intelligence)
    custom func = structured but flexible (our choice)
    auto        = fully LLM-driven (unreliable with Ollama)
"""

import re

from autogen import ConversableAgent, GroupChat, GroupChatManager

from .agents import (
    LLM_CONFIG,
    build_debater_con,
    build_debater_pro,
    build_judge,
    build_moderator,
)
from .models import DebateResult, DebaterScore


# ---------------------------------------------------------------------------
# TERMINATION
#
# How does the debate end? In each framework:
#
#   LangGraph: you route to END explicitly
#       graph.add_conditional_edges("check", {"done": END, "continue": "next"})
#
#   CrewAI: the last task completes → crew is done
#
#   AG2: MESSAGE-BASED termination
#       When any agent sees "DEBATE_OVER" in a message, the chat stops.
#       OR when max_round is reached (safety cap, like P3's MAX_REVISIONS).
#
# We use BOTH:
#   - The Judge says "DEBATE_OVER" → content-based termination
#   - max_round=14 → safety cap in case the Judge forgets the magic word
#
# The is_termination_msg function is checked AFTER every message. If it
# returns True, the GroupChat stops immediately. We apply it to all agents.
# ---------------------------------------------------------------------------

MAX_ROUNDS = 14  # Safety cap — debate ends even if Judge forgets DEBATE_OVER


def is_debate_over(msg: dict) -> bool:
    """Check if the debate should end.

    This function is passed to every agent via is_termination_msg.
    After every message in the GroupChat, AG2 calls this function.
    If it returns True, the conversation stops.

    Compare to:
      LangGraph: conditional edge that routes to END
      CrewAI:    implicit (last task finishes)
      AG2:       is_termination_msg (message content check)
    """
    content = msg.get("content", "") or ""
    return "DEBATE_OVER" in content


# ---------------------------------------------------------------------------
# SPEAKER SELECTION
#
# This is the custom function that controls WHO SPEAKS NEXT.
#
# AG2's GroupChatManager calls this function after every message.
# It receives the last speaker and the list of all agents, and returns
# the next speaker.
#
# Our debate structure:
#   Round 1: Moderator opens → Pro argues → Con argues
#   Round 2: Moderator summarizes → Pro rebuts → Con rebuts
#   Round 3: Moderator asks for final statements → Pro closes → Con closes
#   Final:   Moderator concludes → Judge scores
#
# We track rounds using the conversation length (number of messages).
#
# Compare to LangGraph's conditional edges:
#   In LangGraph, you'd write:
#       def route(state): return "pro" if state["turn"] == "pro" else "con"
#       graph.add_conditional_edges("moderator", route)
#
#   In AG2, the same logic lives in this function.
#   Same control, different mechanism.
# ---------------------------------------------------------------------------


def select_next_speaker(
    last_speaker: ConversableAgent,
    groupchat: GroupChat,
) -> ConversableAgent:
    """Decide who speaks next in the debate.

    This implements a structured debate format:
      Moderator → Pro → Con → Moderator → Pro → Con → ... → Judge

    The Moderator opens, then debaters alternate, with the Moderator
    stepping in periodically to guide the discussion. After enough
    rounds, the Moderator wraps up and the Judge scores.
    """
    agents = groupchat.agents
    # Build a name→agent lookup for readability
    agent_map = {agent.name: agent for agent in agents}
    moderator = agent_map["Moderator"]
    pro = agent_map["Debater_Pro"]
    con = agent_map["Debater_Con"]
    judge = agent_map["Judge"]

    n_messages = len(groupchat.messages)

    # The debate flow based on message count:
    #
    # msg 0: (initial message — the user/system prompt to start)
    # msg 1: Moderator opens the debate
    # msg 2: Pro makes first argument
    # msg 3: Con makes first argument
    # msg 4: Moderator guides round 2
    # msg 5: Pro rebuts
    # msg 6: Con rebuts
    # msg 7: Moderator asks for final statements
    # msg 8: Pro closing statement
    # msg 9: Con closing statement
    # msg 10: Moderator concludes ("Judge, please score")
    # msg 11: Judge scores and says DEBATE_OVER

    # Define the speaker order as a repeating pattern
    # Every 3 messages: Moderator → Pro → Con
    # After enough rounds, switch to Judge

    if n_messages <= 1:
        # First message — Moderator opens
        return moderator
    elif n_messages >= 10:
        # After enough rounds, Moderator concludes then Judge scores
        if last_speaker.name == "Moderator":
            return judge
        return moderator
    else:
        # During the debate: Moderator → Pro → Con → repeat
        if last_speaker.name == "Moderator":
            return pro
        elif last_speaker.name == "Debater_Pro":
            return con
        elif last_speaker.name == "Debater_Con":
            return moderator
        else:
            # Fallback
            return moderator


# ---------------------------------------------------------------------------
# DEBATE RUNNER
#
# This function assembles the GroupChat and runs the debate.
#
# Compare to how we start things in other frameworks:
#   LangGraph: result = graph.invoke({"messages": [...]})
#   CrewAI:    result = crew.kickoff(inputs={...})
#   AG2:       result = manager.initiate_chat(agent, message="...")
#
# In LangGraph and CrewAI, you invoke/kickoff the SYSTEM.
# In AG2, you initiate_chat through ONE agent — that agent sends the
# first message, and the GroupChat takes over from there.
# ---------------------------------------------------------------------------


def run_debate(topic: str) -> DebateResult:
    """Run a full debate on the given topic.

    Creates agents, assembles the GroupChat, runs the debate,
    and returns a structured DebateResult.
    """
    print(f"\n  Setting up debate on: '{topic}'")
    print(f"  Max rounds: {MAX_ROUNDS}\n")

    # --- Create agents ---
    # Each agent gets is_termination_msg so ANY agent seeing "DEBATE_OVER" stops the chat
    pro = build_debater_pro(topic)
    con = build_debater_con(topic)
    moderator = build_moderator(topic)
    judge = build_judge(topic)

    # Apply termination check to all agents
    for agent in [pro, con, moderator, judge]:
        agent._is_termination_msg = is_debate_over

    # --- Assemble GroupChat ---
    #
    # GroupChat is AG2's unique concept — no equivalent in LangGraph or CrewAI.
    # It's a shared conversation space where multiple agents participate.
    #
    # Parameters:
    #   agents: who participates (like CrewAI's agents list)
    #   max_round: safety cap (like P3's MAX_REVISIONS)
    #   speaker_selection_method: who speaks next
    #     - "auto": LLM decides (unreliable with Ollama)
    #     - "round_robin": fixed rotation
    #     - custom function: our choice — structured debate flow
    #   allow_repeat_speaker: can the same agent speak twice in a row?

    group_chat = GroupChat(
        agents=[moderator, pro, con, judge],
        messages=[],
        max_round=MAX_ROUNDS,
        speaker_selection_method=select_next_speaker,
        allow_repeat_speaker=False,  # no agent speaks twice in a row
    )

    # --- Create GroupChatManager ---
    #
    # The manager is a special agent that ORCHESTRATES the group chat.
    # It doesn't participate in the debate — it just manages turns.
    #
    # Compare to:
    #   LangGraph: the graph engine itself (invisible, manages execution)
    #   CrewAI hierarchical: the manager agent (visible, delegates tasks)
    #   AG2 GroupChatManager: manages who speaks next (visible but silent)
    #
    # The manager needs its own LLM config because with "auto" speaker
    # selection, it uses the LLM to decide who speaks next. With our
    # custom function, the LLM isn't used for selection — but AG2
    # still requires the config.

    manager = GroupChatManager(
        groupchat=group_chat,
        llm_config=LLM_CONFIG,
    )

    # --- Run the debate ---
    #
    # initiate_chat() starts the conversation. The first agent (moderator)
    # receives the initial message and responds. Then the GroupChat takes
    # over, calling select_next_speaker after each response.
    #
    # The entire debate runs synchronously — this call blocks until
    # DEBATE_OVER is triggered or max_round is reached.

    print("=" * 60)
    print("  DEBATE BEGINS")
    print("=" * 60 + "\n")

    moderator.initiate_chat(
        manager,
        message=f"The debate topic is: '{topic}'. Please open the debate.",
    )

    print("\n" + "=" * 60)
    print("  DEBATE ENDED")
    print("=" * 60 + "\n")

    # --- Parse results ---
    # Extract the debate history and parse the Judge's scoring
    messages = group_chat.messages
    result = _parse_debate_result(topic, messages)

    return result


def _parse_debate_result(topic: str, messages: list[dict]) -> DebateResult:
    """Parse the GroupChat messages into a structured DebateResult.

    AG2 stores all messages as dicts with 'name' and 'content' keys.
    We look for the Judge's final message and extract scores.

    This is manual parsing — unlike LangGraph's with_structured_output()
    which guarantees the schema. With AG2, we get natural text and need
    to extract structure ourselves.
    """
    # Find the Judge's message (should be the last or second-to-last)
    judge_msg = ""
    for msg in reversed(messages):
        if msg.get("name") == "Judge":
            judge_msg = msg.get("content", "")
            break

    # Try to extract scores from the Judge's message
    scores = _extract_scores(judge_msg)

    # Determine winner
    winner = "Draw"
    if scores:
        winner = max(scores, key=lambda s: s.score).name

    return DebateResult(
        topic=topic,
        scores=scores,
        winner=winner,
        summary=judge_msg if judge_msg else "Debate completed (no judge scoring found).",
        total_rounds=len(messages),
    )


def _extract_scores(judge_text: str) -> list[DebaterScore]:
    """Extract debater scores from the Judge's text.

    We look for patterns like:
      **Debater_Pro**: 7/10 — reasoning
      **Debater_Con**: 8/10 — reasoning

    This is best-effort parsing. If the LLM doesn't follow the format
    exactly, we return empty scores rather than crashing.
    """
    scores = []

    # Pattern: **Name**: score/10 — reasoning
    # Also match without bold markers, with various separators, and decimal scores
    # (the LLM might give 8.5/10 — we round to nearest int)
    pattern = r"\*{0,2}(Debater_(?:Pro|Con))\*{0,2}\s*:?\s*(\d+(?:\.\d+)?)\s*/\s*10\s*[—\-–:]\s*(.+)"

    for match in re.finditer(pattern, judge_text, re.IGNORECASE):
        name = match.group(1)
        score = round(float(match.group(2)))  # round decimals like 8.5 → 9
        reasoning = match.group(3).strip()

        scores.append(DebaterScore(
            name=name,
            score=min(max(score, 1), 10),  # clamp to 1-10
            reasoning=reasoning,
        ))

    return scores
