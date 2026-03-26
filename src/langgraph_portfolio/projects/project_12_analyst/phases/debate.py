"""Phase 3: Bull/Bear Debate — AG2 GroupChat.

CONCEPT: AG2 inside a LangGraph node (from P7)
The parent graph calls run_debate(state) as a node. Inside, we build
3 AG2 ConversableAgents and run a GroupChat with custom speaker selection.

The debate runs for 3 rounds:
  Round 1: Bull opens -> Bear responds
  Round 2: Bull rebuts -> Bear rebuts
  Round 3: Bull closing -> Bear closing
Then the Moderator synthesizes key disagreements.

All financial data and analysis results are injected into agent system
prompts so they argue with evidence, not generalities.
"""

from __future__ import annotations

import re

from autogen import ConversableAgent, GroupChat, GroupChatManager

from langgraph_portfolio.projects.project_12_analyst.llm import get_ag2_llm_config
from langgraph_portfolio.projects.project_12_analyst.state import InvestmentState


MAX_ROUNDS = 10  # 3 rounds * 2 debaters + moderator open + moderator close + buffer


def build_debate_agents(
    data_summary: str,
    analysis_summary: str,
) -> list[ConversableAgent]:
    """Build the 3 debate agents with financial context in system prompts."""
    llm_config = get_ag2_llm_config()

    bull = ConversableAgent(
        name="Bull_Analyst",
        system_message=(
            "You are a senior equity analyst arguing IN FAVOR of buying this stock.\n\n"
            "You have access to the following research:\n"
            f"--- DATA SUMMARY ---\n{data_summary[:4000]}\n\n"
            f"--- ANALYSIS ---\n{analysis_summary[:4000]}\n\n"
            "Rules:\n"
            "- Present clear arguments backed by specific numbers from the data\n"
            "- Address and rebut the Bear's points directly\n"
            "- Acknowledge risks but explain why they're manageable\n"
            "- Keep each response to 2-3 focused paragraphs\n"
        ),
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    bear = ConversableAgent(
        name="Bear_Analyst",
        system_message=(
            "You are a senior equity analyst arguing AGAINST buying this stock.\n\n"
            "You have access to the following research:\n"
            f"--- DATA SUMMARY ---\n{data_summary[:4000]}\n\n"
            f"--- ANALYSIS ---\n{analysis_summary[:4000]}\n\n"
            "Rules:\n"
            "- Find weaknesses, risks, and overvaluation signals in the data\n"
            "- Challenge the Bull's assumptions with specific counter-evidence\n"
            "- Highlight what could go wrong and downside scenarios\n"
            "- Keep each response to 2-3 focused paragraphs\n"
        ),
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    moderator = ConversableAgent(
        name="Moderator",
        system_message=(
            "You are an investment committee moderator. Your job:\n"
            "1. Open the debate with a brief framing of the investment question\n"
            "2. After 3 rounds, synthesize the KEY DISAGREEMENTS between Bull and Bear\n"
            "3. List each disagreement as a bullet point with both sides' positions\n"
            "4. End your final summary with the word DEBATE_OVER on its own line\n\n"
            "Do NOT take sides. Be neutral and precise.\n"
        ),
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    return [bull, bear, moderator]


def _select_next_speaker(
    last_speaker: ConversableAgent,
    groupchat: GroupChat,
) -> ConversableAgent:
    """Custom speaker selection: Moderator opens, then Bull/Bear alternate,
    Moderator closes."""
    agent_map = {a.name: a for a in groupchat.agents}
    n = len(groupchat.messages)

    # Moderator opens
    if n <= 1:
        return agent_map["Moderator"]

    # After 7 messages (open + 3 rounds of 2), Moderator summarizes
    if n >= 7:
        return agent_map["Moderator"]

    # During debate: alternate Bull -> Bear
    if last_speaker.name == "Moderator":
        return agent_map["Bull_Analyst"]
    elif last_speaker.name == "Bull_Analyst":
        return agent_map["Bear_Analyst"]
    elif last_speaker.name == "Bear_Analyst":
        return agent_map["Bull_Analyst"]

    return agent_map["Moderator"]


def _is_debate_over(msg: dict) -> bool:
    """Check if the Moderator has ended the debate."""
    content = msg.get("content", "") or ""
    return "DEBATE_OVER" in content


def _extract_key_disagreements(messages: list[dict]) -> list[str]:
    """Extract key disagreements from the Moderator's final summary."""
    for msg in reversed(messages):
        if msg.get("name") == "Moderator" and "DEBATE_OVER" in (msg.get("content") or ""):
            content = msg["content"]
            # Extract bullet points
            bullets = re.findall(r"[-\u2022]\s*(.+)", content)
            if bullets:
                return bullets
            # Fallback: return the whole summary
            return [content.replace("DEBATE_OVER", "").strip()]
    return ["No disagreements extracted"]


def run_debate(state: InvestmentState) -> dict:
    """LangGraph node: run the AG2 bull/bear debate.

    Takes the data_summary and analysis_summary from previous phases
    and feeds them to the debate agents as context.
    """
    data_summary = state.get("data_summary", "")
    analysis_summary = state.get("analysis_summary", "")
    ticker = state["ticker"]

    # Build agents with financial context
    agents = build_debate_agents(data_summary, analysis_summary)
    bull, bear, moderator = agents

    # Apply termination check
    for agent in agents:
        agent._is_termination_msg = _is_debate_over

    # Build GroupChat
    group_chat = GroupChat(
        agents=[moderator, bull, bear],
        messages=[],
        max_round=MAX_ROUNDS,
        speaker_selection_method=_select_next_speaker,
        allow_repeat_speaker=False,
    )

    manager = GroupChatManager(
        groupchat=group_chat,
        llm_config=get_ag2_llm_config(),
    )

    # Run the debate
    moderator.initiate_chat(
        manager,
        message=(
            f"The investment committee is evaluating {ticker}. "
            f"Bull_Analyst, please open with your investment case."
        ),
    )

    # Collect results
    messages = group_chat.messages
    transcript_parts = []
    for msg in messages:
        name = msg.get("name", "Unknown")
        content = msg.get("content", "")
        transcript_parts.append(f"**{name}:**\n{content}\n")

    debate_transcript = "\n---\n\n".join(transcript_parts)
    key_disagreements = _extract_key_disagreements(messages)

    return {
        "debate_transcript": debate_transcript,
        "key_disagreements": key_disagreements,
        "current_phase": "thesis",
    }
