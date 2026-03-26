"""
Project 7 — Agent Definitions (AG2 ConversableAgent).

This file defines the four debate participants:
  1. Debater_Pro  — argues IN FAVOR of the topic
  2. Debater_Con  — argues AGAINST the topic
  3. Moderator    — controls the debate flow
  4. Judge        — scores arguments and declares a winner

FRAMEWORK COMPARISON — How "agents" differ across frameworks:

  LangGraph:
    - Agents are FUNCTIONS that take state and return updates
    - You control the flow with edges (who runs after whom)
    - The "agent" is just a node in a graph
    - Example: def research_node(state): return {"messages": [...]}

  CrewAI:
    - Agents are OBJECTS with role, goal, backstory
    - You assign them tasks; a process (sequential/hierarchical) controls order
    - The agent decides HOW to complete its task (autonomous)
    - Example: Agent(role="Researcher", goal="Find facts", tools=[...])

  AG2:
    - Agents are CONVERSATIONAL PARTICIPANTS
    - They have a name and a system_message (personality)
    - They respond to messages from other agents
    - The framework manages the back-and-forth
    - Example: ConversableAgent(name="Debater", system_message="Argue for...")

The key insight: AG2 agents don't execute "tasks" — they participate in
CONVERSATIONS. The debate emerges from agents responding to each other,
not from a predefined task list.

OLLAMA CONFIGURATION:

AG2 uses an OpenAI-compatible API format. Ollama exposes an OpenAI-compatible
endpoint at http://localhost:11434/v1, so we can use AG2 directly with Ollama:

    llm_config = {
        "config_list": [{
            "model": "qwen3.5:35b",
            "base_url": "http://localhost:11434/v1",
            "api_key": "ollama",  # required field, but Ollama ignores it
        }]
    }

Compare to how we configured Ollama in other frameworks:
  LangGraph:  ChatOllama(model="qwen3.5:35b")  — LangChain wrapper
  CrewAI:     LLM(model="ollama/qwen3.5:35b")  — LiteLLM prefix
  AG2:        config_list with base_url          — OpenAI-compatible API
"""

from autogen import ConversableAgent


# ---------------------------------------------------------------------------
# LLM CONFIGURATION
#
# AG2 uses a "config_list" pattern — a list of LLM configurations. This
# allows fallback (try model A, if it fails, try model B). We only use
# one model (Ollama), but the list format is required.
#
# The config_list is passed to every agent. In LangGraph, each node can
# use a different LLM. In CrewAI, each agent can have its own LLM. In
# AG2, the config is passed per-agent via llm_config — same flexibility.
# ---------------------------------------------------------------------------

LLM_CONFIG = {
    "config_list": [{
        "model": "qwen3.5:35b",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
    }],
    "temperature": 0.7,
}


def build_debater_pro(topic: str) -> ConversableAgent:
    """Create the PRO debater — argues IN FAVOR of the topic.

    The system_message is the agent's personality and instructions.
    Compare to:
      LangGraph: the system message in ChatPromptTemplate
      CrewAI:    the role + goal + backstory fields

    AG2's system_message is the most direct — it's just a string that
    becomes the system prompt for every LLM call this agent makes.

    human_input_mode="NEVER" means this agent never asks for human input.
    Other options:
      "ALWAYS" — ask human before every response (like P5's interrupt())
      "TERMINATE" — ask human only when termination is triggered
      "NEVER" — fully autonomous (what we want for debaters)
    """
    return ConversableAgent(
        name="Debater_Pro",
        system_message=(
            f"You are a skilled debater arguing IN FAVOR of: '{topic}'.\n\n"
            "Rules:\n"
            "- Present clear, well-structured arguments with evidence\n"
            "- Directly address and rebut the opposing side's points\n"
            "- Be persuasive but respectful\n"
            "- Keep each response to 2-3 paragraphs\n"
            "- When the Moderator asks for final statements, give a brief closing argument\n"
        ),
        llm_config=LLM_CONFIG,
        human_input_mode="NEVER",
    )


def build_debater_con(topic: str) -> ConversableAgent:
    """Create the CON debater — argues AGAINST the topic.

    Same structure as the Pro debater, just a different system_message.
    The LLM is the same — only the instructions differ. This is exactly
    how multi-agent works in every framework: same LLM, different prompts.
    """
    return ConversableAgent(
        name="Debater_Con",
        system_message=(
            f"You are a skilled debater arguing AGAINST: '{topic}'.\n\n"
            "Rules:\n"
            "- Present clear, well-structured counter-arguments with evidence\n"
            "- Directly challenge the opposing side's claims and reasoning\n"
            "- Be critical but fair\n"
            "- Keep each response to 2-3 paragraphs\n"
            "- When the Moderator asks for final statements, give a brief closing argument\n"
        ),
        llm_config=LLM_CONFIG,
        human_input_mode="NEVER",
    )


def build_moderator(topic: str) -> ConversableAgent:
    """Create the Moderator — controls the debate flow.

    The Moderator is like a special node in LangGraph that routes traffic,
    or like CrewAI's hierarchical manager that coordinates agents. But
    instead of routing based on state or delegating tasks, the Moderator
    participates IN the conversation — it speaks as part of the debate.

    The Moderator's job:
      1. Open the debate with the topic
      2. Guide the discussion (keep it on track)
      3. Ask for final statements when it's time to wrap up
      4. Hand off to the Judge for scoring

    In AG2, the Moderator doesn't have special framework powers — it
    influences the debate through its messages, just like a real moderator.
    The GroupChatManager decides who speaks next, not the Moderator.
    """
    return ConversableAgent(
        name="Moderator",
        system_message=(
            f"You are the Moderator of a formal debate on: '{topic}'.\n\n"
            "Your role:\n"
            "1. OPEN the debate by presenting the topic clearly\n"
            "2. After each round of arguments, BRIEFLY summarize key points "
            "and guide the next round (e.g., 'Let's discuss evidence...')\n"
            "3. Keep debaters focused — if they go off-topic, redirect them\n"
            "4. After 2-3 rounds of back-and-forth, ask for FINAL STATEMENTS\n"
            "5. After final statements, say 'The debate is now concluded. "
            "Judge, please provide your scoring.'\n\n"
            "Rules:\n"
            "- Be neutral — never take sides\n"
            "- Keep your interventions brief (2-3 sentences)\n"
            "- Do NOT argue or add your own opinion\n"
        ),
        llm_config=LLM_CONFIG,
        human_input_mode="NEVER",
    )


def build_judge(topic: str) -> ConversableAgent:
    """Create the Judge — scores arguments and declares a winner.

    The Judge speaks LAST. After the Moderator concludes the debate,
    the Judge evaluates both sides and provides:
      - A score (1-10) for each debater
      - Reasoning for each score
      - A winner declaration

    TERMINATION:
    The Judge's response ends with "DEBATE_OVER" — this is the termination
    signal that stops the GroupChat.

    In LangGraph, termination is an edge to END:
        graph.add_edge("judge", END)

    In CrewAI, termination happens when all tasks complete.

    In AG2, termination is MESSAGE-BASED — when an agent says a magic word
    (or when max_round is reached). This is more flexible but less explicit.
    We use is_termination_msg on ALL agents so any agent seeing the Judge's
    "DEBATE_OVER" will stop the conversation.
    """
    return ConversableAgent(
        name="Judge",
        system_message=(
            f"You are the Judge of a formal debate on: '{topic}'.\n\n"
            "You ONLY speak when the Moderator says the debate is concluded.\n"
            "When it's your turn:\n"
            "1. Evaluate both debaters' arguments for:\n"
            "   - Strength of evidence\n"
            "   - Logic and reasoning\n"
            "   - Persuasiveness\n"
            "   - Rebuttal effectiveness\n"
            "2. Give each debater a score from 1-10\n"
            "3. Declare the winner\n\n"
            "Format your response EXACTLY like this:\n"
            "## Scoring\n"
            "**Debater_Pro**: [score]/10 — [brief reasoning]\n"
            "**Debater_Con**: [score]/10 — [brief reasoning]\n\n"
            "## Winner\n"
            "[winner name] wins because [reason]\n\n"
            "End your response with the word DEBATE_OVER on its own line.\n"
        ),
        llm_config=LLM_CONFIG,
        human_input_mode="NEVER",
    )
