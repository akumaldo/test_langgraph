"""
PROJECT 10 — FRAMEWORK SHOWDOWN: AG2 VERSION
==============================================

This is the AG2 implementation of the research assistant.
It reuses patterns from P7 (debate arena):
  - ConversableAgent for participants
  - Message-based termination
  - OpenAI-compatible API for Ollama

KEY CONCEPT — CONVERSATION AS COMPUTATION:
  In LangGraph, the graph IS the program (nodes + edges).
  In CrewAI, the task list IS the program (agents + tasks).
  In AG2, the CONVERSATION IS the program. Agents talk to each other,
  and the orchestration happens through message exchange.

CONTEXT SIZE CONSTRAINT:
  AG2 sends the FULL conversation history to each agent. With GroupChat
  and multiple agents, the context grows fast — too fast for qwen3.5:2b.

  SOLUTION: Use a simple two-agent chat (no GroupChat). The human proxy
  sends the pre-searched KB results, and a single assistant agent
  produces the report. This keeps the context minimal.

  GroupChat was demonstrated in P7 with the larger 35b model. Here we
  show that AG2 also supports simpler patterns when context is limited.

FLOW:
  human_proxy sends message (with truncated KB results)
      ↓
  assistant analyzes and writes JSON report
      ↓
  Termination: "REPORT_COMPLETE" detected
      ↓
  parse_report() → save_result()
"""

import time

from autogen import ConversableAgent

from langgraph_portfolio.projects.project_10_showdown.problem import (
    RESEARCH_QUESTION,
    search_knowledge_base,
    parse_report,
    save_result,
)


# ── MODEL CONFIG ─────────────────────────────────────────────────────────
# AG2 uses OpenAI-compatible API format for Ollama (recap from P7).

# NOTE ON TIMEOUT:
# AG2 0.11.4 does NOT pass 'timeout' from llm_config to the underlying
# OpenAI client. The default httpx timeout (~60s) is too short for
# qwen3.5:2b on local Ollama. We must pass timeout inside config_list
# entries where the OpenAI client picks it up directly.
LLM_CONFIG = {
    "config_list": [
        {
            "model": "qwen3.5:2b",
            "base_url": "http://localhost:11434/v1",
            "api_key": "ollama",
        }
    ],
    "temperature": 0,
    "timeout": 600,  # 10 min — passed to OpenAIWrapper constructor
}


# ── TERMINATION ──────────────────────────────────────────────────────────

def is_termination_msg(message: dict) -> bool:
    """Check if the conversation should end."""
    content = message.get("content", "")
    return "REPORT_COMPLETE" in content.upper() if content else False


# ── AGENTS ───────────────────────────────────────────────────────────────
# Two agents in a simple chat (no GroupChat):
#   - human_proxy: sends the research request (no LLM)
#   - assistant: analyzes documents and produces the JSON report
#
# WHY NOT GROUPCHAT?
#   GroupChat sends the full history to every agent on every turn.
#   With qwen3.5:2b, that causes timeouts. A two-agent chat is simpler
#   and keeps context minimal. P7 used GroupChat with the 35b model
#   where context size wasn't an issue.
#
# THIS IS ITSELF A COMPARISON DATA POINT:
#   AG2's GroupChat pattern is powerful but context-hungry. With small
#   local models, you're pushed toward simpler patterns. Other frameworks
#   (LangGraph, CrewAI) let you control what each step sees.

human_proxy = ConversableAgent(
    name="HumanProxy",
    human_input_mode="NEVER",
    llm_config=False,
    is_termination_msg=is_termination_msg,
    max_consecutive_auto_reply=0,  # don't auto-reply after assistant responds
)

assistant = ConversableAgent(
    name="ResearchAssistant",
    system_message=(
        "You are a research assistant. You will receive documents about AI "
        "orchestration in industry. Analyze them and produce a JSON report "
        "with this exact structure:\n"
        '{"summary": "2-3 sentence overview", '
        '"findings": ["finding 1", "finding 2", ...], '
        '"citations": ["document title - source", ...], '
        '"framework": "ag2"}\n\n'
        "Output the JSON first, then on a new line write REPORT_COMPLETE.\n"
        "Be concise. No other text before the JSON."
    ),
    human_input_mode="NEVER",
    llm_config=LLM_CONFIG,
    is_termination_msg=is_termination_msg,
)


# ── RUN ──────────────────────────────────────────────────────────────────

def run():
    """Run the AG2 research assistant and save results."""
    print("=" * 60)
    print("FRAMEWORK SHOWDOWN — AG2 Version")
    print("=" * 60)
    print(f"\nQuestion: {RESEARCH_QUESTION}\n")

    # Pre-search with HEAVILY truncated documents.
    # AG2 sends the full prompt (system + user message) to the LLM in one call.
    # qwen3.5:2b with thinking mode generates internal reasoning tokens before
    # responding, making large prompts extremely slow (>10 min for 4 docs).
    # We use only 2 docs, truncated to ~400 chars each, to keep total context
    # small enough for a reasonable response time.
    results = search_knowledge_base(RESEARCH_QUESTION)[:2]  # only first 2 docs
    formatted = []
    for doc in results:
        content = doc["content"][:400]
        if len(doc["content"]) > 400:
            content += "..."
        formatted.append(
            f"[{doc['title']}] (Source: {doc['source']})\n{content}"
        )
    search_context = "\n\n---\n\n".join(formatted)

    initial_message = (
        f"Analyze these documents and answer the question.\n\n"
        f"QUESTION: {RESEARCH_QUESTION}\n\n"
        f"DOCUMENTS:\n{search_context}"
    )

    start = time.time()
    chat_result = human_proxy.initiate_chat(
        assistant,
        message=initial_message,
        max_turns=2,  # safety cap: 1 turn should be enough
    )
    elapsed = time.time() - start

    # Extract the assistant's response
    raw_output = ""
    if chat_result and chat_result.chat_history:
        for msg in reversed(chat_result.chat_history):
            if msg.get("role") == "assistant" or msg.get("name") == "ResearchAssistant":
                raw_output = msg.get("content", "")
                break

    # Remove the termination signal before parsing
    raw_output = raw_output.replace("REPORT_COMPLETE", "").strip()

    print(f"\n{'─' * 40}")
    print(f"Result (elapsed: {elapsed:.1f}s):")
    print(raw_output)

    output = parse_report(raw_output, "ag2")
    path = save_result("ag2", output, elapsed)
    print(f"\nResult saved to: {path}")


if __name__ == "__main__":
    run()
