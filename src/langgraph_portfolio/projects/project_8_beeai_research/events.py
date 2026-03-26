"""
Project 8 — Event Handlers for BeeAI Agent Observability.

---------------------------------------------------------------------------
BEEAI'S EVENT SYSTEM (the killer feature)
---------------------------------------------------------------------------

Remember how the P2 research agent worked? You ran it, waited, and got
the final answer. If the answer was wrong, you had to guess WHERE the
agent went wrong — bad search query? Misread a result? Faulty reasoning?

BeeAI solves this with EVENTS. The agent emits structured events at
every step of the ReAct loop:

    Think → tool call → tool result → Think → ... → final answer

Each step fires an event you can listen to. It's like adding print()
statements everywhere, but structured, typed, and built into the
framework.

Compare observability across frameworks:

    LangGraph (P2):
      - You see state AFTER each node finishes
      - To see intermediate steps, you manually print inside node functions
      - Granularity: per-node (coarse)

    CrewAI (P6):
      - verbose=True shows delegation decisions and agent outputs
      - Less control over what gets logged
      - Granularity: per-agent-step (medium)

    AG2 (P7):
      - Print statements in message callbacks
      - GroupChat messages are visible
      - Granularity: per-message (medium)

    BeeAI (P8):
      - Events for EVERY thought, tool call, tool result, final answer
      - You hook into specific events with decorators
      - Granularity: per-ReAct-step (fine)

The event types for ReActAgent:
  - "start"          → iteration begins (tells you which tools are available)
  - "update"         → state changed (thought parsed, tool called, tool returned)
  - "partial_update" → same as update but streaming (line by line)
  - "success"        → agent produced a final answer
  - "error"          → something went wrong
  - "retry"          → agent is retrying after an error
---------------------------------------------------------------------------
"""

from beeai_framework.emitter import Emitter, EventMeta


# ---------------------------------------------------------------------------
# 1. EVENT HANDLER SETUP
#
# BeeAI uses a decorator pattern for events:
#
#     @agent.emitter.on("update")
#     async def handle_update(data, event):
#         ...
#
# The decorator registers an async function to be called whenever that
# event fires. You can listen to:
#   - A specific event: @agent.emitter.on("update")
#   - All events:       @agent.emitter.on("*")
#
# The handler receives:
#   - data:  the event payload (varies by event type)
#   - event: metadata about the event (name, timestamp, etc.)
#
# We define the handlers as standalone functions here, then attach them
# to the agent in agent.py. This keeps event logic separate from agent
# setup — a clean separation of concerns.
# ---------------------------------------------------------------------------

def attach_event_handlers(emitter: Emitter) -> None:
    """Attach all event handlers to an agent's emitter.

    This function registers handlers for the key events in the ReAct loop.
    Call it with agent.emitter after creating the agent.

    Why separate this from agent.py?
    Same reason you'd separate logging from business logic — the agent
    does the thinking, the event handlers do the observing.
    """

    # ----- "update" event -----
    # Fires after each meaningful state change in the ReAct loop.
    # The `data` payload is a ReActAgentUpdateEvent with:
    #   - data.data: the iteration state (thought, tool_name, tool_input, tool_output, final_answer)
    #   - data.update: what specifically changed (key + value)
    #   - data.meta: metadata (iteration number, success flag)
    #
    # This is where we see the agent's reasoning in real time.
    @emitter.on("update")
    async def on_update(data, event: EventMeta):
        iteration_state = data.data

        # The iteration state has these fields (all optional):
        #   thought:      what the agent is thinking
        #   tool_name:    which tool it wants to call
        #   tool_input:   what arguments to pass
        #   tool_output:  what the tool returned
        #   final_answer: the agent's conclusion (only on the last iteration)

        # We only log the tool_output updates — these are the most
        # informative. The thought and tool_name/input are already
        # visible from the "start" and other update events.
        update_key = data.update.key
        if update_key == "tool_output":
            tool_output = iteration_state.tool_output or ""
            # Truncate long outputs for readability
            preview = tool_output[:200] + "..." if len(tool_output) > 200 else tool_output
            print(f"  👁 Observe: {preview}")

    # ----- "success" event -----
    # Fires when the agent reaches a final answer.
    # The `data` payload is a ReActAgentSuccessEvent with:
    #   - data.data: the final AssistantMessage
    #   - data.iterations: list of all iterations the agent went through
    #   - data.memory: the agent's memory state
    @emitter.on("success")
    async def on_success(data, event: EventMeta):
        iterations = data.iterations
        print(f"\n  ✅ Agent finished after {len(iterations)} iteration(s)")

        # Log each iteration's thought and action — this is the full
        # trace of the agent's reasoning. Like a debugger's step-through.
        for i, iteration in enumerate(iterations, 1):
            state = iteration.state
            if state.thought:
                print(f"     Iteration {i} — 🧠 Thought: {state.thought[:100]}...")
            if state.tool_name:
                print(f"     Iteration {i} — 🔧 Action: {state.tool_name}({state.tool_input})")
            if state.final_answer:
                print(f"     Iteration {i} — 💡 Answer: {state.final_answer[:100]}...")

    # ----- "error" event -----
    # Fires when something goes wrong (LLM error, tool error, etc.)
    @emitter.on("error")
    async def on_error(data, event: EventMeta):
        print(f"  ❌ Error: {data}")

    # ----- "retry" event -----
    # Fires when the agent retries after a parsing or tool error.
    # This is useful to know — it means the LLM's output couldn't be
    # parsed as a valid ReAct step, so BeeAI asked it to try again.
    @emitter.on("retry")
    async def on_retry(data, event: EventMeta):
        print(f"  🔄 Retry: {data}")
