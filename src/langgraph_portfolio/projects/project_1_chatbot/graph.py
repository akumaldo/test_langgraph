from __future__ import annotations

from collections.abc import Iterable

from ...core import ChatMessage, GraphBuilder, Transition
from .state import ChatState


def _last_user_message(state: ChatState) -> str:
    for message in reversed(state.messages):
        if message.role == "user":
            return message.content.lower()
    return ""


def greeting_node(state: ChatState) -> Transition[ChatState]:
    history = state.history + ["greeting_node"]
    return Transition(
        updates={
            "history": history,
            "status": "greeted",
        },
        next_node="intent_classifier",
    )


def intent_classifier(state: ChatState) -> Transition[ChatState]:
    text = _last_user_message(state)
    if not text:
        intent = "clarify"
        confidence = 0.2
    elif any(keyword in text for keyword in ("compare", "difference", "vs", "versus")):
        intent = "comparison"
        confidence = 0.95
    elif any(keyword in text for keyword in ("research", "document", "source")):
        intent = "research"
        confidence = 0.9
    elif any(keyword in text for keyword in ("analyze", "analysis", "data")):
        intent = "analysis"
        confidence = 0.88
    else:
        intent = "general"
        confidence = 0.7

    return Transition(
        updates={
            "history": state.history + ["intent_classifier"],
            "intent": intent,
            "confidence": confidence,
        },
        next_node="router",
    )


def router(state: ChatState) -> Transition[ChatState]:
    next_node = "clarification_node" if state.confidence < 0.5 else "response_generator"
    return Transition(updates={"history": state.history + ["router"]}, next_node=next_node)


def response_generator(state: ChatState) -> Transition[ChatState]:
    if state.intent == "comparison":
        response = "I can compare LangGraph and CrewAI by focusing on routing control, abstraction level, and debugging."
    elif state.intent == "research":
        response = "I can help gather sources and summarize documents into a research-oriented answer."
    elif state.intent == "analysis":
        response = "I can break the task into analysis steps, execution, and recovery."
    else:
        response = "I can help with general orchestration, memory, and context-aware responses."

    updated_messages = list(state.messages) + [ChatMessage(role="assistant", content=response)]
    return Transition(
        updates={
            "history": state.history + ["response_generator"],
            "messages": updated_messages,
            "response": response,
            "status": "responded",
            "conversation_turns": state.conversation_turns + 1,
        },
        next_node="end_node",
    )


def clarification_node(state: ChatState) -> Transition[ChatState]:
    response = "Could you clarify whether you want comparison, research, or analysis support?"
    updated_messages = list(state.messages) + [ChatMessage(role="assistant", content=response)]
    return Transition(
        updates={
            "history": state.history + ["clarification_node"],
            "messages": updated_messages,
            "response": response,
            "clarification_needed": True,
            "status": "needs_clarification",
            "conversation_turns": state.conversation_turns + 1,
        },
        next_node="end_node",
    )


def end_node(state: ChatState) -> Transition[ChatState]:
    return Transition(updates={"history": state.history + ["end_node"], "status": "complete"})


def build_graph() -> GraphBuilder[ChatState]:
    return (
        GraphBuilder[ChatState]()
        .add_node("greeting_node", greeting_node)
        .add_node("intent_classifier", intent_classifier)
        .add_node("router", router)
        .add_node("response_generator", response_generator)
        .add_node("clarification_node", clarification_node)
        .add_node("end_node", end_node)
        .set_entrypoint("greeting_node")
    )


def run_chatbot(messages: Iterable[str]) -> ChatState:
    state = ChatState(messages=[ChatMessage(role="user", content=message) for message in messages], status="starting")
    return build_graph().build().run(state)

