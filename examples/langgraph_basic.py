"""Minimal LangGraph example: a simple state machine that greets a user.

Run with:
    python examples/langgraph_basic.py
"""

from __future__ import annotations

import operator
from typing import Annotated, List

from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict


class ChatState(TypedDict):
    """Simple state shared across graph nodes."""

    name: str
    messages: Annotated[List[str], operator.add]


def greet_node(state: ChatState) -> ChatState:
    """Append a greeting message for the user."""
    return {"messages": [f"Hello, {state['name']}! Welcome to LangGraph."]}


def ask_node(state: ChatState) -> ChatState:
    """Append a follow-up question."""
    return {"messages": ["How can I help you today?"]}


def build_basic_graph():
    """Build and compile a two-node greeting graph."""
    builder = StateGraph(ChatState)

    builder.add_node("greet", greet_node)
    builder.add_node("ask", ask_node)

    builder.set_entry_point("greet")
    builder.add_edge("greet", "ask")
    builder.add_edge("ask", END)

    return builder.compile()


if __name__ == "__main__":
    graph = build_basic_graph()
    result = graph.invoke({"name": "Developer", "messages": []})

    print("LangGraph basic example output:")
    for message in result["messages"]:
        print(f"  - {message}")
