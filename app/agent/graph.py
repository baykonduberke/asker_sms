from langgraph.graph import END, START, StateGraph

from .nodes import build_openrouter_messages, generate_response, ingest_user_message
from .state import AgentState


def build_graph(checkpointer):
    """Build the SMS agent pipeline: ingest -> build payload -> generate."""
    graph = StateGraph(AgentState)

    graph.add_node("ingest_user_message", ingest_user_message)
    graph.add_node("build_openrouter_messages", build_openrouter_messages)
    graph.add_node("generate_response", generate_response)

    graph.add_edge(START, "ingest_user_message")
    graph.add_edge("ingest_user_message", "build_openrouter_messages")
    graph.add_edge("build_openrouter_messages", "generate_response")
    graph.add_edge("generate_response", END)

    return graph.compile(checkpointer=checkpointer)
