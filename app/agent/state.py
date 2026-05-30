from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    user_message: str
    agent_message: str
    messages: Annotated[list[BaseMessage], add_messages]
    openrouter_response: list[dict[str, str]]