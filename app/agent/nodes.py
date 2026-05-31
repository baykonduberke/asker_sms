import requests
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.core.config import settings
from .prompts import SYSTEM_PROMPT
from .state import AgentState

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
REQUEST_TIMEOUT = 30

ROLE_BY_MESSAGE_TYPE = {
    "system": "system",
    "human": "user",
    "ai": "assistant",
}


def ingest_user_message(state: AgentState) -> AgentState:
    """Store the incoming user text in the conversation history."""
    user_text = state.get("user_message", "").strip()
    if not user_text:
        return {}
    return {"messages": [HumanMessage(content=user_text)]}


def build_openrouter_messages(state: AgentState) -> AgentState:
    """Prepend the system prompt and convert history to OpenRouter payload."""
    conversation: list[BaseMessage] = [
        SystemMessage(content=SYSTEM_PROMPT),
        *state.get("messages", []),
    ]
    payload = []
    for message in conversation:
        role = ROLE_BY_MESSAGE_TYPE.get(message.type)
        if role is None:
            continue
        payload.append({"role": role, "content": message.content})
    return {"openrouter_response": payload}


def generate_response(state: AgentState) -> AgentState:
    """Call OpenRouter and append the assistant reply to the conversation."""
    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": settings.openrouter_model,
        "messages": state.get("openrouter_response", []),
        "temperature": 0.7,
        "max_tokens": 500,
        "stream": False,
    }

    response = requests.post(OPENROUTER_URL, headers=headers, json=data, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()

    assistant_text = response.json()["choices"][0]["message"]["content"].strip()
    return {
        "messages": [AIMessage(content=assistant_text)],
        "agent_message": assistant_text,
    }
