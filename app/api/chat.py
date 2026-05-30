from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    user_id: str  # SMS senaryosunda telefon numarası burada kullanılır
    message: str


class ChatResponse(BaseModel):
    reply: str


@router.post("", response_model=ChatResponse)
def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    graph = request.app.state.agent_graph

    result = graph.invoke(
        {"user_message": payload.message},
        config={"configurable": {"thread_id": payload.user_id}},
    )
    return ChatResponse(reply=result["agent_message"])
