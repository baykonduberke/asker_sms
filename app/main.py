from contextlib import ExitStack, asynccontextmanager

from fastapi import FastAPI
from langgraph.checkpoint.postgres import PostgresSaver

from app.agent.graph import build_graph
from app.api.chat import router as chat_router
from app.api.health import router as health_router
from app.api.whatsapp import router as whatsapp_router
from app.core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    stack = ExitStack()
    app.state.exit_stack = stack

    checkpointer = stack.enter_context(
        PostgresSaver.from_conn_string(settings.postgres_url)
    )
    checkpointer.setup()  # ilk kurulumda tablo/migration oluşturur

    app.state.agent_graph = build_graph(checkpointer)

    try:
        yield
    finally:
        stack.close()


app = FastAPI(title=settings.app_name, lifespan=lifespan)

app.include_router(health_router)
app.include_router(chat_router)
app.include_router(whatsapp_router)
