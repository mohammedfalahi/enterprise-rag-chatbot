from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.db.connection import init_db_pool, close_db_pool
from app.api.chat import router as chat_router
from app.api.documents import router as documents_router
from app.api.conversations import router as conversations_router
from app.observability.tracing import flush_tracing, shutdown_tracing
from fastapi.middleware.cors import CORSMiddleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db_pool()
    try:
        yield
    finally:
        try:
            await close_db_pool()
        finally:
            flush_tracing()
            shutdown_tracing()

app = FastAPI(title="Enterprise Knowledge Intelligence Assistant", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Conversation-Id"],
)

app.include_router(chat_router, prefix="/api/chat", tags=["chat"])
app.include_router(documents_router, prefix="/api/documents", tags=["documents"])
app.include_router(conversations_router, prefix="/api/conversations", tags=["conversations"])
