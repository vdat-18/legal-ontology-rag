"""FastAPI server for RAG pipelines with Redis conversation history."""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import redis
from fastapi import Depends, FastAPI, Header, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from legal_rag.settings import MilvusSettings, NaiveRagSettings, RedisSettings, Settings


class QueryRequest(BaseModel):
    """Incoming question."""

    query: str = Field(min_length=1)
    top_k: int | None = Field(default=None)


class SourceChunk(BaseModel):
    """One retrieved chunk."""

    chunk_id: str
    text: str
    source_name: str
    kind: str
    score: float


class QueryResponse(BaseModel):
    """Pipeline answer with sources and per-step latency."""

    session_id: str
    query: str
    answer: str
    sources: list[SourceChunk]
    latency_ms: dict[str, float]


class HistoryTurn(BaseModel):
    """One question-answer pair."""

    query: str
    answer: str


class HistoryResponse(BaseModel):
    """Conversation history for a session."""

    session_id: str
    history: list[HistoryTurn]


class _AppState:
    """Runtime resources initialized at startup."""

    pipeline: object | None = None
    redis_client: redis.Redis | None = None  # type: ignore[type-arg]
    redis_settings: RedisSettings | None = None
    api_key: str = ""


_state = _AppState()


def _verify_api_key(x_api_key: str = Header()) -> str:
    """Validate the API key from the request header."""
    if not _state.api_key or x_api_key != _state.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


def _redis_key(session_id: str) -> str:
    return f"rag:history:{session_id}"


def _store_turn(session_id: str, query: str, answer: str) -> None:
    """Append one turn to the session history in Redis."""
    if _state.redis_client is None or _state.redis_settings is None:
        return
    key = _redis_key(session_id)
    turn = json.dumps({"query": query, "answer": answer}, ensure_ascii=False)
    _state.redis_client.rpush(key, turn)
    _state.redis_client.ltrim(key, -_state.redis_settings.max_turns, -1)
    _state.redis_client.expire(key, _state.redis_settings.ttl_seconds)


def _load_turns(session_id: str) -> list[HistoryTurn]:
    """Load conversation turns for a session from Redis."""
    if _state.redis_client is None:
        return []
    key = _redis_key(session_id)
    raw_items: list[bytes] = _state.redis_client.lrange(key, 0, -1)
    return [HistoryTurn(**json.loads(raw.decode("utf-8"))) for raw in raw_items]


def _build_contextual_query(query: str, history: list[HistoryTurn]) -> str:
    """Prepend conversation history to the current query for multi-turn context."""
    if not history:
        return query
    lines: list[str] = []
    for turn in history:
        lines.append(f"User: {turn.query}")
        lines.append(f"Assistant: {turn.answer}")
    return "Conversation history:\n" + "\n".join(lines) + f"\n\nCurrent question: {query}"


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Build the pipeline and connect to Redis at startup."""
    from dotenv import find_dotenv, load_dotenv

    from legal_rag.naive_rag.pipeline import NaiveRAGPipeline

    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path)

    settings = Settings.from_environment()
    milvus_settings = MilvusSettings.from_environment()
    naive_rag_settings = NaiveRagSettings.from_environment()
    redis_settings = RedisSettings.from_environment()
    _state.redis_settings = redis_settings
    _state.api_key = os.environ["RAG_API_KEY"]

    _state.pipeline = NaiveRAGPipeline(settings, milvus_settings, naive_rag_settings)

    connection_kwargs: dict = {
        "decode_responses": False,
        "socket_timeout": redis_settings.socket_timeout_seconds,
        "socket_connect_timeout": redis_settings.socket_timeout_seconds,
    }
    if redis_settings.password:
        connection_kwargs["password"] = redis_settings.password
    _state.redis_client = redis.from_url(redis_settings.url, **connection_kwargs)
    _state.redis_client.ping()
    logger.info("Connected to Redis")

    logger.info("RAG API ready")
    yield

    if _state.redis_client is not None:
        _state.redis_client.close()
    _state.pipeline = None


app = FastAPI(title="RAG API", lifespan=_lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check."""
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(
    request: QueryRequest,
    x_session_id: str | None = Header(default=None),
    _key: str = Depends(_verify_api_key),
) -> QueryResponse:
    """Answer a question with conversation context."""
    if _state.pipeline is None:
        msg = "Pipeline not initialized"
        raise RuntimeError(msg)

    latency: dict[str, float] = {}
    session_id = x_session_id or str(uuid.uuid4())

    t0 = time.perf_counter()
    history = _load_turns(session_id)
    latency["history_load_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    contextual_query = _build_contextual_query(request.query, history)
    latency["context_build_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    result = _state.pipeline.run(contextual_query, top_k=request.top_k)
    latency["pipeline_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    _store_turn(session_id, request.query, result.answer)
    latency["history_store_ms"] = (time.perf_counter() - t0) * 1000

    logger.info("Query completed session={} latency={}", session_id, {k: f"{v:.1f}" for k, v in latency.items()})

    return QueryResponse(
        session_id=session_id,
        query=request.query,
        answer=result.answer,
        sources=[
            SourceChunk(chunk_id=s.chunk_id, text=s.text, source_name=s.source_name, kind=s.kind, score=s.score)
            for s in result.sources
        ],
        latency_ms=latency,
    )


@app.get("/history/{session_id}", response_model=HistoryResponse)
def get_history(session_id: str, _key: str = Depends(_verify_api_key)) -> HistoryResponse:
    """Return conversation history for a session."""
    return HistoryResponse(session_id=session_id, history=_load_turns(session_id))


@app.delete("/history/{session_id}")
def clear_history(session_id: str, _key: str = Depends(_verify_api_key)) -> dict[str, str]:
    """Clear conversation history for a session."""
    if _state.redis_client is not None:
        _state.redis_client.delete(_redis_key(session_id))
    return {"status": "cleared", "session_id": session_id}
