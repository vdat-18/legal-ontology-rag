"""FastAPI application for the Graphiti temporal knowledge-graph RAG pipeline."""

from __future__ import annotations

import argparse
import os
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field

from legal_rag.graphiti_rag.history import ConversationHistory, RedisConversationHistory
from legal_rag.graphiti_rag.indexer import load_corpus_metadata
from legal_rag.graphiti_rag.pipeline import GraphitiRAGPipeline, GraphitiRAGResult
from legal_rag.settings import GraphitiSettings, Neo4jSettings, RedisSettings, Settings


class QueryRequest(BaseModel):
    """Request body for one knowledge-graph question."""

    query: str = Field(..., min_length=1)
    max_facts: int | None = Field(None, ge=1, le=100)
    session_id: str | None = Field(None, min_length=1, max_length=200)


class FactSourceResponse(BaseModel):
    """One supporting episode's provenance: the chunk and document a fact came from."""

    chunk_id: str
    source_name: str


class RetrievedFactResponse(BaseModel):
    """One graph fact with the provenance of every episode supporting it."""

    fact: str
    sources: list[FactSourceResponse]


class QueryResponse(BaseModel):
    """Answer plus the graph facts it was grounded in.

    ``rewritten_query`` is exposed because retrieval runs on it, not on ``query``: the graph
    stores formal legal wording, so a colloquial question is rewritten before searching, and
    seeing the rewrite is what makes an unexpected answer diagnosable.
    """

    query: str
    rewritten_query: str
    answer: str
    sources: list[RetrievedFactResponse]
    unindexed_citations: list[str]


class HistoryTurnResponse(BaseModel):
    """One completed exchange in a session's transcript."""

    question: str
    answer: str


class HistoryResponse(BaseModel):
    """A session's stored transcript, oldest turn first."""

    session_id: str
    turns: list[HistoryTurnResponse]


class HealthResponse(BaseModel):
    """API health and sanitized runtime configuration."""

    status: str
    pipeline_loaded: bool
    group_id: str
    embedding_profile: str
    llm_model: str
    neo4j_uri: str


PipelineFactory = Callable[[], GraphitiRAGPipeline]


class PipelineProvider:
    """Lazily owns the Graphiti pipeline for the API process.

    Building the pipeline loads a sentence-transformers model and opens a Neo4j connection, so
    it is created on first query rather than at import: ``/health`` stays answerable while the
    graph or the model is unavailable, which is what a readiness probe needs.
    """

    def __init__(
        self,
        *,
        pipeline_factory: PipelineFactory | None = None,
        group_id: str,
        corpus_config_path: Path | None = None,
    ) -> None:
        """Initialize the lazy provider without touching Neo4j or the embedder."""
        self._pipeline_factory = pipeline_factory
        self._group_id = group_id
        self._corpus_config_path = corpus_config_path
        self._pipeline: GraphitiRAGPipeline | None = None

    @property
    def loaded(self) -> bool:
        """Return whether the heavy pipeline has been initialized."""
        return self._pipeline is not None

    @property
    def group_id(self) -> str:
        """Return the Graphiti group this API answers from."""
        return self._group_id

    def get_pipeline(self) -> GraphitiRAGPipeline:
        """Return the singleton pipeline, creating it on first use."""
        if self._pipeline is None:
            logger.info("Initializing Graphiti RAG pipeline for FastAPI...")
            self._pipeline = self._pipeline_factory() if self._pipeline_factory else self._build_pipeline()
        return self._pipeline

    def _build_pipeline(self) -> GraphitiRAGPipeline:
        """Build the pipeline from environment settings and the optional corpus config."""
        known_document_numbers: frozenset[str] | None = None
        if self._corpus_config_path is not None:
            corpus_metadata = load_corpus_metadata(self._corpus_config_path)
            known_document_numbers = frozenset(
                document.document_number for document in corpus_metadata.documents.values()
            )
        return GraphitiRAGPipeline(
            Settings.from_environment(),
            Neo4jSettings.from_environment(),
            GraphitiSettings.from_environment(),
            self._group_id,
            known_document_numbers=known_document_numbers,
        )

    async def close(self) -> None:
        """Release the Graphiti/Neo4j connection when the server shuts down."""
        if self._pipeline is not None:
            await self._pipeline.close()
            self._pipeline = None


def _to_response(result: GraphitiRAGResult) -> QueryResponse:
    """Convert one pipeline result into its HTTP response body."""
    return QueryResponse(
        query=result.query,
        rewritten_query=result.rewritten_query,
        answer=result.answer,
        sources=[
            RetrievedFactResponse(
                fact=fact.fact,
                sources=[
                    FactSourceResponse(chunk_id=source.chunk_id, source_name=source.source_name)
                    for source in fact.sources
                ],
            )
            for fact in result.sources
        ],
        unindexed_citations=list(result.unindexed_citations),
    )


DEFAULT_CORS_ORIGINS = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3001",
)


def _configured_cors_origins() -> tuple[str, ...]:
    """Read the browser origins allowed to call this API from the environment.

    The frontend dev server moves to the next free port when its default is taken, so the
    default covers both of the usual ones; anything else (a LAN address, a deployed origin)
    is set through ``GRAPHITI_API_CORS_ORIGINS`` as a comma-separated list.
    """
    configured = os.environ.get("GRAPHITI_API_CORS_ORIGINS", "")
    return tuple(origin.strip() for origin in configured.split(",") if origin.strip()) or DEFAULT_CORS_ORIGINS


def create_app(
    *,
    pipeline_factory: PipelineFactory | None = None,
    group_id: str = "traffic_penalty_groundtruth_v1",
    corpus_config_path: Path | None = None,
    cors_origins: Sequence[str] | None = None,
    history: ConversationHistory | None = None,
) -> FastAPI:
    """Create the FastAPI application serving the Graphiti knowledge-graph pipeline.

    Args:
        pipeline_factory: Builds the pipeline; injected by tests so no Neo4j connection is opened.
        group_id: The Graphiti group to answer from.
        corpus_config_path: Corpus config used to flag citations outside the indexed documents.
        cors_origins: Browser origins allowed to call this API. The chat frontend runs on its own
            dev server, so it is a cross-origin caller and is blocked without an explicit allow
            list. Defaults to the frontend dev-server origins, never a wildcard.
        history: Store of earlier turns per session. Without one the API answers each question on
            its own, so a follow-up that leans on the previous turn cannot be resolved.
    """
    allowed_origins = tuple(cors_origins) if cors_origins is not None else _configured_cors_origins()
    provider = PipelineProvider(
        pipeline_factory=pipeline_factory, group_id=group_id, corpus_config_path=corpus_config_path
    )

    @asynccontextmanager
    async def lifespan(app_instance: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await _get_provider(app_instance).close()

    fastapi_app = FastAPI(
        title="Legal Graphiti Knowledge-Graph RAG API",
        version="0.1.0",
        description="HTTP API for Vietnamese road-traffic-law question answering over a temporal knowledge graph.",
        lifespan=lifespan,
    )
    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=list(allowed_origins),
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    fastapi_app.state.pipeline_provider = provider

    @fastapi_app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        """Report health without loading the embedder or opening Neo4j."""
        request_provider = _get_provider(request.app)
        settings = Settings.from_environment()
        return HealthResponse(
            status="ok",
            pipeline_loaded=request_provider.loaded,
            group_id=request_provider.group_id,
            embedding_profile=GraphitiSettings.from_environment().embedding_profile,
            llm_model=settings.llm_model,
            neo4j_uri=Neo4jSettings.from_environment().uri,
        )

    @fastapi_app.post("/graphiti/query", response_model=QueryResponse)
    async def query_graph(payload: QueryRequest, request: Request) -> QueryResponse:
        """Rewrite the question, search the knowledge graph, and answer from the retrieved facts."""
        pipeline = _get_provider(request.app).get_pipeline()
        session_id = payload.session_id
        turns = await run_in_threadpool(history.load, session_id) if history is not None and session_id else []
        result = await pipeline.run(payload.query, max_facts=payload.max_facts, history=turns)
        if history is not None and session_id:
            await run_in_threadpool(history.append, session_id, payload.query, result.answer)
        return _to_response(result)

    @fastapi_app.get("/graphiti/history/{session_id}", response_model=HistoryResponse)
    async def read_history(session_id: str) -> HistoryResponse:
        """Return a session's transcript so a reloaded page can restore what was said.

        The transcript is server state rather than something the browser keeps, so reopening the
        page restores the same conversation the model is still reasoning against; a copy held only
        in the tab would drift out of step with the context used to resolve follow-up questions.
        """
        turns = await run_in_threadpool(history.load, session_id) if history is not None else []
        return HistoryResponse(
            session_id=session_id,
            turns=[HistoryTurnResponse(question=turn.question, answer=turn.answer) for turn in turns],
        )

    return fastapi_app


def _get_provider(app_instance: FastAPI) -> PipelineProvider:
    """Return the app-level pipeline provider."""
    return app_instance.state.pipeline_provider


def _configured_history() -> ConversationHistory | None:
    """Build the conversation store when Redis is configured, otherwise answer statelessly."""
    if "REDIS_URL" not in os.environ:
        return None
    return RedisConversationHistory(RedisSettings.from_environment())


app = create_app(history=_configured_history())


def main() -> None:
    """Run the API with Uvicorn from the command line."""
    parser = argparse.ArgumentParser(description="Run the Graphiti knowledge-graph RAG FastAPI server.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=8100, help="Port to bind.")
    parser.add_argument("--reload", action="store_true", help="Enable Uvicorn reload mode.")
    args = parser.parse_args()

    uvicorn.run("legal_rag.graphiti_rag.api.server:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
