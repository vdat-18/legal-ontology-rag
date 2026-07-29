"""FastAPI application for the Corrective RAG pipeline."""

from __future__ import annotations

import argparse
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from threading import RLock

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from loguru import logger
from pydantic import BaseModel, Field

from legal_rag.corrective_rag.generate.config import RAGConfig
from legal_rag.corrective_rag.generate.pipeline import CorrectiveRAGPipeline
from legal_rag.corrective_rag.generate.web_search import WebSearchProvider


class QueryRequest(BaseModel):
    """Request body for a CRAG query."""

    query: str = Field(..., min_length=1)
    limit: int = Field(5, ge=1, le=20)


class SearchRequest(BaseModel):
    """Request body for vector-only retrieval."""

    query: str = Field(..., min_length=1)
    limit: int = Field(5, ge=1, le=20)


class GradedChunkResponse(BaseModel):
    """One retrieved chunk with CRAG evaluator metadata."""

    chunk_id: str
    source_name: str
    text: str
    score: float | None = None
    confidence_score: float
    action: str
    rationale: str
    is_relevant: bool


class QueryResponse(BaseModel):
    """Response body for a full CRAG query."""

    query: str
    reformulated_query: str | None
    status: str
    crag_action: str
    retrieved_count: int
    second_retrieved_count: int
    web_search_count: int
    internal_knowledge_count: int
    external_knowledge_count: int
    relevant_count: int
    graded_chunks: list[GradedChunkResponse]
    answer: str


class SearchChunkResponse(BaseModel):
    """One direct vector retrieval result."""

    chunk_id: str | None = None
    score: float | None = None
    text: str
    embedding_text: str
    source_name: str
    scope_id: str
    chunk_index: int
    previous_chunk_id: str | None = None
    next_chunk_id: str | None = None


class SearchResponse(BaseModel):
    """Response body for direct vector retrieval."""

    query: str
    limit: int
    results: list[SearchChunkResponse]


class WebDocumentResponse(BaseModel):
    """One direct web-search result document."""

    source_name: str
    url: str
    text: str


class WebSearchResponse(BaseModel):
    """Response body for direct web search."""

    query: str
    limit: int
    enabled: bool
    results: list[WebDocumentResponse]


class HealthResponse(BaseModel):
    """Basic API health and sanitized runtime configuration."""

    status: str
    pipeline_loaded: bool
    embedding_profile: str
    llm_model: str
    milvus_uri: str
    web_search_enabled: bool


PipelineFactory = Callable[[], CorrectiveRAGPipeline]
ConfigFactory = Callable[[], RAGConfig]


class PipelineProvider:
    """Lazily owns the CRAG pipeline for the API process."""

    def __init__(
        self,
        *,
        pipeline_factory: PipelineFactory | None = None,
        config_factory: ConfigFactory = RAGConfig.from_env,
    ) -> None:
        """Initialize the lazy provider."""
        self._pipeline_factory = pipeline_factory
        self._config_factory = config_factory
        self._pipeline: CorrectiveRAGPipeline | None = None
        self._config: RAGConfig | None = None
        self._lock = RLock()

    @property
    def loaded(self) -> bool:
        """Return whether the heavy CRAG pipeline has been initialized."""
        return self._pipeline is not None

    def get_config(self) -> RAGConfig:
        """Load lightweight config without creating embedding or Milvus clients."""
        with self._lock:
            if self._config is None:
                self._config = self._config_factory()
            return self._config

    def get_pipeline(self) -> CorrectiveRAGPipeline:
        """Return the singleton CRAG pipeline, creating it on first use."""
        with self._lock:
            if self._pipeline is None:
                logger.info("Initializing Corrective RAG pipeline for FastAPI...")
                if self._pipeline_factory is not None:
                    self._pipeline = self._pipeline_factory()
                else:
                    self._pipeline = CorrectiveRAGPipeline(self.get_config())
            return self._pipeline

    def close(self) -> None:
        """Close the pipeline when the API server shuts down."""
        with self._lock:
            if self._pipeline is not None:
                self._pipeline.close()
                self._pipeline = None


def create_app(
    *,
    pipeline_factory: PipelineFactory | None = None,
    config_factory: ConfigFactory = RAGConfig.from_env,
) -> FastAPI:
    """Create the FastAPI application."""
    provider = PipelineProvider(pipeline_factory=pipeline_factory, config_factory=config_factory)

    @asynccontextmanager
    async def lifespan(app_instance: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            app_provider = _get_provider(app_instance)
            app_provider.close()

    fastapi_app = FastAPI(
        title="Legal Corrective RAG API",
        version="0.1.0",
        description="HTTP API for Vietnamese road-traffic-law Corrective RAG.",
        lifespan=lifespan,
    )
    fastapi_app.state.pipeline_provider = provider

    @fastapi_app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        """Return API health without loading the heavy CRAG pipeline."""
        request_provider = _get_provider(request.app)
        try:
            config = request_provider.get_config()
        except Exception as exc:
            logger.exception("FastAPI health check failed while loading config.")
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        return HealthResponse(
            status="ok",
            pipeline_loaded=request_provider.loaded,
            embedding_profile=config.embedding_profile,
            llm_model=config.llm_model,
            milvus_uri=config.milvus_uri,
            web_search_enabled=config.web_search_enabled,
        )

    @fastapi_app.post("/crag/query", response_model=QueryResponse)
    async def query_crag(payload: QueryRequest, request: Request) -> QueryResponse:
        """Run the full Corrective RAG pipeline for one question."""
        request_provider = _get_provider(request.app)
        try:
            pipeline = request_provider.get_pipeline()
            result = await run_in_threadpool(pipeline.run, payload.query, payload.limit)
        except Exception as exc:
            logger.exception("FastAPI CRAG query failed.")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return QueryResponse.model_validate(result)

    @fastapi_app.post("/crag/search", response_model=SearchResponse)
    async def search_vector_db(payload: SearchRequest, request: Request) -> SearchResponse:
        """Run vector-only retrieval against the configured Milvus collection."""
        request_provider = _get_provider(request.app)
        try:
            pipeline = request_provider.get_pipeline()
            results = await run_in_threadpool(pipeline.db.search, payload.query, payload.limit)
        except Exception as exc:
            logger.exception("FastAPI vector search failed.")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return SearchResponse(query=payload.query, limit=payload.limit, results=results)

    @fastapi_app.post("/crag/web-search", response_model=WebSearchResponse)
    async def search_web(payload: SearchRequest, request: Request) -> WebSearchResponse:
        """Run direct web search without loading the vector or LLM pipeline."""
        request_provider = _get_provider(request.app)
        try:
            config = request_provider.get_config()
            web_search = WebSearchProvider(config)
            try:
                results = await run_in_threadpool(web_search.search, payload.query)
            finally:
                web_search.close()
        except Exception as exc:
            logger.exception("FastAPI web search failed.")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return WebSearchResponse(
            query=payload.query,
            limit=payload.limit,
            enabled=config.web_search_enabled,
            results=results[: payload.limit],
        )

    return fastapi_app


def _get_provider(app_instance: FastAPI) -> PipelineProvider:
    """Return the app-level pipeline provider."""
    return app_instance.state.pipeline_provider


app = create_app()


def main() -> None:
    """Run the API with Uvicorn from the command line."""
    parser = argparse.ArgumentParser(description="Run the Corrective RAG FastAPI server.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind.")
    parser.add_argument("--reload", action="store_true", help="Enable Uvicorn reload mode.")
    args = parser.parse_args()

    uvicorn.run(
        "legal_rag.corrective_rag.api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
