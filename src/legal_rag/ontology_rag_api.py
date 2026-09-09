"""FastAPI server for Ontology RAG pipeline."""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from legal_rag.ontology_rag.models import PenaltyRule
from legal_rag.ontology_rag.pipeline import OntologyRAGPipeline
from legal_rag.settings import MilvusSettings, NaiveRagSettings, Settings


class OntologyQueryRequest(BaseModel):
    """Request body for ontology RAG query."""

    query: str = Field(..., min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)
    query_date: str | None = Field(default=None, description="ISO date string YYYY-MM-DD")


class PenaltyRuleResponse(BaseModel):
    """One valid penalty rule."""

    uri: str
    conduct: str
    vehicle_categories: list[str]
    monetary_penalty: dict[str, Any] | None = None
    point_deduction: int | None = None
    valid_from: str
    valid_to: str | None
    provision_path: str
    document_number: str


class OntologyQueryResponse(BaseModel):
    """Response body for ontology RAG query."""

    session_id: str
    query: str
    answer: str
    rules: list[PenaltyRuleResponse]
    latency_ms: dict[str, float]


@dataclass
class _AppState:
    """Runtime resources initialized at startup."""

    pipeline: OntologyRAGPipeline | None = None
    api_key: str = ""
    ontology_paths: list[Path] = field(default_factory=list)


_state = _AppState()


def _verify_api_key(x_api_key: str = Header()) -> str:
    """Validate the API key from the request header."""
    if not _state.api_key or x_api_key != _state.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


def _format_rule(rule: PenaltyRule) -> PenaltyRuleResponse:
    """Format a PenaltyRule for API response."""
    monetary = None
    if rule.monetary_penalty_min is not None and rule.monetary_penalty_max is not None:
        monetary = {
            "min": str(rule.monetary_penalty_min),
            "max": str(rule.monetary_penalty_max),
            "currency": rule.currency or "VND",
        }

    return PenaltyRuleResponse(
        uri=str(rule.uri),
        conduct=rule.conduct_label_vi,
        vehicle_categories=list(rule.vehicle_categories),
        monetary_penalty=monetary,
        point_deduction=rule.point_value,
        valid_from=rule.valid_from.isoformat(),
        valid_to=rule.valid_to.isoformat() if rule.valid_to else None,
        provision_path=rule.provision_path,
        document_number=rule.document_number,
    )


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Build the ontology RAG pipeline at startup."""
    from dotenv import find_dotenv, load_dotenv

    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path)

    settings = Settings.from_environment()
    milvus_settings = MilvusSettings.from_environment()
    naive_rag_settings = NaiveRagSettings.from_environment()

    ontology_paths_str = os.environ.get("ONTOLOGY_PATHS", "")
    paths = [Path(p.strip()) for p in ontology_paths_str.split(",") if p.strip()] if ontology_paths_str else []

    _state.ontology_paths = paths
    _state.api_key = os.environ.get("RAG_API_KEY", "")

    if not paths:
        logger.warning("ONTOLOGY_PATHS not set, pipeline will not initialize")
    else:
        _state.pipeline = OntologyRAGPipeline(settings, milvus_settings, naive_rag_settings, paths)
        logger.info("Ontology RAG API ready with {} ontology files", len(paths))

    yield

    if _state.pipeline is not None:
        _state.pipeline.close()
    _state.pipeline = None


app = FastAPI(title="Ontology RAG API", lifespan=_lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check."""
    return {"status": "ok"}


@app.post("/query", response_model=OntologyQueryResponse)
def query(
    request: OntologyQueryRequest,
    x_session_id: str | None = Header(default=None),
    _key: str = Depends(_verify_api_key),
) -> OntologyQueryResponse:
    """Answer a question using ontology-enhanced RAG."""
    if _state.pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    latency: dict[str, float] = {}
    session_id = x_session_id or str(uuid.uuid4())

    query_date = None
    if request.query_date:
        try:
            query_date = datetime.fromisoformat(request.query_date).date()
        except ValueError as err:
            raise HTTPException(status_code=422, detail="Invalid query_date format. Use YYYY-MM-DD.") from err

    t0 = time.perf_counter()
    result = _state.pipeline.run(request.query, query_date=query_date, top_k=request.top_k)
    latency["pipeline_ms"] = (time.perf_counter() - t0) * 1000

    logger.info(
        "Query completed session={} rules={} latency={:.1f}ms",
        session_id,
        len(result.rules),
        latency["pipeline_ms"],
    )

    return OntologyQueryResponse(
        session_id=session_id,
        query=request.query,
        answer=result.answer,
        rules=[_format_rule(rule) for rule in result.rules],
        latency_ms=latency,
    )
