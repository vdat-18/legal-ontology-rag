"""FastAPI server for Ontology RAG pipeline."""

from __future__ import annotations

import argparse
import os
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from loguru import logger
from pydantic import BaseModel, Field, field_validator

from legal_rag.ontology_rag.models import (
    ComponentProvenance,
    ConditionExpression,
    LegalConsequence,
    LegalException,
    OntologyTerm,
    PenaltyRule,
    ProvisionCitation,
)
from legal_rag.ontology_rag.pipeline import OntologyRAGPipeline
from legal_rag.settings import MilvusSettings, NaiveRagSettings, Settings


class _InvalidQueryDateError(ValueError):
    """Raised when an API query date is not a canonical ISO calendar date."""

    def __init__(self) -> None:
        super().__init__("query_date must use YYYY-MM-DD")


class OntologyQueryRequest(BaseModel):
    """Request body for ontology RAG query."""

    query: str = Field(..., min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)
    query_date: date | None = Field(default=None, description="ISO date string YYYY-MM-DD")

    @field_validator("query_date", mode="before")
    @classmethod
    def _validate_query_date(cls, value: Any) -> date | None:
        """Accept only the canonical ISO calendar-date representation."""
        if value is None:
            return None
        if isinstance(value, date) and not isinstance(value, datetime):
            return value
        if not isinstance(value, str):
            raise _InvalidQueryDateError
        try:
            parsed = date.fromisoformat(value)
        except ValueError as error:
            raise _InvalidQueryDateError from error
        if parsed.isoformat() != value:
            raise _InvalidQueryDateError
        return parsed


class ComponentProvenanceResponse(BaseModel):
    """Source provisions and chunks supporting one rule component."""

    provision_uris: list[str]
    provision_paths: list[str]
    source_chunk_ids: list[str]


class OntologyTermResponse(BaseModel):
    """One labeled ontology term used by a consequence."""

    uri: str
    label_vi: str | None


class ConditionExpressionResponse(BaseModel):
    """One node in a source-grounded applicability expression."""

    uri: str
    kind: str
    label_vi: str | None
    operands: list[ConditionExpressionResponse]
    provenance: ComponentProvenanceResponse


class LegalConsequenceResponse(BaseModel):
    """One independently evidenced legal consequence."""

    uri: str
    kind: str
    label_vi: str | None
    minimum_amount: str | None
    maximum_amount: str | None
    currency: str | None
    point_value: int | None
    minimum_duration_months: int | None
    maximum_duration_months: int | None
    affected_targets: list[OntologyTermResponse]
    responsible_roles: list[OntologyTermResponse]
    provenance: ComponentProvenanceResponse


class ProvisionCitationResponse(BaseModel):
    """One provision citation retained by a legal exception."""

    uri: str
    citation_text_vi: str | None
    referenced_provision_uris: list[str]
    referenced_provision_paths: list[str]


class LegalExceptionResponse(BaseModel):
    """One source-grounded exception attached to a rule."""

    uri: str
    citations: list[ProvisionCitationResponse]
    excluded_provision_uris: list[str]
    excluded_provision_paths: list[str]
    redirected_rule_uris: list[str]
    provenance: ComponentProvenanceResponse


class PenaltyRuleResponse(BaseModel):
    """One valid penalty rule."""

    uri: str
    conduct: str
    vehicle_categories: list[str]
    subject_kind: str | None
    roles: list[str]
    monetary_penalty: dict[str, Any] | None = None
    point_deduction: int | None = None
    valid_from: date | None
    valid_to: date | None
    provision_path: str
    document_number: str
    source_chunk_ids: list[str]
    consequences: list[LegalConsequenceResponse]
    condition_expressions: list[ConditionExpressionResponse]
    exceptions: list[LegalExceptionResponse]
    rule_provenance: ComponentProvenanceResponse
    conduct_provenance: list[ComponentProvenanceResponse]
    applicability_provenance: list[ComponentProvenanceResponse]


class RetrievedContextResponse(BaseModel):
    """One retrieved or generated-answer context with its provenance."""

    chunk_id: str
    text: str
    source_name: str
    kind: str
    score: float
    unit_id: str | None = None
    unit_kind: str | None = None
    evidence_chunk_ids: list[str] = Field(default_factory=list)


class OntologyPipelineMetadataResponse(BaseModel):
    """Temporal decisions and candidate accounting from ontology retrieval."""

    requested_query_date: date | None = None
    temporal_mode: str | None = None
    query_dates: list[date] = Field(default_factory=list)
    temporal_status: dict[str, str] = Field(default_factory=dict)
    candidate_count: int | None = None
    invalid_rule_count: int | None = None
    indeterminate_rule_count: int | None = None


class OntologyQueryResponse(BaseModel):
    """Response body for ontology RAG query."""

    session_id: str
    query: str
    answer: str
    rules: list[PenaltyRuleResponse]
    sources: list[RetrievedContextResponse]
    answer_contexts: list[RetrievedContextResponse]
    pipeline_metadata: OntologyPipelineMetadataResponse
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
    if rule.monetary_penalty_min is not None or rule.monetary_penalty_max is not None:
        monetary = {
            "min": str(rule.monetary_penalty_min) if rule.monetary_penalty_min is not None else None,
            "max": str(rule.monetary_penalty_max) if rule.monetary_penalty_max is not None else None,
            "currency": rule.currency or "VND",
        }

    return PenaltyRuleResponse(
        uri=str(rule.uri),
        conduct=rule.conduct_label_vi,
        vehicle_categories=list(rule.vehicle_categories),
        subject_kind=rule.subject_kind,
        roles=list(rule.roles),
        monetary_penalty=monetary,
        point_deduction=rule.point_value,
        valid_from=rule.valid_from,
        valid_to=rule.valid_to,
        provision_path=rule.provision_path,
        document_number=rule.document_number,
        source_chunk_ids=list(rule.source_chunk_ids),
        consequences=[_format_consequence(item) for item in rule.consequences],
        condition_expressions=[_format_condition(item) for item in rule.condition_expressions],
        exceptions=[_format_exception(item) for item in rule.exceptions],
        rule_provenance=_format_provenance(rule.rule_provenance),
        conduct_provenance=[_format_provenance(item) for item in rule.conduct_provenance],
        applicability_provenance=[_format_provenance(item) for item in rule.applicability_provenance],
    )


def _format_provenance(provenance: ComponentProvenance) -> ComponentProvenanceResponse:
    """Format component provenance without losing URI identity."""
    return ComponentProvenanceResponse(
        provision_uris=[str(item) for item in provenance.provision_uris],
        provision_paths=list(provenance.provision_paths),
        source_chunk_ids=list(provenance.source_chunk_ids),
    )


def _format_term(term: OntologyTerm) -> OntologyTermResponse:
    """Format one labeled ontology term."""
    return OntologyTermResponse(uri=str(term.uri), label_vi=term.label_vi)


def _format_condition(condition: ConditionExpression) -> ConditionExpressionResponse:
    """Format one applicability tree while retaining nested logical structure."""
    return ConditionExpressionResponse(
        uri=str(condition.uri),
        kind=condition.kind,
        label_vi=condition.label_vi,
        operands=[_format_condition(item) for item in condition.operands],
        provenance=_format_provenance(condition.provenance),
    )


def _format_consequence(consequence: LegalConsequence) -> LegalConsequenceResponse:
    """Format one legal consequence and its independent evidence."""
    return LegalConsequenceResponse(
        uri=str(consequence.uri),
        kind=consequence.kind,
        label_vi=consequence.label_vi,
        minimum_amount=str(consequence.minimum_amount) if consequence.minimum_amount is not None else None,
        maximum_amount=str(consequence.maximum_amount) if consequence.maximum_amount is not None else None,
        currency=consequence.currency,
        point_value=consequence.point_value,
        minimum_duration_months=consequence.minimum_duration_months,
        maximum_duration_months=consequence.maximum_duration_months,
        affected_targets=[_format_term(item) for item in consequence.affected_targets],
        responsible_roles=[_format_term(item) for item in consequence.responsible_roles],
        provenance=_format_provenance(consequence.provenance),
    )


def _format_citation(citation: ProvisionCitation) -> ProvisionCitationResponse:
    """Format one exception citation."""
    return ProvisionCitationResponse(
        uri=str(citation.uri),
        citation_text_vi=citation.citation_text_vi,
        referenced_provision_uris=[str(item) for item in citation.referenced_provision_uris],
        referenced_provision_paths=list(citation.referenced_provision_paths),
    )


def _format_exception(exception: LegalException) -> LegalExceptionResponse:
    """Format one legal exception and redirect relationship."""
    return LegalExceptionResponse(
        uri=str(exception.uri),
        citations=[_format_citation(item) for item in exception.citations],
        excluded_provision_uris=[str(item) for item in exception.excluded_provision_uris],
        excluded_provision_paths=list(exception.excluded_provision_paths),
        redirected_rule_uris=[str(item) for item in exception.redirected_rule_uris],
        provenance=_format_provenance(exception.provenance),
    )


def _format_context(context: Any) -> RetrievedContextResponse:
    """Format one exact context returned by the pipeline."""
    evidence_chunk_ids = getattr(context, "effective_evidence_chunk_ids", ())
    return RetrievedContextResponse(
        chunk_id=context.chunk_id,
        text=context.text,
        source_name=context.source_name,
        kind=context.kind,
        score=context.score,
        unit_id=getattr(context, "effective_unit_id", None),
        unit_kind=getattr(context, "unit_kind", None),
        evidence_chunk_ids=list(evidence_chunk_ids),
    )


def _format_pipeline_metadata(result: Any, requested_query_date: date | None) -> OntologyPipelineMetadataResponse:
    """Format optional temporal and candidate diagnostics from the core pipeline."""
    temporal_status = getattr(result, "temporal_status", {})
    return OntologyPipelineMetadataResponse(
        requested_query_date=requested_query_date,
        temporal_mode=getattr(result, "temporal_mode", None),
        query_dates=list(getattr(result, "query_dates", ())),
        temporal_status={str(key): str(value) for key, value in temporal_status.items()},
        candidate_count=getattr(result, "candidate_count", None),
        invalid_rule_count=getattr(result, "invalid_rule_count", None),
        indeterminate_rule_count=getattr(result, "indeterminate_rule_count", None),
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

    t0 = time.perf_counter()
    result = _state.pipeline.run(request.query, query_date=request.query_date, top_k=request.top_k)
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
        sources=[_format_context(context) for context in getattr(result, "sources", ())],
        answer_contexts=[_format_context(context) for context in getattr(result, "answer_contexts", ())],
        pipeline_metadata=_format_pipeline_metadata(result, request.query_date),
        latency_ms=latency,
    )


def main() -> None:
    """Run the Ontology RAG API with Uvicorn from the command line."""
    import uvicorn

    parser = argparse.ArgumentParser(description="Run the ontology-enhanced RAG FastAPI server.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind.")
    parser.add_argument("--reload", action="store_true", help="Enable Uvicorn reload mode.")
    arguments = parser.parse_args()

    uvicorn.run(
        "legal_rag.ontology_rag_api:app",
        host=arguments.host,
        port=arguments.port,
        reload=arguments.reload,
    )


if __name__ == "__main__":
    main()
