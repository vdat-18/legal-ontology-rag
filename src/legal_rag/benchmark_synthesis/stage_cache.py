"""Fingerprint resumable synthesis-stage artifacts."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from legal_rag.benchmark_synthesis.chunk_screening import CHUNK_SCREENING_PROMPT_VERSION
from legal_rag.benchmark_synthesis.config import SynthesisSettings
from legal_rag.benchmark_synthesis.document_metadata import (
    DOCUMENT_METADATA_PROMPT_VERSION,
    DocumentMetadataRecord,
)
from legal_rag.benchmark_synthesis.fact_extraction import FACT_EXTRACTION_PROMPT_VERSION
from legal_rag.benchmark_synthesis.models import LEGAL_FACT_SCHEMA_VERSION
from legal_rag.chunker.models import Chunk
from legal_rag.io_utils import write_json_atomic
from legal_rag.parser.hashing import sha256_json

STAGE_CACHE_SCHEMA_VERSION = "1.0.0"

StageName = Literal["chunk_screening", "document_metadata", "fact_extraction"]


@dataclass(frozen=True, slots=True)
class StageCacheFingerprint:
    """One immutable identity for a resumable synthesis-stage artifact."""

    stage: StageName
    run_id: str
    prompt_version: str
    model: str

    def to_dict(self) -> dict[str, str]:
        """Return a stable JSON-serializable fingerprint record."""
        return {
            "schema_version": STAGE_CACHE_SCHEMA_VERSION,
            "stage": self.stage,
            "run_id": self.run_id,
            "prompt_version": self.prompt_version,
            "model": self.model,
        }


def stage_cache_path(artifact_path: Path) -> Path:
    """Return the run-fingerprint sidecar path for one primary artifact."""
    return artifact_path.with_suffix(".run.json")


def build_chunk_screening_fingerprint(
    chunks: Sequence[Chunk],
    settings: SynthesisSettings,
    *,
    llm_base_url: str,
    llm_model: str,
) -> StageCacheFingerprint:
    """Fingerprint every input that can change chunk-screening output."""
    return _build_fingerprint(
        stage="chunk_screening",
        prompt_version=CHUNK_SCREENING_PROMPT_VERSION,
        model=llm_model,
        identity={
            "chunks": [_chunk_identity(chunk, include_context=True) for chunk in chunks],
            "llm": {"base_url": llm_base_url, "model": llm_model},
            "settings": {
                "batch_size": settings.chunk_screening_batch_size,
                "max_concurrency": settings.chunk_screening_max_concurrency,
                "generation": asdict(settings.chunk_screening),
            },
        },
    )


def build_document_metadata_fingerprint(
    chunks: Sequence[Chunk],
    settings: SynthesisSettings,
    *,
    llm_base_url: str,
    llm_model: str,
) -> StageCacheFingerprint:
    """Fingerprint every input that can change document metadata."""
    root_chunks = [chunk for chunk in chunks if chunk.record_scope == "root"]
    return _build_fingerprint(
        stage="document_metadata",
        prompt_version=DOCUMENT_METADATA_PROMPT_VERSION,
        model=llm_model,
        identity={
            "chunks": [_chunk_identity(chunk, include_context=False) for chunk in root_chunks],
            "llm": {"base_url": llm_base_url, "model": llm_model},
            "settings": {
                "opening_chunk_count": settings.document_metadata_opening_chunk_count,
                "closing_chunk_count": settings.document_metadata_closing_chunk_count,
                "generation": asdict(settings.document_metadata),
            },
        },
    )


def build_fact_extraction_fingerprint(
    chunks: Sequence[Chunk],
    accepted_chunk_ids: Sequence[str],
    metadata: DocumentMetadataRecord,
    settings: SynthesisSettings,
    *,
    screening_run_id: str,
    metadata_run_id: str,
    llm_base_url: str,
    llm_model: str,
) -> StageCacheFingerprint:
    """Fingerprint every input that can change extracted legal facts."""
    return _build_fingerprint(
        stage="fact_extraction",
        prompt_version=FACT_EXTRACTION_PROMPT_VERSION,
        model=llm_model,
        identity={
            "accepted_chunk_ids": list(accepted_chunk_ids),
            "chunks": [_chunk_identity(chunk, include_context=True) for chunk in chunks],
            "input_runs": {
                "chunk_screening": screening_run_id,
                "document_metadata": metadata_run_id,
            },
            "legal_fact_schema": LEGAL_FACT_SCHEMA_VERSION,
            "llm": {"base_url": llm_base_url, "model": llm_model},
            "metadata": metadata.to_dict(),
            "settings": {
                "batch_size": settings.fact_extraction_batch_size,
                "max_concurrency": settings.fact_extraction_max_concurrency,
                "max_preceding_chunks": settings.fact_extraction_max_preceding_chunks,
                "max_following_chunks": settings.fact_extraction_max_following_chunks,
                "max_article_context_chunks": settings.fact_extraction_max_article_context_chunks,
                "generation": asdict(settings.fact_extraction),
            },
        },
    )


def initialize_stage_cache(
    fingerprint_path: Path,
    artifact_paths: Sequence[Path],
    expected: StageCacheFingerprint,
) -> None:
    """Create a new sidecar or validate the fingerprint of resumable artifacts."""
    if fingerprint_path.exists():
        existing = read_stage_cache_fingerprint(fingerprint_path)
        if existing == expected:
            return
        if not any(path.exists() for path in artifact_paths):
            write_stage_cache_fingerprint(fingerprint_path, expected)
            return
        msg = f"Existing {expected.stage} artifacts belong to a different synthesis run"
        raise ValueError(msg)
    if any(path.exists() for path in artifact_paths):
        msg = f"Existing {expected.stage} artifacts are missing fingerprint {fingerprint_path}"
        raise ValueError(msg)
    write_stage_cache_fingerprint(fingerprint_path, expected)


def require_stage_cache(fingerprint_path: Path, expected: StageCacheFingerprint) -> StageCacheFingerprint:
    """Read and validate a required upstream stage fingerprint."""
    existing = read_stage_cache_fingerprint(fingerprint_path)
    if existing != expected:
        msg = f"Existing {expected.stage} artifacts belong to a different synthesis run"
        raise ValueError(msg)
    return existing


def write_stage_cache_fingerprint(output_path: Path, fingerprint: StageCacheFingerprint) -> None:
    """Atomically persist one stage fingerprint."""
    write_json_atomic(output_path, fingerprint.to_dict())


def read_stage_cache_fingerprint(input_path: Path) -> StageCacheFingerprint:
    """Read one persisted stage fingerprint."""
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if payload["schema_version"] != STAGE_CACHE_SCHEMA_VERSION:
        msg = f"Unsupported synthesis-stage cache schema: {payload['schema_version']}"
        raise ValueError(msg)
    return StageCacheFingerprint(
        stage=payload["stage"],
        run_id=payload["run_id"],
        prompt_version=payload["prompt_version"],
        model=payload["model"],
    )


def _build_fingerprint(
    *,
    stage: StageName,
    prompt_version: str,
    model: str,
    identity: dict[str, Any],
) -> StageCacheFingerprint:
    """Build one content-derived stage fingerprint."""
    run_id = sha256_json({"stage": stage, "prompt_version": prompt_version, **identity})
    return StageCacheFingerprint(stage=stage, run_id=run_id, prompt_version=prompt_version, model=model)


def _chunk_identity(chunk: Chunk, *, include_context: bool) -> dict[str, object]:
    """Return the stable chunk fields consumed by one synthesis stage."""
    identity: dict[str, object] = {
        "chunk_id": chunk.chunk_id,
        "chunker_config_sha256": chunk.chunker_config_sha256,
        "content_sha256": chunk.content_sha256,
        "record_scope": chunk.record_scope,
        "source_sha256": chunk.source_sha256,
    }
    if include_context:
        identity["context_text"] = chunk.context_text
    return identity
