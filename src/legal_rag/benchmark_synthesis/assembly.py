"""Validate document artifacts and assemble one corpus-level legal-fact file."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from legal_rag.benchmark_synthesis.candidate_generation import CANDIDATE_GENERATION_PROMPT_VERSION
from legal_rag.benchmark_synthesis.chunk_screening import CHUNK_SCREENING_PROMPT_VERSION
from legal_rag.benchmark_synthesis.document_metadata import DOCUMENT_METADATA_PROMPT_VERSION
from legal_rag.benchmark_synthesis.evidence_graph import EVIDENCE_RELATION_PROMPT_VERSION
from legal_rag.benchmark_synthesis.fact_extraction import FACT_EXTRACTION_PROMPT_VERSION
from legal_rag.benchmark_synthesis.models import LegalFact
from legal_rag.benchmark_synthesis.writer import (
    read_chunk_screening,
    read_document_metadata,
    read_extraction_progress,
    read_legal_facts,
)
from legal_rag.chunker.models import CHUNK_SCHEMA_VERSION
from legal_rag.chunker.writer import read_chunks
from legal_rag.parser.hashing import sha256_json

RUN_MANIFEST_SCHEMA_VERSION = "1.0.0"


class GroundTruthValidationError(ValueError):
    """Raised when an artifact set is incomplete or inconsistent."""


@dataclass(frozen=True, slots=True)
class AssemblyResult:
    """Validated corpus-level facts and their reproducibility manifest."""

    facts: tuple[LegalFact, ...]
    manifest: dict[str, Any]


def assemble_groundtruth_corpus(
    processed_dir: Path,
    corpus_config_path: Path,
    snapshot_path: Path,
    synthesis_config_path: Path,
    config_paths: Sequence[Path],
    *,
    llm_model: str,
) -> AssemblyResult:
    """Validate every configured document and assemble deterministic corpus-level facts."""
    corpus_config = json.loads(corpus_config_path.read_text(encoding="utf-8"))
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    document_ids = tuple(Path(document["file"]).stem for document in corpus_config["documents"])
    facts_by_id: dict[str, LegalFact] = {}
    chunker_config_hashes: set[str] = set()
    document_summaries: list[dict[str, object]] = []

    for document_id in document_ids:
        document_facts, summary, document_chunker_hashes = _validate_document(processed_dir, document_id)
        chunker_config_hashes.update(document_chunker_hashes)
        document_summaries.append(summary)
        for fact in document_facts:
            existing = facts_by_id.get(fact.fact_id)
            if existing is not None and existing.fact_content_sha256 != fact.fact_content_sha256:
                msg = f"Conflicting content for fact_id {fact.fact_id}"
                raise GroundTruthValidationError(msg)
            facts_by_id.setdefault(fact.fact_id, fact)

    prompt_versions = {
        "candidate_generation": CANDIDATE_GENERATION_PROMPT_VERSION,
        "chunk_screening": CHUNK_SCREENING_PROMPT_VERSION,
        "document_metadata": DOCUMENT_METADATA_PROMPT_VERSION,
        "evidence_relation": EVIDENCE_RELATION_PROMPT_VERSION,
        "fact_extraction": FACT_EXTRACTION_PROMPT_VERSION,
    }
    config_hashes = {
        path.as_posix(): _sha256_file(path) for path in (corpus_config_path, synthesis_config_path, *config_paths)
    }
    identity = {
        "chunk_schema_version": CHUNK_SCHEMA_VERSION,
        "chunker_config_sha256": sorted(chunker_config_hashes),
        "config_sha256": config_hashes,
        "corpus_snapshot_id": snapshot["snapshot_id"],
        "documents": document_ids,
        "fact_ledger_sha256": sha256_json(
            sorted((fact.fact_id, fact.fact_content_sha256) for fact in facts_by_id.values())
        ),
        "llm_model": llm_model,
        "prompt_versions": prompt_versions,
    }
    manifest: dict[str, Any] = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "run_id": sha256_json(identity),
        **identity,
        "complete": True,
        "created_at": datetime.now(UTC).isoformat(),
        "document_summaries": document_summaries,
        "fact_count": len(facts_by_id),
    }
    return AssemblyResult(facts=tuple(facts_by_id.values()), manifest=manifest)


def _validate_document(
    processed_dir: Path,
    document_id: str,
) -> tuple[tuple[LegalFact, ...], dict[str, object], set[str]]:
    """Validate one document's phase-one artifacts and return its facts."""
    paths = {
        "chunks": processed_dir / f"{document_id}.chunks.jsonl",
        "screening": processed_dir / f"{document_id}.chunk_screening.jsonl",
        "metadata": processed_dir / f"{document_id}.document_metadata.jsonl",
        "facts": processed_dir / f"{document_id}.legal_facts.jsonl",
        "progress": processed_dir / f"{document_id}.extraction_progress.jsonl",
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        msg = f"Document {document_id} is missing artifacts: {', '.join(missing)}"
        raise GroundTruthValidationError(msg)

    chunks = read_chunks(paths["chunks"])
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    screening = read_chunk_screening(paths["screening"])
    screening_ids = {record.chunk_id for record in screening}
    if screening_ids != set(chunks_by_id):
        msg = f"Document {document_id} screening is incomplete or references stale chunks"
        raise GroundTruthValidationError(msg)

    accepted_ids = {record.chunk_id for record in screening if record.has_penalty_content}
    progress_ids = {record.chunk_id for record in read_extraction_progress(paths["progress"])}
    if progress_ids != accepted_ids:
        msg = f"Document {document_id} extraction progress does not cover every accepted chunk"
        raise GroundTruthValidationError(msg)

    metadata = read_document_metadata(paths["metadata"])
    facts = read_legal_facts(paths["facts"])
    for fact in facts:
        _validate_fact(document_id, fact, metadata.document_number, chunks_by_id)

    summary: dict[str, object] = {
        "document_id": document_id,
        "chunk_count": len(chunks),
        "accepted_chunk_count": len(accepted_ids),
        "fact_count": len(facts),
    }
    return facts, summary, {chunk.chunker_config_sha256 for chunk in chunks}


def _validate_fact(document_id: str, fact: LegalFact, document_number: str, chunks_by_id: dict[str, Any]) -> None:
    """Reject one fact with stale provenance, conflicting metadata, or non-exact evidence."""
    if fact.document_number != document_number:
        msg = f"Fact {fact.fact_id} has document_number inconsistent with {document_id} metadata"
        raise GroundTruthValidationError(msg)
    if fact.source_chunk_id not in chunks_by_id:
        msg = f"Fact {fact.fact_id} references missing source_chunk_id {fact.source_chunk_id}"
        raise GroundTruthValidationError(msg)
    if not fact.evidence:
        msg = f"Fact {fact.fact_id} has no evidence"
        raise GroundTruthValidationError(msg)
    for evidence in fact.evidence:
        chunk = chunks_by_id.get(evidence.chunk_id)
        if chunk is None:
            msg = f"Fact {fact.fact_id} references missing chunk_id {evidence.chunk_id}"
            raise GroundTruthValidationError(msg)
        if not evidence.quoted_span or evidence.quoted_span not in chunk.text:
            msg = f"Fact {fact.fact_id} has a non-exact quoted span in chunk {evidence.chunk_id}"
            raise GroundTruthValidationError(msg)


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one configuration file."""
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for block in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
