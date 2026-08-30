"""Structural readiness checks for ontology population inputs."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from legal_rag.benchmark_synthesis.chunk_screening import ChunkScreeningRecord
from legal_rag.benchmark_synthesis.document_metadata import DocumentMetadataRecord
from legal_rag.benchmark_synthesis.writer import read_chunk_screening, read_document_metadata
from legal_rag.chunker.models import Chunk
from legal_rag.chunker.writer import read_chunks
from legal_rag.ontology_population.config import PopulationSettings
from legal_rag.ontology_population.models import OntologyPopulationRecord, PopulationProgressRecord
from legal_rag.parser.hashing import sha256_json

POPULATION_PREFLIGHT_SCHEMA_VERSION = "0.2.0"
POPULATION_TARGET_SELECTION_SCHEMA_VERSION = "0.1.0"

CheckStatus = Literal["passed", "failed", "warning"]


@dataclass(frozen=True, slots=True)
class PopulationInputBundle:
    """Typed source artifacts required by one population run."""

    chunks: tuple[Chunk, ...]
    screening: tuple[ChunkScreeningRecord, ...]
    metadata: DocumentMetadataRecord


@dataclass(frozen=True, slots=True)
class PopulationTargetSelection:
    """Reviewed chunk IDs selected for one population pilot."""

    document_number: str
    chunk_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PopulationInputArtifact:
    """One fingerprinted input used by a population preflight."""

    artifact_type: str
    path: str
    sha256: str
    size_bytes: int

    def to_dict(self) -> dict[str, str | int]:
        """Return a JSON-serializable input fingerprint."""
        return {
            "artifact_type": self.artifact_type,
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class PopulationPreflightCheck:
    """One structural population-readiness result."""

    check_id: str
    status: CheckStatus
    summary: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serializable readiness result."""
        return {"check_id": self.check_id, "status": self.status, "summary": self.summary}


@dataclass(frozen=True, slots=True)
class PopulationPreflightReport:
    """Machine-readable readiness and pilot-selection report."""

    checked_at: datetime
    overall_status: Literal["ready", "blocked"]
    pilot_run_id: str
    ontology_semantic_sha256: str
    document_number: str
    chunk_count: int
    screening_count: int
    accepted_chunk_count: int
    selected_target_ids: tuple[str, ...]
    estimated_llm_batches: int
    artifacts: tuple[PopulationInputArtifact, ...]
    checks: tuple[PopulationPreflightCheck, ...]

    def to_dict(self) -> dict[str, object]:
        """Return the stable population-preflight report schema."""
        return {
            "schema_version": POPULATION_PREFLIGHT_SCHEMA_VERSION,
            "checked_at": self.checked_at.isoformat(),
            "overall_status": self.overall_status,
            "pilot_run_id": self.pilot_run_id,
            "ontology_semantic_sha256": self.ontology_semantic_sha256,
            "document_number": self.document_number,
            "chunk_count": self.chunk_count,
            "screening_count": self.screening_count,
            "accepted_chunk_count": self.accepted_chunk_count,
            "selected_target_ids": list(self.selected_target_ids),
            "estimated_llm_batches": self.estimated_llm_batches,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "checks": [check.to_dict() for check in self.checks],
        }


def load_population_inputs(chunks_path: Path, screening_path: Path, metadata_path: Path) -> PopulationInputBundle:
    """Load the three typed population input artifacts."""
    return PopulationInputBundle(
        chunks=read_chunks(chunks_path),
        screening=read_chunk_screening(screening_path),
        metadata=read_document_metadata(metadata_path),
    )


def population_ontology_semantic_sha256(project_root: Path) -> str:
    """Hash the complete versioned ontology input closure used by population."""
    from legal_rag.ontology_validation.paths import OntologyPaths
    from legal_rag.ontology_validation.semantic_config import SemanticQaSettings
    from legal_rag.ontology_validation.semantic_qa import semantic_input_sha256

    paths = OntologyPaths(project_root)
    settings = SemanticQaSettings.from_path(paths.semantic_qa_contract, project_root)
    return semantic_input_sha256(paths, settings)


def load_population_target_selection(path: Path) -> PopulationTargetSelection:
    """Load one versioned population-target selection."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["schema_version"] != POPULATION_TARGET_SELECTION_SCHEMA_VERSION:
        msg = f"Unsupported population target selection schema: {payload['schema_version']}"
        raise ValueError(msg)
    return PopulationTargetSelection(
        document_number=payload["document_number"],
        chunk_ids=tuple(payload["chunk_ids"]),
    )


def select_population_targets(
    bundle: PopulationInputBundle,
    target_limit: int | None,
    target_selection: PopulationTargetSelection | None = None,
) -> tuple[Chunk, ...]:
    """Select accepted targets in stable document order for a full or bounded pilot run."""
    if target_limit is not None and target_limit < 1:
        msg = "Population target limit must be at least one"
        raise ValueError(msg)
    accepted_ids = {record.chunk_id for record in bundle.screening if record.has_penalty_content}
    accepted_chunks = tuple(chunk for chunk in bundle.chunks if chunk.chunk_id in accepted_ids)
    if target_selection is not None:
        if target_limit is not None:
            msg = "Population target limit and target selection are mutually exclusive"
            raise ValueError(msg)
        if target_selection.document_number != bundle.metadata.document_number:
            msg = "Population target selection document number does not match metadata"
            raise ValueError(msg)
        duplicate_ids = _duplicates(target_selection.chunk_ids)
        if duplicate_ids:
            msg = f"Duplicate population target IDs: {', '.join(duplicate_ids)}"
            raise ValueError(msg)
        chunks_by_id = {chunk.chunk_id: chunk for chunk in bundle.chunks}
        unknown_ids = set(target_selection.chunk_ids) - set(chunks_by_id)
        rejected_ids = set(target_selection.chunk_ids) - accepted_ids
        if unknown_ids or rejected_ids:
            msg = (
                f"Invalid population target selection: {len(unknown_ids)} unknown IDs, "
                f"{len(rejected_ids - unknown_ids)} screening-rejected IDs"
            )
            raise ValueError(msg)
        return tuple(chunks_by_id[chunk_id] for chunk_id in target_selection.chunk_ids)
    return accepted_chunks if target_limit is None else accepted_chunks[:target_limit]


def validate_population_resume_state(
    selected_targets: Sequence[Chunk],
    records: Sequence[OntologyPopulationRecord],
    progress: Sequence[PopulationProgressRecord],
    *,
    expected_run_id: str | None = None,
    expected_ontology_semantic_sha256: str | None = None,
) -> None:
    """Require persisted outputs to belong uniquely to the current target selection."""
    duplicate_population_ids = _duplicates(record.population_id for record in records)
    duplicate_progress_ids = _duplicates(record.chunk_id for record in progress)
    if duplicate_population_ids or duplicate_progress_ids:
        msg = (
            f"Duplicate persisted population IDs: {len(duplicate_population_ids)}; "
            f"duplicate progress chunk IDs: {len(duplicate_progress_ids)}"
        )
        raise ValueError(msg)
    selected_ids = {chunk.chunk_id for chunk in selected_targets}
    unexpected_record_ids = {record.source_chunk_id for record in records} - selected_ids
    unexpected_progress_ids = {record.chunk_id for record in progress} - selected_ids
    if unexpected_record_ids or unexpected_progress_ids:
        msg = (
            f"Persisted population state is outside the current target selection: "
            f"{len(unexpected_record_ids)} record source IDs, {len(unexpected_progress_ids)} progress IDs"
        )
        raise ValueError(msg)
    if expected_run_id is not None and any(record.run_id != expected_run_id for record in progress):
        msg = "Persisted population progress belongs to a different prompt, model, input, or ontology run"
        raise ValueError(msg)
    if expected_run_id is not None and any(record.population_run_id != expected_run_id for record in records):
        msg = "Persisted population records belong to a different prompt, model, input, or ontology run"
        raise ValueError(msg)
    if expected_ontology_semantic_sha256 is not None and any(
        record.ontology_semantic_sha256 != expected_ontology_semantic_sha256 for record in records
    ):
        msg = "Persisted population records belong to different ontology semantics"
        raise ValueError(msg)


def run_population_preflight(
    bundle: PopulationInputBundle,
    settings: PopulationSettings,
    artifact_paths: Sequence[tuple[str, Path]],
    checked_at: datetime,
    *,
    ontology_semantic_sha256: str,
    target_limit: int | None = None,
    target_selection: PopulationTargetSelection | None = None,
) -> PopulationPreflightReport:
    """Assess structural readiness without making semantic or external-service decisions."""
    selected_targets = select_population_targets(bundle, target_limit, target_selection)
    artifacts = tuple(_fingerprint(artifact_type, path) for artifact_type, path in artifact_paths)
    checks = (
        _chunk_inventory_check(bundle.chunks),
        _document_scope_check(bundle.chunks),
        _adjacency_check(bundle.chunks),
        _screening_coverage_check(bundle.chunks, bundle.screening),
        _target_availability_check(bundle.screening, selected_targets),
        _metadata_check(bundle.metadata),
    )
    overall_status = "blocked" if any(check.status == "failed" for check in checks) else "ready"
    pilot_run_id = sha256_json(
        {
            "artifacts": [{"artifact_type": item.artifact_type, "sha256": item.sha256} for item in artifacts],
            "ontology_semantic_sha256": ontology_semantic_sha256,
            "selected_target_ids": [chunk.chunk_id for chunk in selected_targets],
        }
    )
    accepted_count = sum(record.has_penalty_content for record in bundle.screening)
    return PopulationPreflightReport(
        checked_at=checked_at,
        overall_status=overall_status,
        pilot_run_id=pilot_run_id,
        ontology_semantic_sha256=ontology_semantic_sha256,
        document_number=bundle.metadata.document_number,
        chunk_count=len(bundle.chunks),
        screening_count=len(bundle.screening),
        accepted_chunk_count=accepted_count,
        selected_target_ids=tuple(chunk.chunk_id for chunk in selected_targets),
        estimated_llm_batches=math.ceil(len(selected_targets) / settings.batch_size),
        artifacts=artifacts,
        checks=checks,
    )


def _fingerprint(artifact_type: str, path: Path) -> PopulationInputArtifact:
    """Fingerprint one immutable run input."""
    content = path.read_bytes()
    return PopulationInputArtifact(
        artifact_type=artifact_type,
        path=str(path.resolve()),
        sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
    )


def _chunk_inventory_check(chunks: Sequence[Chunk]) -> PopulationPreflightCheck:
    """Require a non-empty chunk file with unique identifiers."""
    duplicate_ids = _duplicates(chunk.chunk_id for chunk in chunks)
    if not chunks:
        return PopulationPreflightCheck("PF-CHUNKS-001", "failed", "Chunk input is empty.")
    if duplicate_ids:
        return PopulationPreflightCheck("PF-CHUNKS-001", "failed", f"Duplicate chunk IDs: {', '.join(duplicate_ids)}")
    return PopulationPreflightCheck("PF-CHUNKS-001", "passed", f"Loaded {len(chunks)} unique chunks.")


def _document_scope_check(chunks: Sequence[Chunk]) -> PopulationPreflightCheck:
    """Require one root document identity while allowing attachment-child scopes."""
    root_identities = {
        (chunk.scope_id, chunk.source_name, chunk.source_sha256) for chunk in chunks if chunk.record_scope == "root"
    }
    if len(root_identities) != 1:
        return PopulationPreflightCheck(
            "PF-DOCUMENT-001",
            "failed",
            f"Expected one root document identity, found {len(root_identities)}.",
        )
    return PopulationPreflightCheck("PF-DOCUMENT-001", "passed", "Chunks belong to one root legal document.")


def _adjacency_check(chunks: Sequence[Chunk]) -> PopulationPreflightCheck:
    """Require each scope's persisted neighbor links to match chunk order."""
    by_scope: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        by_scope[chunk.scope_id].append(chunk)
    issues: list[str] = []
    for scope_id, scope_chunks in by_scope.items():
        ordered = sorted(scope_chunks, key=lambda chunk: chunk.chunk_index)
        if _duplicates(chunk.chunk_index for chunk in ordered):
            issues.append(f"{scope_id}:duplicate-index")
            continue
        for index, chunk in enumerate(ordered):
            expected_previous = ordered[index - 1].chunk_id if index else None
            expected_next = ordered[index + 1].chunk_id if index + 1 < len(ordered) else None
            if chunk.previous_chunk_id != expected_previous or chunk.next_chunk_id != expected_next:
                issues.append(chunk.chunk_id)
    if issues:
        return PopulationPreflightCheck(
            "PF-ADJACENCY-001",
            "failed",
            f"Invalid previous/next links for: {', '.join(sorted(issues))}",
        )
    return PopulationPreflightCheck(
        "PF-ADJACENCY-001", "passed", f"Neighbor links are consistent across {len(by_scope)} scopes."
    )


def _screening_coverage_check(
    chunks: Sequence[Chunk], screening: Sequence[ChunkScreeningRecord]
) -> PopulationPreflightCheck:
    """Require exactly one screening verdict for every chunk and no foreign IDs."""
    chunk_ids = {chunk.chunk_id for chunk in chunks}
    screening_ids = [record.chunk_id for record in screening]
    duplicate_ids = set(_duplicates(screening_ids))
    missing_ids = chunk_ids - set(screening_ids)
    unknown_ids = set(screening_ids) - chunk_ids
    if duplicate_ids or missing_ids or unknown_ids:
        summary = (
            f"Screening coverage mismatch: {len(missing_ids)} missing, "
            f"{len(unknown_ids)} unknown, {len(duplicate_ids)} duplicate IDs."
        )
        return PopulationPreflightCheck("PF-SCREENING-001", "failed", summary)
    return PopulationPreflightCheck(
        "PF-SCREENING-001", "passed", f"Screening covers all {len(chunks)} chunks exactly once."
    )


def _target_availability_check(
    screening: Sequence[ChunkScreeningRecord], selected_targets: Sequence[Chunk]
) -> PopulationPreflightCheck:
    """Require the screening stage to provide at least one pilot population target."""
    accepted_count = sum(record.has_penalty_content for record in screening)
    if accepted_count == 0:
        return PopulationPreflightCheck("PF-TARGETS-001", "failed", "Screening accepted no population targets.")
    return PopulationPreflightCheck(
        "PF-TARGETS-001",
        "passed",
        f"Selected {len(selected_targets)} of {accepted_count} accepted chunks for this run.",
    )


def _metadata_check(metadata: DocumentMetadataRecord) -> PopulationPreflightCheck:
    """Require document identity and flag a missing validity baseline."""
    if not metadata.document_number.strip():
        return PopulationPreflightCheck("PF-METADATA-001", "failed", "Document number is empty.")
    if metadata.valid_from is None:
        return PopulationPreflightCheck(
            "PF-METADATA-001",
            "warning",
            "Document valid_from is unknown; every populated rule must supply source-specific validity.",
        )
    if metadata.valid_to is not None and metadata.valid_to < metadata.valid_from:
        return PopulationPreflightCheck("PF-METADATA-001", "failed", "Document valid_to precedes valid_from.")
    return PopulationPreflightCheck(
        "PF-METADATA-001", "passed", f"Document metadata has a validity baseline from {metadata.valid_from}."
    )


def _duplicates(values: Iterable[object]) -> tuple[str, ...]:
    """Return stable string representations of repeated values."""
    counts = Counter(values)
    return tuple(sorted(str(value) for value, count in counts.items() if count > 1))
