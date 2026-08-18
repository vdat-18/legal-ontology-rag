"""Tests for ontology population input readiness."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from legal_rag import ontology_population_preflight_cli
from legal_rag.benchmark_synthesis.chunk_screening import ChunkScreeningRecord
from legal_rag.benchmark_synthesis.document_metadata import DocumentMetadataRecord
from legal_rag.benchmark_synthesis.writer import write_chunk_screening, write_document_metadata
from legal_rag.chunker.models import Chunk
from legal_rag.chunker.writer import write_chunks
from legal_rag.ontology_population.config import PopulationSettings
from legal_rag.ontology_population.models import OntologyPopulationRecord, PopulationProgressRecord
from legal_rag.ontology_population.preflight import (
    load_population_inputs,
    run_population_preflight,
    validate_population_resume_state,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = PROJECT_ROOT / "ontology" / "population_contract.yaml"
CHECKED_AT = datetime(2026, 8, 18, tzinfo=UTC)


def _write_inputs(
    directory: Path,
    chunks: tuple[Chunk, ...],
    screening: tuple[ChunkScreeningRecord, ...],
    metadata: DocumentMetadataRecord,
) -> tuple[Path, Path, Path]:
    """Write one complete set of typed preflight inputs."""
    chunks_path = directory / "chunks.jsonl"
    screening_path = directory / "screening.jsonl"
    metadata_path = directory / "metadata.jsonl"
    write_chunks(chunks_path, chunks)
    write_chunk_screening(screening_path, screening)
    write_document_metadata(metadata_path, metadata)
    return chunks_path, screening_path, metadata_path


def _run_report(paths: tuple[Path, Path, Path], *, target_limit: int | None = None):
    """Load inputs and return their deterministic readiness report."""
    chunks_path, screening_path, metadata_path = paths
    bundle = load_population_inputs(chunks_path, screening_path, metadata_path)
    return run_population_preflight(
        bundle,
        PopulationSettings.from_path(CONTRACT_PATH),
        (
            ("chunks", chunks_path),
            ("screening", screening_path),
            ("metadata", metadata_path),
            ("population_contract", CONTRACT_PATH),
        ),
        CHECKED_AT,
        target_limit=target_limit,
    )


def test_preflight_reports_ready_fingerprinted_pilot(
    tmp_path: Path, legal_chunk: Chunk, document_metadata: DocumentMetadataRecord
) -> None:
    """Complete structural inputs produce a reproducible pilot selection."""
    screening = (
        ChunkScreeningRecord(chunk_id=legal_chunk.chunk_id, has_penalty_content=True, reason="Confirmed by model."),
    )
    paths = _write_inputs(tmp_path, (legal_chunk,), screening, document_metadata)

    report = _run_report(paths)
    repeated = _run_report(paths)

    assert report.overall_status == "ready"
    assert report.selected_target_ids == (legal_chunk.chunk_id,)
    assert report.estimated_llm_batches == 1
    assert report.pilot_run_id == repeated.pilot_run_id
    assert {check.status for check in report.checks} == {"passed"}
    assert {artifact.artifact_type for artifact in report.artifacts} == {
        "chunks",
        "screening",
        "metadata",
        "population_contract",
    }


def test_preflight_blocks_incomplete_screening(
    tmp_path: Path, legal_chunk: Chunk, document_metadata: DocumentMetadataRecord
) -> None:
    """Every persisted chunk requires exactly one screening verdict."""
    paths = _write_inputs(tmp_path, (legal_chunk,), (), document_metadata)

    report = _run_report(paths)

    checks = {check.check_id: check for check in report.checks}
    assert report.overall_status == "blocked"
    assert checks["PF-SCREENING-001"].status == "failed"
    assert checks["PF-TARGETS-001"].status == "failed"


def test_preflight_blocks_broken_chunk_adjacency(
    tmp_path: Path, legal_chunk: Chunk, document_metadata: DocumentMetadataRecord
) -> None:
    """Reading windows cannot run when persisted neighbor links disagree with order."""
    broken = replace(legal_chunk, next_chunk_id="missing-chunk")
    screening = (ChunkScreeningRecord(broken.chunk_id, True, "Confirmed by model."),)
    paths = _write_inputs(tmp_path, (broken,), screening, document_metadata)

    report = _run_report(paths)

    checks = {check.check_id: check for check in report.checks}
    assert report.overall_status == "blocked"
    assert checks["PF-ADJACENCY-001"].status == "failed"


def test_preflight_warns_when_document_validity_has_no_baseline(tmp_path: Path, legal_chunk: Chunk) -> None:
    """Missing fallback validity remains visible without replacing source-specific extraction."""
    screening = (ChunkScreeningRecord(legal_chunk.chunk_id, True, "Confirmed by model."),)
    metadata = DocumentMetadataRecord(document_number="168/2024/NĐ-CP", valid_from=None, valid_to=None)
    paths = _write_inputs(tmp_path, (legal_chunk,), screening, metadata)

    report = _run_report(paths)

    checks = {check.check_id: check for check in report.checks}
    assert report.overall_status == "ready"
    assert checks["PF-METADATA-001"].status == "warning"


def test_preflight_selects_a_bounded_pilot_in_document_order(
    tmp_path: Path, legal_chunk: Chunk, document_metadata: DocumentMetadataRecord
) -> None:
    """A target limit selects the earliest accepted chunks without semantic scoring."""
    first = replace(legal_chunk, next_chunk_id="chunk-2")
    second = replace(
        legal_chunk,
        chunk_id="chunk-2",
        content_sha256="content-sha-2",
        chunk_index=1,
        sequence_start=1,
        sequence_end=1,
        previous_chunk_id=first.chunk_id,
    )
    screening = (
        ChunkScreeningRecord(first.chunk_id, True, "Confirmed by model."),
        ChunkScreeningRecord(second.chunk_id, True, "Confirmed by model."),
    )
    paths = _write_inputs(tmp_path, (first, second), screening, document_metadata)

    report = _run_report(paths, target_limit=1)

    assert report.overall_status == "ready"
    assert report.accepted_chunk_count == 2
    assert report.selected_target_ids == (first.chunk_id,)
    assert report.estimated_llm_batches == 1


def test_preflight_cli_persists_blocked_report_before_failing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    legal_chunk: Chunk,
) -> None:
    """Operators receive a durable report for cross-file readiness failures."""
    metadata = DocumentMetadataRecord(document_number="", valid_from=date(2025, 1, 1), valid_to=None)
    screening = (ChunkScreeningRecord(legal_chunk.chunk_id, True, "Confirmed by model."),)
    chunks_path, screening_path, metadata_path = _write_inputs(tmp_path, (legal_chunk,), screening, metadata)
    report_path = tmp_path / "preflight.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "legal-check-ontology-population",
            str(chunks_path),
            str(screening_path),
            str(metadata_path),
            str(CONTRACT_PATH),
            str(report_path),
        ],
    )

    with pytest.raises(ValueError, match="inputs are blocked"):
        ontology_population_preflight_cli.main()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["overall_status"] == "blocked"
    assert report["document_number"] == ""


def test_resume_state_rejects_records_outside_bounded_pilot(
    legal_chunk: Chunk,
    population_record: OntologyPopulationRecord,
) -> None:
    """A bounded pilot cannot silently rematerialize records from a larger previous run."""
    other_record = replace(
        population_record,
        population_id="population-outside-pilot",
        source_chunk_id="chunk-outside-pilot",
    )

    with pytest.raises(ValueError, match="outside the current target selection"):
        validate_population_resume_state((legal_chunk,), (population_record, other_record), ())


def test_resume_state_rejects_duplicate_progress(legal_chunk: Chunk) -> None:
    """Duplicate checkpoints cannot be collapsed silently during resume."""
    progress = PopulationProgressRecord(chunk_id=legal_chunk.chunk_id)

    with pytest.raises(ValueError, match="duplicate progress chunk IDs: 1"):
        validate_population_resume_state((legal_chunk,), (), (progress, progress))
