"""Tests for corpus-level ground-truth assembly and validation."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from legal_rag.benchmark_synthesis.assembly import GroundTruthValidationError, assemble_groundtruth_corpus
from legal_rag.benchmark_synthesis.chunk_screening import ChunkScreeningRecord
from legal_rag.benchmark_synthesis.document_metadata import DocumentMetadataRecord
from legal_rag.benchmark_synthesis.fact_extraction import ExtractionProgressRecord
from legal_rag.benchmark_synthesis.models import EvidenceReference, LegalFact
from legal_rag.benchmark_synthesis.writer import (
    write_chunk_screening,
    write_document_metadata,
    write_extraction_progress,
    write_legal_facts,
)
from legal_rag.chunker.models import Chunk
from legal_rag.chunker.writer import write_chunks

_DOCUMENT_ID = "168_2024_ND-CP_619502"
_SPAN = "Phat tien tu 4.000.000 dong den 6.000.000 dong."
_SYNTHESIS_CONFIG_PATH = Path(__file__).resolve().parents[2] / "data" / "config" / "groundtruth" / "synthesis_v1.json"


def _chunk() -> Chunk:
    """Build one chunk used by a complete assembly fixture."""
    return Chunk(
        chunk_id="chunk-1",
        content_sha256="content-hash",
        chunker_config_sha256="chunker-config-hash",
        scope_id="scope-id",
        chunk_index=0,
        kind="paragraph_group",
        source_name=f"{_DOCUMENT_ID}.docx",
        source_sha256="source-hash",
        record_scope="root",
        parent_attachment_id=None,
        attachment_member_path=None,
        attachment_member_sha256=None,
        unit_ids=("unit-0",),
        source_refs=("word/document.xml#/body/0",),
        sequence_start=0,
        sequence_end=0,
        text=_SPAN,
        context_text="",
        embedding_text=_SPAN,
        text_token_count=10,
        embedding_token_count=10,
        previous_chunk_id=None,
        next_chunk_id=None,
    )


def _fact() -> LegalFact:
    """Build one fact with exact chunk provenance."""
    return LegalFact(
        fact_id="fact-1",
        fact_content_sha256="fact-content-hash",
        conduct="Vuot den do",
        subject="Nguoi dieu khien oto",
        conditions=(),
        monetary_penalty="4.000.000-6.000.000 dong",
        license_points_deducted=None,
        additional_penalties=(),
        remedial_measures=(),
        valid_from=None,
        valid_to=None,
        document_number="168/2024/ND-CP",
        provision_path="Dieu 6",
        evidence=(EvidenceReference(chunk_id="chunk-1", quoted_span=_SPAN),),
        source_chunk_id="chunk-1",
    )


def _write_fixture(tmp_path: Path, *, facts: tuple[LegalFact, ...] | None = None) -> tuple[Path, Path, Path]:
    """Write one complete document phase plus corpus config and snapshot."""
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    write_chunks(processed_dir / f"{_DOCUMENT_ID}.chunks.jsonl", [_chunk()])
    write_chunk_screening(
        processed_dir / f"{_DOCUMENT_ID}.chunk_screening.jsonl",
        [ChunkScreeningRecord(chunk_id="chunk-1", has_penalty_content=True, reason="Penalty content")],
    )
    write_document_metadata(
        processed_dir / f"{_DOCUMENT_ID}.document_metadata.jsonl",
        DocumentMetadataRecord(document_number="168/2024/ND-CP", valid_from=None, valid_to=None),
    )
    write_legal_facts(processed_dir / f"{_DOCUMENT_ID}.legal_facts.jsonl", facts or (_fact(),))
    write_extraction_progress(
        processed_dir / f"{_DOCUMENT_ID}.extraction_progress.jsonl",
        [ExtractionProgressRecord(chunk_id="chunk-1")],
    )
    config_path = tmp_path / "corpus.json"
    config_path.write_text(json.dumps({"documents": [{"file": f"{_DOCUMENT_ID}.docx"}]}), encoding="utf-8")
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(json.dumps({"snapshot_id": "snapshot-1"}), encoding="utf-8")
    return processed_dir, config_path, snapshot_path


def _assemble(tmp_path: Path, *, facts: tuple[LegalFact, ...] | None = None):
    """Assemble the fixture with one versioned config hash input."""
    processed_dir, config_path, snapshot_path = _write_fixture(tmp_path, facts=facts)
    return assemble_groundtruth_corpus(
        processed_dir,
        config_path,
        snapshot_path,
        _SYNTHESIS_CONFIG_PATH,
        [config_path],
        llm_model="provider/test-model",
    )


def test_assembly_emits_complete_reproducibility_manifest(tmp_path: Path) -> None:
    """A complete phase assembles facts and records snapshot, model, prompts, and config hashes."""
    result = _assemble(tmp_path)

    assert result.facts == (_fact(),)
    assert result.manifest["complete"] is True
    assert result.manifest["corpus_snapshot_id"] == "snapshot-1"
    assert result.manifest["llm_model"] == "provider/test-model"
    assert result.manifest["prompt_versions"]
    assert result.manifest["config_sha256"]
    assert _SYNTHESIS_CONFIG_PATH.as_posix() in result.manifest["config_sha256"]


def test_assembly_rejects_incomplete_extraction_progress(tmp_path: Path) -> None:
    """An accepted screening task without extraction progress blocks assembly."""
    processed_dir, config_path, snapshot_path = _write_fixture(tmp_path)
    write_extraction_progress(processed_dir / f"{_DOCUMENT_ID}.extraction_progress.jsonl", [])

    with pytest.raises(GroundTruthValidationError, match="progress does not cover"):
        assemble_groundtruth_corpus(
            processed_dir, config_path, snapshot_path, _SYNTHESIS_CONFIG_PATH, [], llm_model="model"
        )


def test_assembly_rejects_missing_evidence_chunk(tmp_path: Path) -> None:
    """A fact cannot reference a chunk absent from the document artifact."""
    fact = replace(
        _fact(),
        evidence=(EvidenceReference(chunk_id="chunk-missing", quoted_span=_SPAN),),
    )

    with pytest.raises(GroundTruthValidationError, match="missing chunk_id"):
        _assemble(tmp_path, facts=(fact,))


def test_assembly_rejects_non_exact_evidence_span(tmp_path: Path) -> None:
    """Assembly requires the persisted quote to occur exactly in its source chunk."""
    fact = replace(
        _fact(),
        evidence=(EvidenceReference(chunk_id="chunk-1", quoted_span="not present"),),
    )

    with pytest.raises(GroundTruthValidationError, match="non-exact quoted span"):
        _assemble(tmp_path, facts=(fact,))


def test_assembly_rejects_duplicate_fact_identity_with_conflicting_content(tmp_path: Path) -> None:
    """The same fact id cannot silently represent two different fact contents."""
    conflict = replace(_fact(), fact_content_sha256="different-content", monetary_penalty="different")

    with pytest.raises(GroundTruthValidationError, match="Conflicting content"):
        _assemble(tmp_path, facts=(_fact(), conflict))
