"""Tests for synthesis-stage cache fingerprints."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from legal_rag.benchmark_synthesis.chunk_screening import ChunkScreeningRecord
from legal_rag.benchmark_synthesis.config import SynthesisSettings
from legal_rag.benchmark_synthesis.document_metadata import DocumentMetadataRecord
from legal_rag.benchmark_synthesis.fact_extraction import ExtractionProgressRecord
from legal_rag.benchmark_synthesis.models import EvidenceReference, LegalFact
from legal_rag.benchmark_synthesis.scenario_sampling import Scenario
from legal_rag.benchmark_synthesis.stage_cache import (
    build_candidate_generation_fingerprint,
    build_chunk_screening_artifact_fingerprint,
    build_chunk_screening_fingerprint,
    build_document_metadata_fingerprint,
    build_fact_extraction_artifact_fingerprint,
    build_fact_extraction_fingerprint,
    initialize_stage_cache,
    read_stage_cache_fingerprint,
)
from legal_rag.chunker.models import Chunk

_SYNTHESIS_CONFIG_PATH = Path(__file__).resolve().parents[2] / "data" / "config" / "groundtruth" / "synthesis_v1.json"


def _chunk(content_sha256: str = "content-hash", *, context_text: str = "") -> Chunk:
    """Build one chunk whose source identity can be changed per test."""
    return Chunk(
        chunk_id="chunk-1",
        content_sha256=content_sha256,
        chunker_config_sha256="config-hash",
        scope_id="scope-id",
        chunk_index=0,
        kind="paragraph_group",
        source_name="source.docx",
        source_sha256="source-hash",
        record_scope="root",
        parent_attachment_id=None,
        attachment_member_path=None,
        attachment_member_sha256=None,
        unit_ids=("unit-1",),
        source_refs=("word/document.xml#/body/0",),
        sequence_start=0,
        sequence_end=0,
        text="Phạt tiền.",
        context_text=context_text,
        embedding_text="Phạt tiền.",
        text_token_count=3,
        embedding_token_count=3,
        previous_chunk_id=None,
        next_chunk_id=None,
    )


def _fact() -> LegalFact:
    """Build one fact referenced by the candidate-stage fingerprint."""
    return LegalFact(
        fact_id="fact-1",
        statement="Người điều khiển ô tô vượt đèn đỏ bị phạt tiền.",
        valid_from=date(2025, 1, 1),
        valid_to=None,
        document_number="168/2024/NĐ-CP",
        evidence=(EvidenceReference(chunk_id="chunk-1"),),
        source_chunk_id="chunk-1",
    )


def _scenario() -> Scenario:
    """Build one scenario referenced by the candidate-stage fingerprint."""
    return Scenario(
        scenario_id="scenario-1",
        query_form="specific",
        answerability="answerable",
        persona="tai_xe_ca_nhan",
        query_style="standard",
        query_length="short",
        reasoning_hops=1,
        fact_ids=("fact-1",),
        bridge=None,
        gold_chunk_ids=("chunk-1",),
    )


def test_screening_fingerprint_changes_with_source_or_model() -> None:
    """A cache becomes stale when source content or the serving model changes."""
    settings = SynthesisSettings.from_path(_SYNTHESIS_CONFIG_PATH)
    original = build_chunk_screening_fingerprint(
        (_chunk(),), settings, llm_base_url="https://llm.test/v1", llm_model="provider/model-a"
    )
    changed_source = build_chunk_screening_fingerprint(
        (_chunk("changed-content"),), settings, llm_base_url="https://llm.test/v1", llm_model="provider/model-a"
    )
    changed_model = build_chunk_screening_fingerprint(
        (_chunk(),), settings, llm_base_url="https://llm.test/v1", llm_model="provider/model-b"
    )

    assert original.run_id != changed_source.run_id
    assert original.run_id != changed_model.run_id


def test_screening_fingerprint_changes_with_structural_context() -> None:
    """Screening caches become stale when parser-derived legal structure changes."""
    settings = SynthesisSettings.from_path(_SYNTHESIS_CONFIG_PATH)
    original = build_chunk_screening_fingerprint(
        (_chunk(),), settings, llm_base_url="https://llm.test/v1", llm_model="provider/model-a"
    )
    changed = build_chunk_screening_fingerprint(
        (_chunk(context_text="Điều 6"),),
        settings,
        llm_base_url="https://llm.test/v1",
        llm_model="provider/model-a",
    )

    assert original.run_id != changed.run_id


def test_screening_fingerprint_changes_with_prompt_facing_provenance() -> None:
    """Screening caches become stale when provenance shown to the model changes."""
    settings = SynthesisSettings.from_path(_SYNTHESIS_CONFIG_PATH)
    chunk = _chunk()
    original = build_chunk_screening_fingerprint(
        (chunk,), settings, llm_base_url="https://llm.test/v1", llm_model="provider/model-a"
    )

    for changed_chunk in (
        replace(chunk, source_name="attachment.doc"),
        replace(chunk, attachment_member_path="word/embeddings/attachment.doc"),
        replace(chunk, kind="table"),
    ):
        changed = build_chunk_screening_fingerprint(
            (changed_chunk,), settings, llm_base_url="https://llm.test/v1", llm_model="provider/model-a"
        )
        assert original.run_id != changed.run_id


def test_screening_fingerprint_changes_with_stage_config() -> None:
    """A cache becomes stale when reproducibility-sensitive stage settings change."""
    settings = SynthesisSettings.from_path(_SYNTHESIS_CONFIG_PATH)
    original = build_chunk_screening_fingerprint(
        (_chunk(),), settings, llm_base_url="https://llm.test/v1", llm_model="provider/model-a"
    )
    changed_config = build_chunk_screening_fingerprint(
        (_chunk(),),
        replace(settings, chunk_screening_batch_size=settings.chunk_screening_batch_size + 1),
        llm_base_url="https://llm.test/v1",
        llm_model="provider/model-a",
    )

    assert original.run_id != changed_config.run_id


def test_screening_fingerprint_changes_with_reasoning_effort() -> None:
    """Resume rejects screening artifacts generated with another reasoning effort."""
    settings = SynthesisSettings.from_path(_SYNTHESIS_CONFIG_PATH)
    original = build_chunk_screening_fingerprint(
        (_chunk(),), settings, llm_base_url="https://llm.test/v1", llm_model="provider/model-a"
    )
    changed = build_chunk_screening_fingerprint(
        (_chunk(),),
        replace(settings, chunk_screening=replace(settings.chunk_screening, reasoning_effort="high")),
        llm_base_url="https://llm.test/v1",
        llm_model="provider/model-a",
    )

    assert original.run_id != changed.run_id


def test_screening_artifact_fingerprint_changes_with_one_replaced_verdict() -> None:
    """A selective verdict replacement invalidates every downstream stage cache."""
    original = build_chunk_screening_artifact_fingerprint(
        (_chunk(),),
        (ChunkScreeningRecord("chunk-1", True, "Accepted.", prompt_version="5.8.0", screening_run_id="run-a"),),
    )
    changed = build_chunk_screening_artifact_fingerprint(
        (_chunk(),),
        (ChunkScreeningRecord("chunk-1", False, "Rejected.", prompt_version="5.9.0", screening_run_id="run-b"),),
    )

    assert original.run_id != changed.run_id


def test_metadata_fingerprint_changes_with_screening_run() -> None:
    """Metadata selection becomes stale when upstream screening changes."""
    settings = SynthesisSettings.from_path(_SYNTHESIS_CONFIG_PATH)
    original = build_document_metadata_fingerprint(
        (_chunk(),),
        settings,
        screening_run_id="screening-a",
        llm_base_url="https://llm.test/v1",
        llm_model="provider/model",
    )
    changed = build_document_metadata_fingerprint(
        (_chunk(),),
        settings,
        screening_run_id="screening-b",
        llm_base_url="https://llm.test/v1",
        llm_model="provider/model",
    )

    assert original.run_id != changed.run_id


def test_fact_extraction_fingerprint_changes_with_upstream_run() -> None:
    """Fact extraction cannot resume after either upstream artifact changes."""
    settings = SynthesisSettings.from_path(_SYNTHESIS_CONFIG_PATH)
    metadata = DocumentMetadataRecord("168/2024/NĐ-CP", date(2025, 1, 1), None)
    original = build_fact_extraction_fingerprint(
        (_chunk(),),
        ("chunk-1",),
        metadata,
        settings,
        screening_run_id="screening-a",
        metadata_run_id="metadata-a",
        llm_base_url="https://llm.test/v1",
        llm_model="provider/model",
    )
    changed = build_fact_extraction_fingerprint(
        (_chunk(),),
        ("chunk-1",),
        metadata,
        settings,
        screening_run_id="screening-b",
        metadata_run_id="metadata-a",
        llm_base_url="https://llm.test/v1",
        llm_model="provider/model",
    )

    assert original.run_id != changed.run_id


def test_fact_extraction_artifact_fingerprint_tracks_chunk_provenance() -> None:
    """Artifact identity records the prompt and model persisted for its chunks."""
    metadata = DocumentMetadataRecord("168/2024/NĐ-CP", date(2025, 1, 1), None)
    fingerprint = build_fact_extraction_artifact_fingerprint(
        (_chunk(),),
        ("chunk-1",),
        metadata,
        (_fact(),),
        (ExtractionProgressRecord("chunk-1", "5.5.0", "provider/model", "run-a"),),
    )

    assert fingerprint.prompt_version == "5.5.0"
    assert fingerprint.model == "provider/model"


def test_candidate_fingerprint_changes_with_persona_or_model() -> None:
    """Candidate progress becomes stale when prompt-facing persona text or the model changes."""
    settings = SynthesisSettings.from_path(_SYNTHESIS_CONFIG_PATH)
    original = build_candidate_generation_fingerprint(
        (_fact(),),
        (_scenario(),),
        {"tai_xe_ca_nhan": "Tài xế cá nhân"},
        settings,
        candidates_per_scenario=1,
        llm_base_url="https://llm.test/v1",
        llm_model="provider/model-a",
    )
    changed_persona = build_candidate_generation_fingerprint(
        (_fact(),),
        (_scenario(),),
        {"tai_xe_ca_nhan": "Người mới học lái"},
        settings,
        candidates_per_scenario=1,
        llm_base_url="https://llm.test/v1",
        llm_model="provider/model-a",
    )
    changed_model = build_candidate_generation_fingerprint(
        (_fact(),),
        (_scenario(),),
        {"tai_xe_ca_nhan": "Tài xế cá nhân"},
        settings,
        candidates_per_scenario=1,
        llm_base_url="https://llm.test/v1",
        llm_model="provider/model-b",
    )

    assert original.run_id != changed_persona.run_id
    assert original.run_id != changed_model.run_id


def test_initialize_stage_cache_writes_and_validates_a_sidecar(tmp_path: Path) -> None:
    """A fresh stage writes its fingerprint and accepts the same run on resume."""
    settings = SynthesisSettings.from_path(_SYNTHESIS_CONFIG_PATH)
    expected = build_chunk_screening_fingerprint(
        (_chunk(),), settings, llm_base_url="https://llm.test/v1", llm_model="provider/model-a"
    )
    fingerprint_path = tmp_path / "screening.run.json"
    artifact_path = tmp_path / "screening.jsonl"

    initialize_stage_cache(fingerprint_path, (artifact_path,), expected)
    initialize_stage_cache(fingerprint_path, (artifact_path,), expected)

    assert read_stage_cache_fingerprint(fingerprint_path) == expected


def test_initialize_stage_cache_rejects_legacy_artifacts_without_a_sidecar(tmp_path: Path) -> None:
    """An unversioned legacy cache is never stamped as if it came from the current run."""
    settings = SynthesisSettings.from_path(_SYNTHESIS_CONFIG_PATH)
    expected = build_chunk_screening_fingerprint(
        (_chunk(),), settings, llm_base_url="https://llm.test/v1", llm_model="provider/model-a"
    )
    artifact_path = tmp_path / "screening.jsonl"
    artifact_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing fingerprint"):
        initialize_stage_cache(tmp_path / "screening.run.json", (artifact_path,), expected)


def test_initialize_stage_cache_rejects_a_different_run(tmp_path: Path) -> None:
    """Existing artifacts cannot be resumed under a different source or model identity."""
    settings = SynthesisSettings.from_path(_SYNTHESIS_CONFIG_PATH)
    original = build_chunk_screening_fingerprint(
        (_chunk(),), settings, llm_base_url="https://llm.test/v1", llm_model="provider/model-a"
    )
    changed = build_chunk_screening_fingerprint(
        (_chunk(),), settings, llm_base_url="https://llm.test/v1", llm_model="provider/model-b"
    )
    fingerprint_path = tmp_path / "screening.run.json"
    artifact_path = tmp_path / "screening.jsonl"
    initialize_stage_cache(fingerprint_path, (artifact_path,), original)
    artifact_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="different synthesis run"):
        initialize_stage_cache(fingerprint_path, (artifact_path,), changed)
