"""Tests for synthesis-stage cache fingerprints."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from legal_rag.benchmark_synthesis.config import SynthesisSettings
from legal_rag.benchmark_synthesis.stage_cache import (
    build_chunk_screening_fingerprint,
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
