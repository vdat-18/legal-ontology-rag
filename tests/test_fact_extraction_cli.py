"""Tests for the end-to-end fact-extraction command."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from legal_rag import fact_extraction_cli
from legal_rag.baml_client.types import ChunkScreeningResult
from legal_rag.baml_client.types import EvidenceEntry as BamlEvidenceEntry
from legal_rag.baml_client.types import LegalFact as BamlLegalFact
from legal_rag.benchmark_synthesis import chunk_screening, fact_extraction
from legal_rag.benchmark_synthesis.writer import read_fact_ledger, read_legal_facts
from legal_rag.chunker.models import Chunk
from legal_rag.chunker.writer import write_chunks


def _chunk(chunk_id: str, text: str) -> Chunk:
    """Build one final chunk as fact-extraction CLI input."""
    return Chunk(
        chunk_id=chunk_id,
        content_sha256="content-hash",
        chunker_config_sha256="config-hash",
        scope_id="scope-id",
        chunk_index=0,
        kind="paragraph_group",
        source_name="168_2024_ND-CP.docx",
        source_sha256="source-hash",
        record_scope="root",
        parent_attachment_id=None,
        attachment_member_path=None,
        attachment_member_sha256=None,
        unit_ids=("unit-0",),
        source_refs=("word/document.xml#/body/0",),
        sequence_start=0,
        sequence_end=0,
        text=text,
        context_text="",
        embedding_text=text,
        text_token_count=5,
        embedding_token_count=5,
        previous_chunk_id=None,
        next_chunk_id=None,
    )


class _AlwaysHasPenaltyContentClient:
    """Accept every chunk as having penalty content, ignoring settings."""

    def __init__(self, settings: object, evaluation_settings: object) -> None:
        return None

    def screen(self, chunk_text: str) -> ChunkScreeningResult:
        """Return a positive screening result for any chunk text."""
        return ChunkScreeningResult(has_penalty_content=True, reason="Contains penalty content.")


class _OneFactExtractionClient:
    """Return one valid fact citing the window's target chunk."""

    def __init__(self, settings: object, evaluation_settings: object) -> None:
        return None

    def extract(self, window: object) -> tuple[BamlLegalFact, ...]:
        """Return one fact citing the first chunk in the window."""
        target_chunk_id = next(iter(window)).chunk_id
        return (
            BamlLegalFact(
                conduct="Vượt đèn đỏ",
                subject="Người điều khiển ô tô",
                conditions=[],
                monetary_penalty="4.000.000 đồng đến 6.000.000 đồng",
                license_points_deducted="4 điểm",
                additional_penalties=[],
                remedial_measures=[],
                valid_from=None,
                valid_to=None,
                document_number="168/2024/NĐ-CP",
                provision_path="Điều 6 khoản 9 điểm b",
                evidence=[BamlEvidenceEntry(chunk_id=target_chunk_id, quoted_span="Phạt tiền từ 4.000.000 đồng")],
            ),
        )


class _FakeEncoderBackend:
    """Return one fixed unit vector per input text."""

    def encode(self, sentences: list[str], *, convert_to_numpy: bool, normalize_embeddings: bool) -> np.ndarray:
        """Return a fixed unit vector for every input sentence."""
        return np.array([[1.0, 0.0] for _ in sentences])


def _configure_cli(
    monkeypatch: pytest.MonkeyPatch,
    chunks_path: Path,
    facts_output_path: Path,
    ledger_path: Path,
) -> None:
    """Set deterministic CLI arguments, environment, and fake LLM/embedding clients."""
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:20128/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "provider/test-model")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("EMBEDDING_PROFILE", "qwen3_0_6b")
    monkeypatch.setenv("GROUNDTRUTH_SPAN_MATCH_THRESHOLD", "0.85")
    monkeypatch.setenv("GROUNDTRUTH_BUNDLE_OVERLAP_THRESHOLD", "0.5")
    monkeypatch.setenv("GROUNDTRUTH_DEDUP_COSINE_THRESHOLD", "0.9")
    monkeypatch.setenv("GROUNDTRUTH_MAX_QUALITY_RETRIES", "2")
    monkeypatch.setenv("GROUNDTRUTH_MAX_SIMULATOR_TURNS", "8")
    monkeypatch.setattr(chunk_screening, "BamlChunkScreeningClient", _AlwaysHasPenaltyContentClient)
    monkeypatch.setattr(fact_extraction, "BamlFactExtractionClient", _OneFactExtractionClient)
    from legal_rag.embedding.encoder import EmbeddingEncoder

    monkeypatch.setattr(
        EmbeddingEncoder, "from_profile", lambda profile: EmbeddingEncoder(profile, _FakeEncoderBackend())
    )
    monkeypatch.setattr(
        sys, "argv", ["legal-extract-facts", str(chunks_path), str(facts_output_path), str(ledger_path)]
    )


def test_fact_extraction_cli_extracts_facts_and_queues_them_for_review(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The command reads chunks, extracts one fact, and queues it for expert review."""
    chunks_path = tmp_path / "168_2024_ND-CP.chunks.jsonl"
    facts_path = tmp_path / "legal_facts.jsonl"
    ledger_path = tmp_path / "fact_ledger.jsonl"
    write_chunks(chunks_path, (_chunk("chunk-1", "Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng."),))
    _configure_cli(monkeypatch, chunks_path, facts_path, ledger_path)

    fact_extraction_cli.main()

    facts = read_legal_facts(facts_path)
    entries = read_fact_ledger(ledger_path)
    assert len(facts) == 1
    assert facts[0].conduct == "Vượt đèn đỏ"
    assert len(entries) == 1
    assert entries[0].status == "needs_expert_review"
    assert entries[0].fact_id == facts[0].fact_id


def test_fact_extraction_cli_second_run_preserves_approved_entry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Re-running extraction on unchanged chunks does not disturb an already-approved fact."""
    from datetime import UTC, datetime

    from legal_rag.benchmark_synthesis.fact_ledger import approve
    from legal_rag.benchmark_synthesis.writer import write_fact_ledger

    chunks_path = tmp_path / "168_2024_ND-CP.chunks.jsonl"
    facts_path = tmp_path / "legal_facts.jsonl"
    ledger_path = tmp_path / "fact_ledger.jsonl"
    write_chunks(chunks_path, (_chunk("chunk-1", "Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng."),))
    _configure_cli(monkeypatch, chunks_path, facts_path, ledger_path)
    fact_extraction_cli.main()

    entries = read_fact_ledger(ledger_path)
    reviewed_at = datetime(2026, 1, 1, tzinfo=UTC)
    write_fact_ledger(ledger_path, [approve(entries[0], reviewer="an.nguyen", reviewed_at=reviewed_at)])

    fact_extraction_cli.main()

    entries = read_fact_ledger(ledger_path)
    assert entries[0].status == "approved"
    assert entries[0].reviewer == "an.nguyen"
