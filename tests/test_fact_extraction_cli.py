"""Tests for the end-to-end fact-extraction command."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from legal_rag import fact_extraction_cli
from legal_rag.baml_client.types import EvidenceEntry as BamlEvidenceEntry
from legal_rag.baml_client.types import LegalFact as BamlLegalFact
from legal_rag.benchmark_synthesis import fact_extraction
from legal_rag.benchmark_synthesis.chunk_screening import ChunkScreeningRecord
from legal_rag.benchmark_synthesis.writer import read_legal_facts, write_chunk_screening
from legal_rag.chunker.models import Chunk
from legal_rag.chunker.writer import write_chunks


def _chunk(chunk_id: str, text: str, *, previous_chunk_id: str | None = None) -> Chunk:
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
        previous_chunk_id=previous_chunk_id,
        next_chunk_id=None,
    )


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


def _configure_cli(
    monkeypatch: pytest.MonkeyPatch,
    chunks_path: Path,
    screening_path: Path,
    facts_output_path: Path,
) -> None:
    """Set deterministic CLI arguments, environment, and fake LLM/embedding clients."""
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:20128/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "provider/test-model")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("EMBEDDING_PROFILE", "qwen3_0_6b")
    monkeypatch.setenv("GROUNDTRUTH_SPAN_MATCH_THRESHOLD", "0.85")
    monkeypatch.setenv("GROUNDTRUTH_BUNDLE_OVERLAP_THRESHOLD", "0.5")
    monkeypatch.setenv("GROUNDTRUTH_MAX_QUALITY_RETRIES", "2")
    monkeypatch.setattr(fact_extraction, "BamlFactExtractionClient", _OneFactExtractionClient)
    monkeypatch.setattr(
        sys, "argv", ["legal-extract-facts", str(chunks_path), str(screening_path), str(facts_output_path)]
    )


def test_fact_extraction_cli_extracts_and_writes_facts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The command reads accepted chunks from the screening file, extracts one fact, and writes it directly."""
    chunks_path = tmp_path / "168_2024_ND-CP.chunks.jsonl"
    screening_path = tmp_path / "chunk_screening.jsonl"
    facts_path = tmp_path / "legal_facts.jsonl"
    write_chunks(chunks_path, (_chunk("chunk-1", "Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng."),))
    write_chunk_screening(
        screening_path,
        [ChunkScreeningRecord(chunk_id="chunk-1", has_penalty_content=True, reason="Contains penalty content.")],
    )
    _configure_cli(monkeypatch, chunks_path, screening_path, facts_path)

    fact_extraction_cli.main()

    facts = read_legal_facts(facts_path)
    assert len(facts) == 1
    assert facts[0].conduct == "Vượt đèn đỏ"


def test_fact_extraction_cli_uses_excluded_chunk_as_adjacent_context(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A chunk excluded by screening is still resolved as Adjacent context for a later target chunk."""
    chunks_path = tmp_path / "168_2024_ND-CP.chunks.jsonl"
    screening_path = tmp_path / "chunk_screening.jsonl"
    facts_path = tmp_path / "legal_facts.jsonl"
    write_chunks(
        chunks_path,
        (
            _chunk("chunk-1", "Trong Nghị định này, xe cơ giới được hiểu là ô tô, mô tô."),
            _chunk(
                "chunk-2",
                "Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng đối với xe cơ giới.",
                previous_chunk_id="chunk-1",
            ),
        ),
    )
    write_chunk_screening(
        screening_path,
        [
            ChunkScreeningRecord(chunk_id="chunk-1", has_penalty_content=False, reason="Definitional, no penalty."),
            ChunkScreeningRecord(chunk_id="chunk-2", has_penalty_content=True, reason="Contains penalty content."),
        ],
    )
    seen_chunk_ids: list[str] = []

    class _RecordingClient:
        """Record every chunk_id seen in the window and return no facts."""

        def __init__(self, settings: object, evaluation_settings: object) -> None:
            return None

        def extract(self, window: object) -> tuple[BamlLegalFact, ...]:
            """Record the window's chunk ids."""
            seen_chunk_ids.extend(entry.chunk_id for entry in window)
            return ()

    _configure_cli(monkeypatch, chunks_path, screening_path, facts_path)
    monkeypatch.setattr(fact_extraction, "BamlFactExtractionClient", _RecordingClient)

    fact_extraction_cli.main()

    assert set(seen_chunk_ids) == {"chunk-1", "chunk-2"}


def test_fact_extraction_cli_skips_chunks_excluded_by_screening(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A chunk marked has_penalty_content=False in the screening file never reaches extraction."""
    chunks_path = tmp_path / "168_2024_ND-CP.chunks.jsonl"
    screening_path = tmp_path / "chunk_screening.jsonl"
    facts_path = tmp_path / "legal_facts.jsonl"
    write_chunks(
        chunks_path,
        (
            _chunk("chunk-1", "Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng."),
            _chunk("chunk-2", "MỤC LỤC"),
        ),
    )
    write_chunk_screening(
        screening_path,
        [
            ChunkScreeningRecord(chunk_id="chunk-1", has_penalty_content=True, reason="Contains penalty content."),
            ChunkScreeningRecord(chunk_id="chunk-2", has_penalty_content=False, reason="Table of contents."),
        ],
    )
    _configure_cli(monkeypatch, chunks_path, screening_path, facts_path)

    fact_extraction_cli.main()

    facts = read_legal_facts(facts_path)
    assert len(facts) == 1
