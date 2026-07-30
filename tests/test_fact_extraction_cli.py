"""Tests for the end-to-end fact-extraction command."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

from legal_rag import fact_extraction_cli
from legal_rag.baml_client.types import EvidenceEntry as BamlEvidenceEntry
from legal_rag.baml_client.types import LegalFact as BamlLegalFact
from legal_rag.benchmark_synthesis import fact_extraction
from legal_rag.benchmark_synthesis.chunk_screening import ChunkScreeningRecord
from legal_rag.benchmark_synthesis.document_metadata import DocumentMetadataRecord
from legal_rag.benchmark_synthesis.models import EvidenceReference, LegalFact
from legal_rag.benchmark_synthesis.writer import (
    read_legal_facts,
    write_chunk_screening,
    write_document_metadata,
    write_legal_facts,
)
from legal_rag.chunker.models import Chunk
from legal_rag.chunker.writer import write_chunks


def _existing_fact(fact_id: str, source_chunk_id: str) -> LegalFact:
    """Build one already-extracted legal fact for resume tests."""
    return LegalFact(
        fact_id=fact_id,
        fact_content_sha256=f"content-{fact_id}",
        conduct="Vượt đèn đỏ",
        subject="Người điều khiển ô tô",
        conditions=(),
        monetary_penalty="4.000.000 đồng đến 6.000.000 đồng",
        license_points_deducted=None,
        additional_penalties=(),
        remedial_measures=(),
        valid_from=None,
        valid_to=None,
        document_number="168/2024/NĐ-CP",
        provision_path="Điều 6",
        evidence=(EvidenceReference(chunk_id=source_chunk_id, quoted_span="Phạt tiền"),),
        source_chunk_id=source_chunk_id,
    )


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
                provision_path="Điều 6 khoản 9 điểm b",
                evidence=[BamlEvidenceEntry(chunk_id=target_chunk_id, quoted_span="Phạt tiền từ 4.000.000 đồng")],
            ),
        )


def _configure_cli(
    monkeypatch: pytest.MonkeyPatch,
    chunks_path: Path,
    screening_path: Path,
    facts_output_path: Path,
    *,
    metadata: DocumentMetadataRecord | None = None,
) -> None:
    """Set deterministic CLI arguments, environment, document metadata, and fake LLM/embedding clients."""
    metadata_path = chunks_path.parent / "document_metadata.jsonl"
    if metadata is None:
        metadata = DocumentMetadataRecord(document_number="168/2024/NĐ-CP", valid_from=date(2025, 1, 1), valid_to=None)
    write_document_metadata(metadata_path, metadata)
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
        sys,
        "argv",
        [
            "legal-extract-facts",
            str(chunks_path),
            str(screening_path),
            str(metadata_path),
            str(facts_output_path),
        ],
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


def test_fact_extraction_cli_stamps_document_metadata_onto_every_fact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The resolved document number and effective dates land on facts the reading window could not supply them for."""
    chunks_path = tmp_path / "336_2025_ND-CP.chunks.jsonl"
    screening_path = tmp_path / "chunk_screening.jsonl"
    facts_path = tmp_path / "legal_facts.jsonl"
    write_chunks(chunks_path, (_chunk("chunk-1", "Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng."),))
    write_chunk_screening(
        screening_path,
        [ChunkScreeningRecord(chunk_id="chunk-1", has_penalty_content=True, reason="Contains penalty content.")],
    )
    _configure_cli(
        monkeypatch,
        chunks_path,
        screening_path,
        facts_path,
        metadata=DocumentMetadataRecord(document_number="336/2025/NĐ-CP", valid_from=date(2025, 8, 1), valid_to=None),
    )

    fact_extraction_cli.main()

    facts = read_legal_facts(facts_path)
    assert facts[0].document_number == "336/2025/NĐ-CP"
    assert facts[0].valid_from == date(2025, 8, 1)


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


def test_fact_extraction_cli_resumes_by_skipping_already_extracted_chunks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A chunk whose fact was already extracted is not sent to the LLM again."""
    chunks_path = tmp_path / "168_2024_ND-CP.chunks.jsonl"
    screening_path = tmp_path / "chunk_screening.jsonl"
    facts_path = tmp_path / "legal_facts.jsonl"
    write_chunks(
        chunks_path,
        (
            _chunk("chunk-1", "Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng."),
            _chunk("chunk-2", "Trừ 6 điểm giấy phép lái xe."),
        ),
    )
    write_chunk_screening(
        screening_path,
        [
            ChunkScreeningRecord(chunk_id="chunk-1", has_penalty_content=True, reason="Contains penalty content."),
            ChunkScreeningRecord(chunk_id="chunk-2", has_penalty_content=True, reason="Contains penalty content."),
        ],
    )
    write_legal_facts(facts_path, [_existing_fact("existing-fact", "chunk-1")])
    extracted_chunk_ids: list[str] = []

    class _ResumingClient:
        """Extract only the chunks still passed to it, recording their ids."""

        def __init__(self, settings: object, evaluation_settings: object) -> None:
            return None

        def extract(self, window: object) -> tuple[BamlLegalFact, ...]:
            """Record the target chunk id and return one fact citing it."""
            target_chunk_id = next(iter(window)).chunk_id
            extracted_chunk_ids.append(target_chunk_id)
            return (
                BamlLegalFact(
                    conduct="Trừ điểm giấy phép",
                    subject="Người điều khiển ô tô",
                    conditions=[],
                    monetary_penalty=None,
                    license_points_deducted="6 điểm",
                    additional_penalties=[],
                    remedial_measures=[],
                    valid_from=None,
                    valid_to=None,
                    provision_path="Điều 6 khoản 10",
                    evidence=[BamlEvidenceEntry(chunk_id=target_chunk_id, quoted_span="Trừ 6 điểm")],
                ),
            )

    _configure_cli(monkeypatch, chunks_path, screening_path, facts_path)
    monkeypatch.setattr(fact_extraction, "BamlFactExtractionClient", _ResumingClient)

    fact_extraction_cli.main()

    assert extracted_chunk_ids == ["chunk-2"]
    facts = read_legal_facts(facts_path)
    assert {fact.source_chunk_id for fact in facts} == {"chunk-1", "chunk-2"}


def test_fact_extraction_cli_skips_the_llm_entirely_when_fully_resumed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When every accepted chunk already has an extracted fact, the LLM client is never constructed."""
    chunks_path = tmp_path / "168_2024_ND-CP.chunks.jsonl"
    screening_path = tmp_path / "chunk_screening.jsonl"
    facts_path = tmp_path / "legal_facts.jsonl"
    write_chunks(chunks_path, (_chunk("chunk-1", "Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng."),))
    write_chunk_screening(
        screening_path,
        [ChunkScreeningRecord(chunk_id="chunk-1", has_penalty_content=True, reason="Contains penalty content.")],
    )
    write_legal_facts(facts_path, [_existing_fact("existing-fact", "chunk-1")])

    def _fail(*args: object, **kwargs: object) -> None:
        msg = "BamlFactExtractionClient must not be constructed when every chunk is resumed"
        raise AssertionError(msg)

    _configure_cli(monkeypatch, chunks_path, screening_path, facts_path)
    monkeypatch.setattr(fact_extraction, "BamlFactExtractionClient", _fail)

    fact_extraction_cli.main()

    facts = read_legal_facts(facts_path)
    assert len(facts) == 1
    assert facts[0].fact_id == "existing-fact"


def test_fact_extraction_cli_persists_progress_if_interrupted_mid_document(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A fact already extracted is durably written even if the process dies before the run finishes."""
    chunks_path = tmp_path / "168_2024_ND-CP.chunks.jsonl"
    screening_path = tmp_path / "chunk_screening.jsonl"
    facts_path = tmp_path / "legal_facts.jsonl"
    write_chunks(
        chunks_path,
        (
            _chunk("chunk-1", "Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng."),
            _chunk("chunk-2", "Trừ 6 điểm giấy phép lái xe."),
        ),
    )
    write_chunk_screening(
        screening_path,
        [
            ChunkScreeningRecord(chunk_id="chunk-1", has_penalty_content=True, reason="Contains penalty content."),
            ChunkScreeningRecord(chunk_id="chunk-2", has_penalty_content=True, reason="Contains penalty content."),
        ],
    )

    class _CrashesOnSecondChunkClient:
        """Succeed on the first target, then simulate the whole process being killed."""

        def __init__(self, settings: object, evaluation_settings: object) -> None:
            self._calls = 0

        def extract(self, window: object) -> tuple[BamlLegalFact, ...]:
            """Return one fact on the first call, then raise an uncaught fatal error."""
            self._calls += 1
            if self._calls > 1:
                raise SystemExit
            target_chunk_id = next(iter(window)).chunk_id
            return (
                BamlLegalFact(
                    conduct="Vượt đèn đỏ",
                    subject="Người điều khiển ô tô",
                    conditions=[],
                    monetary_penalty="4.000.000 đồng đến 6.000.000 đồng",
                    license_points_deducted=None,
                    additional_penalties=[],
                    remedial_measures=[],
                    valid_from=None,
                    valid_to=None,
                    provision_path="Điều 6",
                    evidence=[BamlEvidenceEntry(chunk_id=target_chunk_id, quoted_span="Phạt tiền")],
                ),
            )

    _configure_cli(monkeypatch, chunks_path, screening_path, facts_path)
    monkeypatch.setattr(fact_extraction, "BamlFactExtractionClient", _CrashesOnSecondChunkClient)

    with pytest.raises(SystemExit):
        fact_extraction_cli.main()

    facts = read_legal_facts(facts_path)
    assert len(facts) == 1
    assert facts[0].source_chunk_id == "chunk-1"


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
