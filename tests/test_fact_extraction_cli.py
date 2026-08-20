"""Tests for the end-to-end fact-extraction command."""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pytest

from legal_rag import fact_extraction_cli
from legal_rag.baml_client.types import EvidenceEntry as BamlEvidenceEntry
from legal_rag.baml_client.types import FactExtractionBatchItem, FactExtractionBatchResult
from legal_rag.baml_client.types import LegalFact as BamlLegalFact
from legal_rag.benchmark_synthesis import fact_extraction
from legal_rag.benchmark_synthesis.chunk_screening import ChunkScreeningRecord
from legal_rag.benchmark_synthesis.config import SynthesisSettings
from legal_rag.benchmark_synthesis.document_metadata import DocumentMetadataRecord
from legal_rag.benchmark_synthesis.fact_extraction import ExtractionProgressRecord
from legal_rag.benchmark_synthesis.models import EvidenceReference, LegalFact
from legal_rag.benchmark_synthesis.stage_cache import (
    build_chunk_screening_fingerprint,
    build_document_metadata_fingerprint,
    build_fact_extraction_fingerprint,
    stage_cache_path,
    write_stage_cache_fingerprint,
)
from legal_rag.benchmark_synthesis.writer import (
    read_chunk_screening,
    read_extraction_progress,
    read_legal_facts,
    write_chunk_screening,
    write_document_metadata,
    write_extraction_progress,
    write_legal_facts,
)
from legal_rag.chunker.models import Chunk
from legal_rag.chunker.writer import read_chunks, write_chunks

_SYNTHESIS_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "config" / "groundtruth" / "synthesis_v1.json"
)


def _existing_fact(fact_id: str, source_chunk_id: str) -> LegalFact:
    """Build one already-extracted legal fact for resume tests."""
    return LegalFact(
        fact_id=fact_id,
        statement="Người điều khiển ô tô vượt đèn đỏ bị phạt tiền.",
        valid_from=None,
        valid_to=None,
        document_number="168/2024/NĐ-CP",
        evidence=(EvidenceReference(chunk_id=source_chunk_id),),
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

    def __init__(self, settings: object, synthesis_settings: object) -> None:
        return None

    def extract_batch(self, items: list[FactExtractionBatchItem]) -> tuple[FactExtractionBatchResult, ...]:
        """Return one fact for every target in the batch."""
        return tuple(
            FactExtractionBatchResult(
                target_chunk_id=item.target_chunk_id,
                facts=[
                    BamlLegalFact(
                        statement="Người điều khiển ô tô vượt đèn đỏ bị phạt tiền và trừ điểm.",
                        evidence=[BamlEvidenceEntry(chunk_id=item.target_chunk_id)],
                    )
                ],
            )
            for item in items
        )


def _configure_cli(
    monkeypatch: pytest.MonkeyPatch,
    chunks_path: Path,
    screening_path: Path,
    facts_output_path: Path,
    *,
    metadata: DocumentMetadataRecord | None = None,
    synthesis_config_path: Path = _SYNTHESIS_CONFIG_PATH,
) -> None:
    """Set deterministic CLI arguments, environment, document metadata, and fake LLM/embedding clients."""
    from legal_rag.benchmark_synthesis import llm_execution

    metadata_path = chunks_path.parent / "document_metadata.jsonl"
    progress_path = facts_output_path.parent / "extraction_progress.jsonl"
    if metadata is None:
        metadata = DocumentMetadataRecord(document_number="168/2024/NĐ-CP", valid_from=date(2025, 1, 1), valid_to=None)
    write_document_metadata(metadata_path, metadata)
    chunks = read_chunks(chunks_path)
    synthesis_settings = SynthesisSettings.from_path(synthesis_config_path)
    screening_fingerprint = build_chunk_screening_fingerprint(
        chunks,
        synthesis_settings,
        llm_base_url="http://localhost:20128/v1",
        llm_model="provider/test-model",
    )
    metadata_fingerprint = build_document_metadata_fingerprint(
        chunks,
        synthesis_settings,
        llm_base_url="http://localhost:20128/v1",
        llm_model="provider/test-model",
    )
    write_stage_cache_fingerprint(stage_cache_path(screening_path), screening_fingerprint)
    write_stage_cache_fingerprint(stage_cache_path(metadata_path), metadata_fingerprint)
    if facts_output_path.exists() or progress_path.exists():
        accepted_chunk_ids = {
            record.chunk_id for record in read_chunk_screening(screening_path) if record.has_penalty_content
        }
        accepted_chunks = tuple(chunk for chunk in chunks if chunk.chunk_id in accepted_chunk_ids)
        extraction_fingerprint = build_fact_extraction_fingerprint(
            chunks,
            tuple(chunk.chunk_id for chunk in accepted_chunks),
            metadata,
            synthesis_settings,
            screening_run_id=screening_fingerprint.run_id,
            metadata_run_id=metadata_fingerprint.run_id,
            llm_base_url="http://localhost:20128/v1",
            llm_model="provider/test-model",
        )
        write_stage_cache_fingerprint(stage_cache_path(facts_output_path), extraction_fingerprint)
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:20128/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "provider/test-model")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "30")
    monkeypatch.setattr(llm_execution.time, "sleep", lambda _seconds: None)
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
            str(progress_path),
            "--synthesis-config",
            str(synthesis_config_path),
        ],
    )


def _serial_fact_extraction_config(tmp_path: Path) -> Path:
    """Write a test config that makes interruption ordering deterministic."""
    payload = json.loads(_SYNTHESIS_CONFIG_PATH.read_text(encoding="utf-8"))
    payload["stages"]["fact_extraction"]["max_concurrency"] = 1
    output_path = tmp_path / "synthesis.serial-fact-extraction.json"
    output_path.write_text(json.dumps(payload), encoding="utf-8")
    return output_path


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
    assert facts[0].statement == "Người điều khiển ô tô vượt đèn đỏ bị phạt tiền và trừ điểm."
    assert facts[0].evidence == (EvidenceReference(chunk_id="chunk-1"),)
    assert stage_cache_path(facts_path).exists()


def test_fact_extraction_cli_persists_distinct_content_addressed_facts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Different statements and evidence receive distinct identities and are both persisted."""
    chunks_path = tmp_path / "168_2024_ND-CP.chunks.jsonl"
    screening_path = tmp_path / "chunk_screening.jsonl"
    facts_path = tmp_path / "legal_facts.jsonl"
    chunks = (
        _chunk("chunk-1", "Phạt tiền đối với hành vi vượt đèn đỏ."),
        _chunk("chunk-2", "Phạt tiền đối với hành vi vượt đèn đỏ."),
    )
    write_chunks(chunks_path, chunks)
    write_chunk_screening(
        screening_path,
        [
            ChunkScreeningRecord(chunk_id=chunk.chunk_id, has_penalty_content=True, reason="Penalty.")
            for chunk in chunks
        ],
    )

    class _ConflictingClient:
        def __init__(self, settings: object, synthesis_settings: object) -> None:
            return None

        def extract_batch(self, items: list[FactExtractionBatchItem]) -> tuple[FactExtractionBatchResult, ...]:
            return tuple(
                FactExtractionBatchResult(
                    target_chunk_id=item.target_chunk_id,
                    facts=[
                        BamlLegalFact(
                            statement=f"Người điều khiển ô tô vượt đèn đỏ chịu mức phạt {item.target_chunk_id}.",
                            evidence=[BamlEvidenceEntry(chunk_id=item.target_chunk_id)],
                        )
                    ],
                )
                for item in items
            )

    _configure_cli(monkeypatch, chunks_path, screening_path, facts_path)
    monkeypatch.setattr(fact_extraction, "BamlFactExtractionClient", _ConflictingClient)

    fact_extraction_cli.main()

    facts = read_legal_facts(facts_path)
    assert len(facts) == 2
    assert len({fact.fact_id for fact in facts}) == 2
    assert {fact.source_chunk_id for fact in facts} == {"chunk-1", "chunk-2"}


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
    """A chunk excluded by screening is still resolved as preceding context for a later target chunk."""
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

        def __init__(self, settings: object, synthesis_settings: object) -> None:
            return None

        def extract_batch(self, items: list[FactExtractionBatchItem]) -> tuple[FactExtractionBatchResult, ...]:
            """Record every window's chunk ids."""
            for item in items:
                seen_chunk_ids.extend(entry.chunk_id for entry in item.window)
            return tuple(FactExtractionBatchResult(target_chunk_id=item.target_chunk_id, facts=[]) for item in items)

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
    write_extraction_progress(tmp_path / "extraction_progress.jsonl", [ExtractionProgressRecord(chunk_id="chunk-1")])
    extracted_chunk_ids: list[str] = []

    class _ResumingClient:
        """Extract only the chunks still passed to it, recording their ids."""

        def __init__(self, settings: object, synthesis_settings: object) -> None:
            return None

        def extract_batch(self, items: list[FactExtractionBatchItem]) -> tuple[FactExtractionBatchResult, ...]:
            """Record target ids and return one fact for each."""
            extracted_chunk_ids.extend(item.target_chunk_id for item in items)
            return tuple(
                FactExtractionBatchResult(
                    target_chunk_id=item.target_chunk_id,
                    facts=[
                        BamlLegalFact(
                            statement="Người điều khiển ô tô thực hiện hành vi bị trừ điểm giấy phép.",
                            evidence=[BamlEvidenceEntry(chunk_id=item.target_chunk_id)],
                        )
                    ],
                )
                for item in items
            )

    _configure_cli(monkeypatch, chunks_path, screening_path, facts_path)
    monkeypatch.setattr(fact_extraction, "BamlFactExtractionClient", _ResumingClient)

    fact_extraction_cli.main()

    assert extracted_chunk_ids == ["chunk-2"]
    facts = read_legal_facts(facts_path)
    assert {fact.source_chunk_id for fact in facts} == {"chunk-1", "chunk-2"}


def test_fact_extraction_cli_does_not_reprocess_a_chunk_that_yielded_no_facts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A chunk the model returned nothing for is recorded as processed, so a second run does not pay for it again."""
    chunks_path = tmp_path / "168_2024_ND-CP.chunks.jsonl"
    screening_path = tmp_path / "chunk_screening.jsonl"
    facts_path = tmp_path / "legal_facts.jsonl"
    write_chunks(chunks_path, (_chunk("chunk-1", "QUYẾT ĐỊNH Kéo dài thời hạn tạm giữ tang vật."),))
    write_chunk_screening(
        screening_path,
        [ChunkScreeningRecord(chunk_id="chunk-1", has_penalty_content=True, reason="Contains penalty content.")],
    )
    extraction_calls: list[str] = []

    class _NoFactClient:
        """Return no facts at all, recording every target it was asked about."""

        def __init__(self, settings: object, synthesis_settings: object) -> None:
            return None

        def extract_batch(self, items: list[FactExtractionBatchItem]) -> tuple[FactExtractionBatchResult, ...]:
            """Record target ids and return empty facts for each."""
            extraction_calls.extend(item.target_chunk_id for item in items)
            return tuple(FactExtractionBatchResult(target_chunk_id=item.target_chunk_id, facts=[]) for item in items)

    _configure_cli(monkeypatch, chunks_path, screening_path, facts_path)
    monkeypatch.setattr(fact_extraction, "BamlFactExtractionClient", _NoFactClient)

    fact_extraction_cli.main()
    fact_extraction_cli.main()

    assert extraction_calls == ["chunk-1"]
    assert read_extraction_progress(tmp_path / "extraction_progress.jsonl") == (
        ExtractionProgressRecord(chunk_id="chunk-1"),
    )
    assert read_legal_facts(facts_path) == ()


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
    write_extraction_progress(tmp_path / "extraction_progress.jsonl", [ExtractionProgressRecord(chunk_id="chunk-1")])

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
    """A successful batch is durable if the process dies while starting the next batch."""
    chunks_path = tmp_path / "168_2024_ND-CP.chunks.jsonl"
    screening_path = tmp_path / "chunk_screening.jsonl"
    facts_path = tmp_path / "legal_facts.jsonl"
    chunks = tuple(_chunk(f"chunk-{index}", f"Phạt tiền cho hành vi {index}.") for index in range(1, 7))
    write_chunks(chunks_path, chunks)
    write_chunk_screening(
        screening_path,
        [
            ChunkScreeningRecord(chunk_id=chunk.chunk_id, has_penalty_content=True, reason="Penalty.")
            for chunk in chunks
        ],
    )

    class _CrashesOnSecondBatchClient:
        """Succeed on the first batch, then simulate the whole process being killed."""

        def __init__(self, settings: object, synthesis_settings: object) -> None:
            self._calls = 0

        def extract_batch(self, items: list[FactExtractionBatchItem]) -> tuple[FactExtractionBatchResult, ...]:
            """Return facts for the first batch, then raise an uncaught fatal error."""
            self._calls += 1
            if self._calls > 1:
                raise SystemExit
            return tuple(
                FactExtractionBatchResult(
                    target_chunk_id=item.target_chunk_id,
                    facts=[
                        BamlLegalFact(
                            statement=f"Người điều khiển thực hiện hành vi {item.target_chunk_id} bị phạt tiền.",
                            evidence=[BamlEvidenceEntry(chunk_id=item.target_chunk_id)],
                        )
                    ],
                )
                for item in items
            )

    _configure_cli(
        monkeypatch,
        chunks_path,
        screening_path,
        facts_path,
        synthesis_config_path=_serial_fact_extraction_config(tmp_path),
    )
    monkeypatch.setattr(fact_extraction, "BamlFactExtractionClient", _CrashesOnSecondBatchClient)

    with pytest.raises(SystemExit):
        fact_extraction_cli.main()

    facts = read_legal_facts(facts_path)
    assert len(facts) == 1
    assert facts[0].source_chunk_id == "chunk-1"
    assert read_extraction_progress(tmp_path / "extraction_progress.jsonl") == (
        ExtractionProgressRecord(chunk_id="chunk-1"),
    )


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
