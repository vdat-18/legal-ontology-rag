"""Tests for the end-to-end chunk-screening command."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

from legal_rag import chunk_screening_cli
from legal_rag.baml_client.types import ChunkScreeningItem, ChunkScreeningResult
from legal_rag.benchmark_synthesis import chunk_screening
from legal_rag.benchmark_synthesis.chunk_screening import ChunkScreeningRecord
from legal_rag.benchmark_synthesis.writer import read_chunk_screening, write_chunk_screening
from legal_rag.chunker.models import Chunk
from legal_rag.chunker.writer import write_chunks

_SYNTHESIS_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "config" / "groundtruth" / "synthesis_v1.json"
)


def _chunk(chunk_id: str, text: str) -> Chunk:
    """Build one final chunk as chunk-screening CLI input."""
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


class _AlternatingScreeningClient:
    """Accept only the first chunk of every batch, ignoring settings."""

    def __init__(self, settings: object, synthesis_settings: object) -> None:
        return None

    def screen(self, chunks: Sequence[ChunkScreeningItem]) -> Sequence[ChunkScreeningResult]:
        """Return has_penalty_content=True only for the batch's first chunk."""
        return [
            ChunkScreeningResult(
                chunk_id=chunk.chunk_id,
                has_penalty_content=index == 0,
                reason="Contains penalty content." if index == 0 else "Table of contents.",
            )
            for index, chunk in enumerate(chunks)
        ]


def _configure_cli(monkeypatch: pytest.MonkeyPatch, chunks_path: Path, screening_output_path: Path) -> None:
    """Set deterministic CLI arguments, environment, and a fake LLM client."""
    from legal_rag.benchmark_synthesis import llm_execution

    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:20128/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "provider/test-model")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "30")
    monkeypatch.setattr(llm_execution.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(chunk_screening, "BamlChunkScreeningClient", _AlternatingScreeningClient)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "legal-screen-chunks",
            str(chunks_path),
            str(screening_output_path),
            "--synthesis-config",
            str(_SYNTHESIS_CONFIG_PATH),
        ],
    )


def test_chunk_screening_cli_writes_one_verdict_per_chunk(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The command reads chunks, screens every one, and writes a verdict record for each."""
    chunks_path = tmp_path / "168_2024_ND-CP.chunks.jsonl"
    screening_path = tmp_path / "chunk_screening.jsonl"
    write_chunks(
        chunks_path,
        (
            _chunk("chunk-1", "Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng."),
            _chunk("chunk-2", "MỤC LỤC"),
        ),
    )
    _configure_cli(monkeypatch, chunks_path, screening_path)

    chunk_screening_cli.main()

    records = read_chunk_screening(screening_path)
    assert [record.chunk_id for record in records] == ["chunk-1", "chunk-2"]
    assert records[0].has_penalty_content is True
    assert records[1].has_penalty_content is False
    assert records[1].reason == "Table of contents."


def test_chunk_screening_cli_resumes_by_skipping_already_screened_chunks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A chunk with a prior verdict is kept unchanged and is not screened again."""
    chunks_path = tmp_path / "168_2024_ND-CP.chunks.jsonl"
    screening_path = tmp_path / "chunk_screening.jsonl"
    write_chunks(
        chunks_path,
        (
            _chunk("chunk-1", "Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng."),
            _chunk("chunk-2", "MỤC LỤC"),
        ),
    )
    write_chunk_screening(
        screening_path,
        [ChunkScreeningRecord(chunk_id="chunk-1", has_penalty_content=True, reason="Đã sàng lọc trước đó.")],
    )
    screened_chunk_ids: list[str] = []

    class _ResumingClient:
        """Screen only the chunks still passed to it, recording their ids."""

        def __init__(self, settings: object, synthesis_settings: object) -> None:
            return None

        def screen(self, chunks: Sequence[ChunkScreeningItem]) -> Sequence[ChunkScreeningResult]:
            """Record screened chunk ids and return a fixed verdict for each."""
            screened_chunk_ids.extend(chunk.chunk_id for chunk in chunks)
            return [
                ChunkScreeningResult(chunk_id=chunk.chunk_id, has_penalty_content=False, reason="Table of contents.")
                for chunk in chunks
            ]

    _configure_cli(monkeypatch, chunks_path, screening_path)
    monkeypatch.setattr(chunk_screening, "BamlChunkScreeningClient", _ResumingClient)

    chunk_screening_cli.main()

    assert screened_chunk_ids == ["chunk-2"]
    records = read_chunk_screening(screening_path)
    assert [record.chunk_id for record in records] == ["chunk-1", "chunk-2"]
    assert records[0].reason == "Đã sàng lọc trước đó."
    assert records[1].has_penalty_content is False


def test_chunk_screening_cli_writes_partial_output_when_a_batch_keeps_failing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A failed batch remains resumable and makes the incomplete command fail."""
    chunks_path = tmp_path / "168_2024_ND-CP.chunks.jsonl"
    screening_path = tmp_path / "chunk_screening.jsonl"
    first_batch = tuple(_chunk(f"chunk-{i}", f"Phạt tiền {i}.") for i in range(50))
    failing_chunk = _chunk("chunk-fail", "Nội dung gây lỗi mạng.")
    write_chunks(chunks_path, (*first_batch, failing_chunk))

    class _FailsForFinalBatchClient:
        """Succeed on prior batches, then raise forever on the batch containing chunk-fail."""

        def __init__(self, settings: object, synthesis_settings: object) -> None:
            return None

        def screen(self, chunks: Sequence[ChunkScreeningItem]) -> Sequence[ChunkScreeningResult]:
            """Raise for the batch containing chunk-fail, otherwise return accepted verdicts."""
            if any(chunk.chunk_id == "chunk-fail" for chunk in chunks):
                msg = "provider timed out"
                raise TimeoutError(msg)
            return [
                ChunkScreeningResult(chunk_id=chunk.chunk_id, has_penalty_content=True, reason="Contains penalty.")
                for chunk in chunks
            ]

    _configure_cli(monkeypatch, chunks_path, screening_path)
    monkeypatch.setattr(chunk_screening, "BamlChunkScreeningClient", _FailsForFinalBatchClient)

    with pytest.raises(RuntimeError, match="chunk screening left 1 incomplete"):
        chunk_screening_cli.main()

    records = read_chunk_screening(screening_path)
    assert [record.chunk_id for record in records] == [chunk.chunk_id for chunk in first_batch]
    assert "chunk-fail" not in {record.chunk_id for record in records}


def test_chunk_screening_cli_persists_progress_if_interrupted_mid_document(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A batch already screened is durably written even if the process dies before the run finishes."""
    chunks_path = tmp_path / "168_2024_ND-CP.chunks.jsonl"
    screening_path = tmp_path / "chunk_screening.jsonl"
    first_batch = tuple(_chunk(f"chunk-{i}", f"Phạt tiền {i}.") for i in range(50))
    second_batch = tuple(_chunk(f"chunk-b-{i}", f"Phạt tiền b{i}.") for i in range(50))
    write_chunks(chunks_path, (*first_batch, *second_batch))

    class _CrashesOnSecondBatchClient:
        """Succeed on the first batch, then simulate the whole process being killed."""

        def __init__(self, settings: object, synthesis_settings: object) -> None:
            self._calls = 0

        def screen(self, chunks: Sequence[ChunkScreeningItem]) -> Sequence[ChunkScreeningResult]:
            """Return accepted verdicts on the first call, then raise an uncaught fatal error."""
            self._calls += 1
            if self._calls > 1:
                raise SystemExit
            return [
                ChunkScreeningResult(chunk_id=chunk.chunk_id, has_penalty_content=True, reason="Contains penalty.")
                for chunk in chunks
            ]

    _configure_cli(monkeypatch, chunks_path, screening_path)
    monkeypatch.setattr(chunk_screening, "BamlChunkScreeningClient", _CrashesOnSecondBatchClient)

    with pytest.raises(SystemExit):
        chunk_screening_cli.main()

    records = read_chunk_screening(screening_path)
    assert [record.chunk_id for record in records] == [chunk.chunk_id for chunk in first_batch]


def test_chunk_screening_cli_skips_the_llm_entirely_when_fully_resumed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When every chunk already has a verdict, the LLM client is never constructed."""
    chunks_path = tmp_path / "168_2024_ND-CP.chunks.jsonl"
    screening_path = tmp_path / "chunk_screening.jsonl"
    write_chunks(chunks_path, (_chunk("chunk-1", "Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng."),))
    write_chunk_screening(
        screening_path,
        [ChunkScreeningRecord(chunk_id="chunk-1", has_penalty_content=True, reason="Đã sàng lọc trước đó.")],
    )

    def _fail(*args: object, **kwargs: object) -> None:
        msg = "BamlChunkScreeningClient must not be constructed when every chunk is resumed"
        raise AssertionError(msg)

    _configure_cli(monkeypatch, chunks_path, screening_path)
    monkeypatch.setattr(chunk_screening, "BamlChunkScreeningClient", _fail)

    chunk_screening_cli.main()

    records = read_chunk_screening(screening_path)
    assert [record.chunk_id for record in records] == ["chunk-1"]
    assert records[0].reason == "Đã sàng lọc trước đó."
