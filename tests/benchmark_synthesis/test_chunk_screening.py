"""Tests for the chunk-screening gate."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest

from legal_rag.baml_client.types import ChunkScreeningItem, ChunkScreeningResult
from legal_rag.benchmark_synthesis import chunk_screening
from legal_rag.benchmark_synthesis.chunk_screening import BamlChunkScreeningClient, screen_chunks
from legal_rag.benchmark_synthesis.config import SynthesisSettings
from legal_rag.chunker.models import Chunk, ChunkKind, RecordScope
from legal_rag.settings import Settings

_SYNTHESIS_CONFIG_PATH = Path(__file__).resolve().parents[2] / "data" / "config" / "groundtruth" / "synthesis_v1.json"
_SCREENING_PROMPT_PATH = Path(__file__).resolve().parents[2] / "baml_src" / "chunk_screening.baml"


def _chunk(
    chunk_id: str,
    text: str,
    *,
    context_text: str = "",
    source_name: str = "source.docx",
    record_scope: RecordScope = "root",
    attachment_member_path: str | None = None,
    kind: ChunkKind = "paragraph_group",
) -> Chunk:
    """Build one minimal chunk for chunk-screening tests."""
    return Chunk(
        chunk_id=chunk_id,
        content_sha256="content-hash",
        chunker_config_sha256="config-hash",
        scope_id="scope-id",
        chunk_index=0,
        kind=kind,
        source_name=source_name,
        source_sha256="source-hash",
        record_scope=record_scope,
        parent_attachment_id=None,
        attachment_member_path=attachment_member_path,
        attachment_member_sha256=None,
        unit_ids=("unit-0",),
        source_refs=("word/document.xml#/body/0",),
        sequence_start=0,
        sequence_end=0,
        text=text,
        context_text=context_text,
        embedding_text=text,
        text_token_count=5,
        embedding_token_count=5,
        previous_chunk_id=None,
        next_chunk_id=None,
    )


class _FakeScreeningClient:
    """Return pre-programmed results keyed by chunk_id, ignoring batch grouping."""

    def __init__(self, results_by_chunk_id: dict[str, ChunkScreeningResult]) -> None:
        self._results_by_chunk_id = results_by_chunk_id

    def screen(self, chunks: Sequence[ChunkScreeningItem]) -> Sequence[ChunkScreeningResult]:
        """Return the configured result for each chunk_id in this batch."""
        return [self._results_by_chunk_id[chunk.chunk_id] for chunk in chunks]


def test_screen_chunks_returns_one_record_per_chunk_in_order() -> None:
    """Every chunk gets a verdict record, in input order, with the model's reason."""
    high = _chunk("chunk-high", "Điều 6 khoản 9 điểm b quy định mức phạt tiền.")
    low = _chunk("chunk-low", "Mục lục")
    client = _FakeScreeningClient(
        {
            "chunk-high": ChunkScreeningResult(
                chunk_id="chunk-high",
                has_penalty_content=True,
                has_document_metadata=True,
                reason="Nêu rõ mức phạt tiền cụ thể.",
            ),
            "chunk-low": ChunkScreeningResult(
                chunk_id="chunk-low",
                has_penalty_content=False,
                has_document_metadata=False,
                reason="Đây là mục lục, không có chế tài.",
            ),
        }
    )

    records = screen_chunks([high, low], client, batch_size=50, max_retries=2, retry_delay_seconds=0)

    assert [record.chunk_id for record in records] == ["chunk-high", "chunk-low"]
    assert records[0].has_penalty_content is True
    assert records[0].has_document_metadata is True
    assert records[1].has_penalty_content is False
    assert records[1].reason == "Đây là mục lục, không có chế tài."


def test_screen_chunks_passes_structural_context_separately_from_target_text() -> None:
    """Screening receives parser-derived structure without merging it into source text."""
    context_text = "[LEGAL_STRUCTURE 1.0.0]\nĐiều 6. Xử phạt người điều khiển ô tô\n[/LEGAL_STRUCTURE]"
    chunk = _chunk("chunk-1", "b) Không chấp hành hiệu lệnh.", context_text=context_text)
    seen_items: list[ChunkScreeningItem] = []

    class _RecordingClient:
        def screen(self, items: Sequence[ChunkScreeningItem]) -> Sequence[ChunkScreeningResult]:
            seen_items.extend(items)
            return [
                ChunkScreeningResult(
                    chunk_id="chunk-1", has_penalty_content=True, has_document_metadata=False, reason="Penalty item."
                )
            ]

    screen_chunks([chunk], _RecordingClient(), batch_size=50, max_retries=0, retry_delay_seconds=0)

    assert seen_items[0].context_text == context_text
    assert seen_items[0].text == chunk.text


def test_screen_chunks_passes_previous_chunk_for_split_penalty_list() -> None:
    """A selective target still receives its adjacent controlling penalty clause."""
    previous = _chunk("chunk-previous", "3. Phạt tiền đối với các hành vi sau đây:")
    target = replace(
        _chunk("chunk-target", "b) Cơ sở đào tạo sử dụng xe tập lái không có giấy phép."),
        previous_chunk_id=previous.chunk_id,
    )
    seen_items: list[ChunkScreeningItem] = []

    class _RecordingClient:
        def screen(self, items: Sequence[ChunkScreeningItem]) -> Sequence[ChunkScreeningResult]:
            seen_items.extend(items)
            return [
                ChunkScreeningResult(
                    chunk_id="chunk-target",
                    has_penalty_content=True,
                    has_document_metadata=False,
                    reason="Target tiếp nối danh sách hành vi bị phạt.",
                )
            ]

    screen_chunks(
        [target],
        _RecordingClient(),
        context_chunks=[previous, target],
        batch_size=1,
        max_retries=0,
        retry_delay_seconds=0,
    )

    assert seen_items[0].previous_chunk_text == previous.text


def test_screening_prompt_limits_previous_text_to_the_controlling_clause() -> None:
    """Adjacent context may restore a split list without becoming extractable evidence."""
    prompt = " ".join(_SCREENING_PROMPT_PATH.read_text(encoding="utf-8").split())

    assert "Target bắt đầu giữa một danh sách Điểm/Khoản" in prompt
    assert "Previous text chỉ cấp phạm vi chi phối" in prompt
    assert "conduct và evidence vẫn phải nằm trong Target" in prompt


def test_screen_chunks_passes_source_provenance_separately_from_target_text() -> None:
    """Screening receives parser provenance so the model can identify attachment boilerplate."""
    chunk = _chunk(
        "chunk-1",
        "Ghi chú: ghi rõ họ tên người lập.",
        source_name="Bieu mau.doc",
        record_scope="attachment_child",
        attachment_member_path="word/embeddings/Bieu mau.doc",
        kind="paragraph_group",
    )
    seen_items: list[ChunkScreeningItem] = []

    class _RecordingClient:
        def screen(self, items: Sequence[ChunkScreeningItem]) -> Sequence[ChunkScreeningResult]:
            seen_items.extend(items)
            return [
                ChunkScreeningResult(
                    chunk_id="chunk-1",
                    has_penalty_content=False,
                    has_document_metadata=False,
                    reason="Đây là ghi chú của biểu mẫu đính kèm.",
                )
            ]

    screen_chunks([chunk], _RecordingClient(), batch_size=10, max_retries=0, retry_delay_seconds=0)

    assert seen_items[0].source_name == "Bieu mau.doc"
    assert seen_items[0].record_scope == "attachment_child"
    assert seen_items[0].attachment_member_path == "word/embeddings/Bieu mau.doc"
    assert seen_items[0].kind == "paragraph_group"


def test_screen_chunks_batches_calls_by_batch_size() -> None:
    """More chunks than batch_size are screened across multiple calls, preserving order."""
    chunks = [_chunk(f"chunk-{i}", f"Nội dung {i}") for i in range(5)]
    calls: list[int] = []
    results_by_chunk_id = {
        chunk.chunk_id: ChunkScreeningResult(
            chunk_id=chunk.chunk_id, has_penalty_content=True, has_document_metadata=False, reason=""
        )
        for chunk in chunks
    }

    class _CountingClient:
        def screen(self, batch: Sequence[ChunkScreeningItem]) -> Sequence[ChunkScreeningResult]:
            calls.append(len(batch))
            return [results_by_chunk_id[item.chunk_id] for item in batch]

    records = screen_chunks(chunks, _CountingClient(), batch_size=2, max_retries=2, retry_delay_seconds=0)

    assert calls == [2, 2, 1]
    assert [record.chunk_id for record in records] == [chunk.chunk_id for chunk in chunks]
    assert all(record.has_penalty_content for record in records)


def test_screen_chunks_supports_configured_batches_of_ten() -> None:
    """The configured production size sends at most ten chunks in one screening request."""
    chunks = [_chunk(f"chunk-{i}", f"Nội dung {i}") for i in range(11)]
    calls: list[int] = []

    class _CountingClient:
        def screen(self, batch: Sequence[ChunkScreeningItem]) -> Sequence[ChunkScreeningResult]:
            calls.append(len(batch))
            return [
                ChunkScreeningResult(
                    chunk_id=item.chunk_id, has_penalty_content=True, has_document_metadata=False, reason=""
                )
                for item in batch
            ]

    screen_chunks(chunks, _CountingClient(), batch_size=10, max_retries=2, retry_delay_seconds=0)

    assert calls == [10, 1]


def test_screen_chunks_runs_batches_concurrently_and_preserves_output_order() -> None:
    """Configured screening concurrency allows batches to overlap while final records stay ordered."""
    chunks = [_chunk(f"chunk-{index}", f"Nội dung {index}.") for index in range(2)]
    barrier = threading.Barrier(2)
    calls: list[str] = []

    class _ConcurrentClient:
        def screen(self, batch: Sequence[ChunkScreeningItem]) -> Sequence[ChunkScreeningResult]:
            calls.append(batch[0].chunk_id)
            barrier.wait(timeout=2)
            return [
                ChunkScreeningResult(
                    chunk_id=item.chunk_id,
                    has_penalty_content=True,
                    has_document_metadata=False,
                    reason="Contains penalty.",
                )
                for item in batch
            ]

    records = screen_chunks(
        chunks,
        _ConcurrentClient(),
        batch_size=1,
        max_concurrency=2,
        max_retries=0,
        retry_delay_seconds=0,
    )

    assert set(calls) == {"chunk-0", "chunk-1"}
    assert [record.chunk_id for record in records] == ["chunk-0", "chunk-1"]


def test_screen_chunks_logs_batch_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    """Operators can follow batch starts, completions, verdict counts, and overall progress."""
    chunks = [_chunk(f"chunk-{index}", f"Nội dung {index}.") for index in range(3)]
    messages: list[str] = []

    class _RecordingLogger:
        def info(self, message: str, *args: object) -> None:
            messages.append(message.format(*args))

        def error(self, message: str, *args: object) -> None:
            messages.append(message.format(*args))

    monkeypatch.setattr(chunk_screening, "logger", _RecordingLogger())
    client = _FakeScreeningClient(
        {
            chunk.chunk_id: ChunkScreeningResult(
                chunk_id=chunk.chunk_id,
                has_penalty_content=True,
                has_document_metadata=False,
                reason="Contains legal content.",
            )
            for chunk in chunks
        }
    )

    screen_chunks(
        chunks,
        client,
        batch_size=2,
        max_concurrency=1,
        max_retries=0,
        retry_delay_seconds=0,
    )

    assert "Chunk screening started: 3 chunks, 2 batches, batch_size=2, max_concurrency=1" in messages
    assert "Chunk screening batch 1/2 started: chunks=2" in messages
    assert "Chunk screening batch 1/2 completed: accepted=2, excluded=0, progress=2/3 chunks" in messages
    assert "Chunk screening batch 2/2 started: chunks=1" in messages
    assert "Chunk screening batch 2/2 completed: accepted=1, excluded=0, progress=3/3 chunks" in messages


def test_screen_chunks_calls_on_batch_screened_after_each_batch() -> None:
    """on_batch_screened receives just the newly screened batch's records, once per batch."""
    chunks = [_chunk(f"chunk-{i}", f"Nội dung {i}") for i in range(5)]
    results_by_chunk_id = {
        chunk.chunk_id: ChunkScreeningResult(
            chunk_id=chunk.chunk_id, has_penalty_content=True, has_document_metadata=False, reason=""
        )
        for chunk in chunks
    }

    class _StaticClient:
        def screen(self, batch: Sequence[ChunkScreeningItem]) -> Sequence[ChunkScreeningResult]:
            return [results_by_chunk_id[item.chunk_id] for item in batch]

    batches_seen: list[list[str]] = []

    screen_chunks(
        chunks,
        _StaticClient(),
        batch_size=2,
        max_retries=2,
        retry_delay_seconds=0,
        on_batch_screened=lambda records: batches_seen.append([record.chunk_id for record in records]),
    )

    assert batches_seen == [["chunk-0", "chunk-1"], ["chunk-2", "chunk-3"], ["chunk-4"]]


def test_screen_chunks_checkpoints_matches_and_retries_only_incomplete_chunks() -> None:
    """Valid verdicts persist immediately while duplicate, missing, and unexpected IDs are retried."""
    chunks = [_chunk(f"chunk-{index}", f"Nội dung {index}") for index in range(3)]
    calls: list[list[str]] = []
    checkpoints: list[list[str]] = []

    class _PartialClient:
        def screen(self, batch: Sequence[ChunkScreeningItem]) -> Sequence[ChunkScreeningResult]:
            calls.append([item.chunk_id for item in batch])
            if len(calls) == 1:
                return [
                    ChunkScreeningResult(
                        chunk_id="chunk-0", has_penalty_content=True, has_document_metadata=False, reason="ok"
                    ),
                    ChunkScreeningResult(
                        chunk_id="chunk-1", has_penalty_content=True, has_document_metadata=False, reason="duplicate"
                    ),
                    ChunkScreeningResult(
                        chunk_id="chunk-1", has_penalty_content=True, has_document_metadata=False, reason="duplicate"
                    ),
                    ChunkScreeningResult(
                        chunk_id="unexpected", has_penalty_content=True, has_document_metadata=False, reason="wrong id"
                    ),
                ]
            return [
                ChunkScreeningResult(
                    chunk_id=item.chunk_id, has_penalty_content=True, has_document_metadata=False, reason="ok"
                )
                for item in batch
            ]

    records = screen_chunks(
        chunks,
        _PartialClient(),
        batch_size=3,
        max_retries=1,
        retry_delay_seconds=0,
        on_batch_screened=lambda completed: checkpoints.append([record.chunk_id for record in completed]),
    )

    assert calls == [["chunk-0", "chunk-1", "chunk-2"], ["chunk-1", "chunk-2"]]
    assert checkpoints == [["chunk-0"], ["chunk-1", "chunk-2"]]
    assert [record.chunk_id for record in records] == ["chunk-0", "chunk-1", "chunk-2"]


def test_screen_chunks_keeps_partial_checkpoint_when_missing_chunks_exhaust_retries() -> None:
    """A failed batch still checkpoints every uniquely matched verdict returned before exhaustion."""
    chunks = [_chunk("chunk-0", "Nội dung 0"), _chunk("chunk-1", "Nội dung 1")]
    checkpoints: list[list[str]] = []

    class _IncompleteClient:
        def screen(self, batch: Sequence[ChunkScreeningItem]) -> Sequence[ChunkScreeningResult]:
            return [
                ChunkScreeningResult(
                    chunk_id="chunk-0", has_penalty_content=True, has_document_metadata=False, reason="ok"
                )
            ]

    with pytest.raises(RuntimeError, match="chunk screening left 1 incomplete"):
        screen_chunks(
            chunks,
            _IncompleteClient(),
            batch_size=2,
            max_retries=0,
            retry_delay_seconds=0,
            on_batch_screened=lambda completed: checkpoints.append([record.chunk_id for record in completed]),
        )

    assert checkpoints == [["chunk-0"]]


def test_screen_chunks_retries_a_failing_batch_then_succeeds() -> None:
    """A batch that raises is retried, and a later successful attempt still produces a verdict."""
    chunk = _chunk("chunk-1", "Điều 6 khoản 9 điểm b quy định mức phạt tiền.")
    attempts = 0

    class _FlakyClient:
        def screen(self, batch: Sequence[ChunkScreeningItem]) -> Sequence[ChunkScreeningResult]:
            nonlocal attempts
            attempts += 1
            if attempts < 2:
                msg = "provider timed out"
                raise TimeoutError(msg)
            return [
                ChunkScreeningResult(
                    chunk_id="chunk-1", has_penalty_content=True, has_document_metadata=False, reason="Nêu rõ mức phạt."
                )
            ]

    records = screen_chunks([chunk], _FlakyClient(), batch_size=50, max_retries=2, retry_delay_seconds=0)

    assert attempts == 2
    assert len(records) == 1
    assert records[0].chunk_id == "chunk-1"
    assert records[0].has_penalty_content is True


def test_screen_chunks_skips_a_batch_that_fails_every_attempt() -> None:
    """A batch that keeps raising is skipped, but other batches are still screened."""
    failing = _chunk("chunk-fail", "Nội dung gây lỗi mạng.")
    ok_one = _chunk("chunk-ok-1", "Điều 6 khoản 9 điểm b quy định mức phạt tiền.")
    ok_two = _chunk("chunk-ok-2", "Mục lục")

    class _FailsOnlyForOneChunkClient:
        def screen(self, batch: Sequence[ChunkScreeningItem]) -> Sequence[ChunkScreeningResult]:
            if batch[0].chunk_id == "chunk-fail":
                msg = "provider timed out"
                raise TimeoutError(msg)
            return [
                ChunkScreeningResult(
                    chunk_id=item.chunk_id, has_penalty_content=True, has_document_metadata=False, reason="ok"
                )
                for item in batch
            ]

    with pytest.raises(RuntimeError, match="chunk screening left 1 incomplete"):
        screen_chunks(
            [failing, ok_one, ok_two],
            _FailsOnlyForOneChunkClient(),
            batch_size=1,
            max_retries=2,
            retry_delay_seconds=0,
        )


def test_screen_chunks_respects_max_retries_override() -> None:
    """max_retries=0 means a single attempt; a failure skips the batch immediately."""
    chunk = _chunk("chunk-1", "Điều 6 khoản 9 điểm b quy định mức phạt tiền.")
    attempts = 0

    class _AlwaysFailsClient:
        def screen(self, batch: Sequence[ChunkScreeningItem]) -> Sequence[ChunkScreeningResult]:
            nonlocal attempts
            attempts += 1
            msg = "provider timed out"
            raise TimeoutError(msg)

    with pytest.raises(RuntimeError, match="chunk screening left 1 incomplete"):
        screen_chunks([chunk], _AlwaysFailsClient(), batch_size=50, max_retries=0, retry_delay_seconds=0)

    assert attempts == 1


def _settings() -> Settings:
    """Build runtime settings for an offline BAML request."""
    return Settings(
        llm_base_url="http://localhost:20128/v1",
        llm_api_key="provider-test-key",
        llm_model="provider/test-model",
        llm_timeout_seconds=30,
    )


def _synthesis_settings() -> SynthesisSettings:
    """Build ground-truth synthesis settings for an offline BAML request."""
    return SynthesisSettings.from_path(_SYNTHESIS_CONFIG_PATH)


def test_baml_chunk_screening_client_parses_typed_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """The generated non-stream client returns the typed BAML contract for a batch."""
    client = BamlChunkScreeningClient(_settings(), _synthesis_settings())
    monkeypatch.setattr(
        chunk_screening,
        "execute_baml_request",
        lambda *args, **kwargs: (
            '[{"chunk_id":"chunk-1","has_penalty_content":true,'
            '"has_document_metadata":false,"reason":"Nêu mức phạt tiền cụ thể."}]'
        ),
    )

    results = client.screen(
        [
            ChunkScreeningItem(
                chunk_id="chunk-1",
                source_name="source.docx",
                record_scope="root",
                attachment_member_path=None,
                kind="paragraph_group",
                context_text="Điều 6",
                previous_chunk_text=None,
                text="Điều 6 khoản 9 điểm b quy định mức phạt.",
            )
        ]
    )

    assert results[0].chunk_id == "chunk-1"
    assert results[0].has_penalty_content is True
    assert results[0].has_document_metadata is False
    assert results[0].reason == "Nêu mức phạt tiền cụ thể."
