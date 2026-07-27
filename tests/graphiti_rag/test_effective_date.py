"""Tests for extracting a legal document's own commencement date from its chunk text."""

from __future__ import annotations

from datetime import date

from legal_rag.chunker.models import Chunk
from legal_rag.graphiti_rag.effective_date import extract_effective_date


class _FakeEffectiveDateClient:
    """Record each window of text it was asked to extract from, returning results in order."""

    def __init__(self, results: list[date | None]) -> None:
        self.calls: list[str] = []
        self._results = list(results)

    def extract(self, document_text: str) -> date | None:
        """Record the call and return the next configured result."""
        self.calls.append(document_text)
        return self._results[len(self.calls) - 1]


def _chunk(chunk_index: int, text: str, text_token_count: int = 5) -> Chunk:
    """Build one minimal chunk for effective-date extraction tests."""
    return Chunk(
        chunk_id=f"chunk-{chunk_index}",
        content_sha256="content-hash",
        chunker_config_sha256="config-hash",
        scope_id="scope-id",
        chunk_index=chunk_index,
        kind="paragraph_group",
        source_name="doc.docx",
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
        text_token_count=text_token_count,
        embedding_token_count=text_token_count,
        previous_chunk_id=None,
        next_chunk_id=None,
    )


def test_returns_the_clients_result_when_the_whole_document_fits_one_window() -> None:
    """A short document is sent to the client as a single window, in chunk-index order."""
    chunks = [_chunk(0, "Điều 1. Phạm vi điều chỉnh"), _chunk(1, "Điều 20. Hiệu lực thi hành...")]
    client = _FakeEffectiveDateClient([date(2026, 8, 15)])

    assert extract_effective_date(chunks, client) == date(2026, 8, 15)
    assert client.calls == ["Điều 1. Phạm vi điều chỉnh\nĐiều 20. Hiệu lực thi hành..."]


def test_returns_none_when_the_client_finds_no_date_anywhere() -> None:
    """A None result from the client (no stated commencement date) is passed through."""
    chunks = [_chunk(0, "Điều 1. Phạm vi điều chỉnh")]
    client = _FakeEffectiveDateClient([None])

    assert extract_effective_date(chunks, client) is None


def test_sorts_chunks_by_index_before_windowing_even_if_passed_out_of_order() -> None:
    """Chunks passed out of order are sorted by chunk_index before windows are built."""
    chunks = [_chunk(2, "third"), _chunk(0, "first"), _chunk(1, "second")]
    client = _FakeEffectiveDateClient([date(2025, 1, 1)])

    extract_effective_date(chunks, client)

    assert client.calls == ["first\nsecond\nthird"]


def test_splits_into_multiple_windows_once_the_token_budget_is_exceeded() -> None:
    """Chunks that together exceed the per-window token budget are split into separate windows."""
    chunks = [
        _chunk(0, "first-half", text_token_count=39_000),
        _chunk(1, "second-half", text_token_count=39_000),
    ]
    client = _FakeEffectiveDateClient([None, date(2025, 1, 1)])

    result = extract_effective_date(chunks, client)

    assert result == date(2025, 1, 1)
    assert client.calls == ["second-half", "first-half"]


def test_scans_windows_from_the_end_and_stops_at_the_first_non_null_result() -> None:
    """Windows are checked last-to-first; the first non-null result short-circuits the scan."""
    chunks = [
        _chunk(0, "early", text_token_count=39_000),
        _chunk(1, "middle", text_token_count=39_000),
        _chunk(2, "late", text_token_count=39_000),
    ]
    client = _FakeEffectiveDateClient([date(2025, 1, 1), date(2099, 1, 1)])

    result = extract_effective_date(chunks, client)

    assert result == date(2025, 1, 1)
    assert client.calls == ["late"]
