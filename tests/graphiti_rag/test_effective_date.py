"""Tests for extracting a legal document's own commencement date from its chunk text."""

from __future__ import annotations

from datetime import date

from legal_rag.chunker.models import Chunk
from legal_rag.graphiti_rag.effective_date import extract_effective_date


def _chunk(chunk_index: int, text: str) -> Chunk:
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
        text_token_count=5,
        embedding_token_count=5,
        previous_chunk_id=None,
        next_chunk_id=None,
    )


def test_extracts_word_form_date_from_the_heading_article() -> None:
    """The word-form date in the numbered "Hiệu lực thi hành" article is extracted."""
    chunks = [
        _chunk(0, "Điều 1. Phạm vi điều chỉnh\n\nVăn bản này quy định..."),
        _chunk(1, "Điều 20. Hiệu lực thi hành\n\n1. Nghị định này có hiệu lực thi hành từ ngày 15 tháng 8 năm 2026."),
    ]

    assert extract_effective_date(chunks) == date(2026, 8, 15)


def test_extracts_slash_form_date_from_the_heading_article() -> None:
    """The slash-form date (dd/mm/yyyy) in the heading article is also recognized."""
    chunks = [
        _chunk(0, "Quyết định này có hiệu lực thi hành kể từ ngày 01/01/2025.\n\nĐiều 3. ..."),
    ]

    assert extract_effective_date(chunks) == date(2025, 1, 1)


def test_ignores_unrelated_sub_clause_dates_before_the_real_heading() -> None:
    """A phase-in date for an unrelated sub-provision, in a chunk without the heading, is ignored."""
    chunks = [
        _chunk(0, "1. Quy định này có hiệu lực thi hành từ ngày 01 tháng 01 năm 2028 đối với thiết bị X."),
        _chunk(1, "Điều 20. Hiệu lực thi hành\n\n1. Nghị định này có hiệu lực thi hành từ ngày 15 tháng 8 năm 2026."),
    ]

    assert extract_effective_date(chunks) == date(2026, 8, 15)


def test_falls_back_to_the_next_chunk_when_heading_and_date_are_split() -> None:
    """If the date sentence spills into the next chunk, it is still found."""
    chunks = [
        _chunk(93, "...\n\nĐiều 88. Hiệu lực thi hành"),
        _chunk(94, "1. Luật này có hiệu lực thi hành từ ngày 01 tháng 01 năm 2025, trừ trường hợp..."),
    ]

    assert extract_effective_date(chunks) == date(2025, 1, 1)


def test_falls_back_to_whole_document_search_when_no_heading_exists() -> None:
    """Document types without a numbered heading (e.g. Quyết định) still resolve via fallback."""
    chunks = [
        _chunk(0, "Điều 1. Ban hành kèm theo Quyết định này..."),
        _chunk(1, "Quyết định này có hiệu lực thi hành kể từ ngày 01/01/2025.\n\nĐiều 3. ..."),
    ]

    assert extract_effective_date(chunks) == date(2025, 1, 1)


def test_returns_none_when_nothing_matches() -> None:
    """A document with no recognizable commencement-date clause returns None, not a guess."""
    chunks = [_chunk(0, "Điều 1. Phạm vi điều chỉnh\n\nVăn bản này không có điều khoản hiệu lực rõ ràng.")]

    assert extract_effective_date(chunks) is None
