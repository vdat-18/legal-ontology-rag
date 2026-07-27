"""Tests for flagging legal document citations outside the indexed corpus."""

from __future__ import annotations

from legal_rag.graphiti_rag.provenance import find_unindexed_document_citations


def test_returns_empty_when_all_citations_are_known() -> None:
    """A citation matching a known so_hieu is not flagged."""
    result = find_unindexed_document_citations(
        ["Vi phạm quy định tại Nghị định 168/2024/NĐ-CP."], frozenset({"168/2024/NĐ-CP"})
    )

    assert result == []


def test_flags_a_citation_not_in_the_known_set() -> None:
    """A document number not in known_so_hieu is flagged."""
    result = find_unindexed_document_citations(
        ["sửa đổi một số điều của Nghị định số 100/2019/NĐ-CP ngày 30 tháng 12 năm 2019"],
        frozenset({"168/2024/NĐ-CP"}),
    )

    assert result == ["100/2019/NĐ-CP"]


def test_deduplicates_and_sorts_citations_across_multiple_texts() -> None:
    """The same unindexed citation repeated across texts appears once, sorted."""
    result = find_unindexed_document_citations(
        [
            "Theo Nghị định 100/2019/NĐ-CP...",
            "...và Thông tư 65/2024/TT-BCA, cùng Nghị định 100/2019/NĐ-CP.",
        ],
        frozenset(),
    )

    assert result == ["100/2019/NĐ-CP", "65/2024/TT-BCA"]


def test_returns_empty_when_no_document_number_is_mentioned() -> None:
    """Text with no document-number-shaped substring flags nothing."""
    result = find_unindexed_document_citations(["Không có căn cứ để trả lời câu hỏi này."], frozenset())

    assert result == []
