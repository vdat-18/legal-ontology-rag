"""Tests for flagging legal document citations outside the indexed corpus."""

from __future__ import annotations

import pytest

from legal_rag.graphiti_rag.provenance import find_unindexed_document_citations


class _FakeDocumentCitationClient:
    """Return a fixed list of cited document numbers without making any real LLM call."""

    def __init__(self, citations: list[str]) -> None:
        self.calls: list[str] = []
        self._citations = citations

    def extract_citations(self, text: str) -> list[str]:
        """Record the scanned text and return the configured citations."""
        self.calls.append(text)
        return list(self._citations)


def test_returns_empty_when_all_citations_are_known() -> None:
    """A citation matching a known document number is not flagged."""
    client = _FakeDocumentCitationClient(["168/2024/NĐ-CP"])

    result = find_unindexed_document_citations(
        ["Vi phạm quy định tại Nghị định 168/2024/NĐ-CP."], frozenset({"168/2024/NĐ-CP"}), client
    )

    assert result == []


def test_flags_a_citation_not_in_the_known_set() -> None:
    """A document number not in the known document-number set is flagged."""
    client = _FakeDocumentCitationClient(["100/2019/NĐ-CP"])

    result = find_unindexed_document_citations(["sửa đổi Nghị định số 100/2019/NĐ-CP"], frozenset(), client)

    assert result == ["100/2019/NĐ-CP"]


def test_deduplicates_and_sorts_citations_across_multiple_texts() -> None:
    """The same unindexed citation repeated across texts appears once, sorted."""
    client = _FakeDocumentCitationClient(["100/2019/NĐ-CP", "65/2024/TT-BCA", "100/2019/NĐ-CP"])

    result = find_unindexed_document_citations(
        ["Theo Nghị định 100/2019/NĐ-CP...", "...và 65/2024/TT-BCA."], frozenset(), client
    )

    assert result == ["100/2019/NĐ-CP", "65/2024/TT-BCA"]


def test_scans_all_texts_in_one_call() -> None:
    """All texts are joined into a single scan, keeping the check to one model call."""
    client = _FakeDocumentCitationClient([])

    find_unindexed_document_citations(["first text", "second text"], frozenset(), client)

    assert client.calls == ["first text\nsecond text"]


def test_discards_a_document_number_the_model_did_not_copy_from_the_text() -> None:
    """Model output is untrusted: a number absent from the scanned text is not reported."""
    client = _FakeDocumentCitationClient(["100/2019/NĐ-CP", "999/1999/NĐ-CP"])

    result = find_unindexed_document_citations(["Theo Nghị định 100/2019/NĐ-CP."], frozenset(), client)

    assert result == ["100/2019/NĐ-CP"]


@pytest.mark.parametrize("hallucinated", ["68/2024/NĐ-CP", "168/2024/NĐ-C", "168/2024"])
def test_discards_a_document_number_copied_from_inside_a_longer_number(hallucinated: str) -> None:
    """A model-returned substring is not accepted as an independently cited document number."""
    client = _FakeDocumentCitationClient([hallucinated])

    result = find_unindexed_document_citations(["Nghị định 168/2024/NĐ-CP."], frozenset(), client)

    assert result == []


def test_returns_empty_when_no_document_is_cited() -> None:
    """Text citing no legal document flags nothing."""
    client = _FakeDocumentCitationClient([])

    result = find_unindexed_document_citations(["Không có căn cứ để trả lời câu hỏi này."], frozenset(), client)

    assert result == []
