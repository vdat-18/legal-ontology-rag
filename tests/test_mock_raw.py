"""Offline tests for the mock raw-document loader (parser input fixtures)."""

from __future__ import annotations

from datetime import date

from legal_rag.mock_raw import iter_mock_raw_documents, mock_raw_dir


def test_mock_raw_dir_exists() -> None:
    """The bundled mock raw-document directory is present."""
    assert mock_raw_dir().is_dir()


def test_all_mock_documents_validate() -> None:
    """Every mock file loads as a valid RawDocument with English meta fields."""
    documents = list(iter_mock_raw_documents())

    assert len(documents) == 100
    numbers = {document.meta.document_number for document in documents}
    assert len(numbers) == 100  # unique
    for document in documents:
        assert document.ok is True
        assert document.method
        assert "[MOCK]" in document.text
        assert document.meta.document_number
        assert document.meta.effective_date is not None


def test_temporal_chain_is_representable() -> None:
    """Mock documents include a 100/2019 -> 123/2021 -> 168/2024 effective-date chain."""
    by_number = {d.meta.document_number: d.meta for d in iter_mock_raw_documents()}

    assert by_number["100/2019/NĐ-CP"].effective_date == date(2020, 1, 1)
    assert by_number["123/2021/NĐ-CP"].document_type == "nghị định sửa đổi"
    assert by_number["123/2021/NĐ-CP"].expiry_date == date(2025, 1, 1)
    assert by_number["168/2024/NĐ-CP"].effective_date == date(2025, 1, 1)
    assert by_number["168/2024/NĐ-CP"].expiry_date is None
