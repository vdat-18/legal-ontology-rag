"""Tests for the merged native DOCX source view."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from legal_rag.parser.document_source import (
    KIND_PARAGRAPH,
    KIND_TABLE,
    extract_document_source,
)
from legal_rag.parser.docx_package import InvalidDocxPackageError

_DOCUMENT_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>Điều 1. Phạm vi điều chỉnh</w:t></w:r></w:p>
    <w:p><w:r><w:t>   </w:t></w:r></w:p>
    <w:tbl>
      <w:tr>
        <w:tc><w:p><w:r><w:t>Hành vi</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>Mức phạt</w:t></w:r></w:p></w:tc>
      </w:tr>
    </w:tbl>
    <w:p><w:r><w:t>Sau bảng</w:t></w:r></w:p>
    <w:sectPr/>
  </w:body>
</w:document>
"""


def _write_docx(path: Path) -> None:
    """Write the minimal OOXML part required by the source extractor."""
    with zipfile.ZipFile(path, mode="w") as archive:
        archive.writestr("word/document.xml", _DOCUMENT_XML)


def test_extract_document_source_interleaves_paragraphs_and_tables(tmp_path: Path) -> None:
    """Paragraphs and tables share one strictly increasing document order."""
    input_path = tmp_path / "source.docx"
    _write_docx(input_path)

    source = extract_document_source(input_path)

    assert [unit.kind for unit in source.units] == [KIND_PARAGRAPH, KIND_TABLE, KIND_PARAGRAPH]
    assert [unit.sequence_index for unit in source.units] == [0, 1, 2]
    assert [unit.body_index for unit in source.units] == [0, 2, 3]
    assert source.units[0].text == "Điều 1. Phạm vi điều chỉnh"
    assert source.units[2].text == "Sau bảng"
    assert source.units[1].rows[0].cells[1].text == "Mức phạt"


def test_document_source_unit_ids_match_native_extractors(tmp_path: Path) -> None:
    """Merged units reuse the content-addressed identity of native units."""
    input_path = tmp_path / "source.docx"
    _write_docx(input_path)

    source = extract_document_source(input_path)

    unit_ids = [unit.unit_id for unit in source.units]
    assert len(unit_ids) == len(set(unit_ids))
    paragraph = source.units[0]
    assert paragraph.to_dict()["runs"] == [run.to_dict() for run in paragraph.runs]
    assert "rows" not in paragraph.to_dict()
    table = source.units[1]
    assert "text" not in table.to_dict()


def test_extract_document_source_preserves_package_errors(tmp_path: Path) -> None:
    """Invalid DOCX input exposes the package boundary error unchanged."""
    input_path = tmp_path / "invalid.docx"
    input_path.write_bytes(b"not a DOCX archive")

    with pytest.raises(InvalidDocxPackageError):
        extract_document_source(input_path)
