"""Tests for the merged native DOCX source view."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from legal_rag.parser.document_source import (
    KIND_PARAGRAPH,
    KIND_TABLE,
    extract_document_source,
    write_document_source,
)

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
    assert source.paragraph_count == 2
    assert source.table_count == 1
    assert source.cell_count == 2
    assert source.skipped_empty_paragraph_refs == ("word/document.xml#/body/1",)


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


def test_write_document_source_defaults_to_a_single_file(tmp_path: Path) -> None:
    """Without a manifest path, only the merged source view is written."""
    input_path = tmp_path / "source.docx"
    output_path = tmp_path / "source.jsonl"
    manifest_path = tmp_path / "source.manifest.json"
    _write_docx(input_path)
    source = extract_document_source(input_path)

    write_document_source(output_path, source)

    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert records == [unit.to_dict() for unit in source.units]
    assert {record["kind"] for record in records} == {KIND_PARAGRAPH, KIND_TABLE}
    assert not manifest_path.exists()


def test_write_document_source_writes_manifest_when_requested(tmp_path: Path) -> None:
    """A manifest path opts into the audit sidecar describing the same extraction."""
    input_path = tmp_path / "source.docx"
    output_path = tmp_path / "source.jsonl"
    manifest_path = tmp_path / "source.manifest.json"
    _write_docx(input_path)
    source = extract_document_source(input_path)

    write_document_source(output_path, source, manifest_path=manifest_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest == source.manifest()
    assert manifest["unit_count"] == 3
    assert manifest["table_count"] == 1
