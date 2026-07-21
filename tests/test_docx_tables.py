"""Tests for native DOCX table extraction."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from legal_rag.docx_tables import extract_docx_tables, write_docx_tables

_DOCUMENT_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>Before</w:t></w:r></w:p>
    <w:tbl>
      <w:tr>
        <w:tc>
          <w:tcPr><w:gridSpan w:val="2"/><w:vMerge w:val="restart"/></w:tcPr>
          <w:p><w:r><w:t>50 cm</w:t></w:r><w:r><w:rPr><w:vertAlign w:val="superscript"/></w:rPr><w:t>3</w:t></w:r></w:p>
          <w:p><w:r><w:t>   </w:t></w:r></w:p>
        </w:tc>
        <w:tc><w:p><w:r><w:t>Cell B</w:t></w:r></w:p></w:tc>
      </w:tr>
      <w:tr>
        <w:tc><w:tcPr><w:vMerge/></w:tcPr><w:p><w:r><w:t>Cell C</w:t></w:r></w:p></w:tc>
      </w:tr>
    </w:tbl>
    <w:p><w:r><w:t>After</w:t></w:r></w:p>
    <w:sectPr/>
  </w:body>
</w:document>
"""


def _write_docx(path: Path) -> None:
    """Write the minimal OOXML part required by the table extractor."""
    with zipfile.ZipFile(path, mode="w") as archive:
        archive.writestr("word/document.xml", _DOCUMENT_XML)


def test_extract_docx_tables_preserves_rows_cells_and_runs(tmp_path: Path) -> None:
    """Native table boundaries and formatting remain explicit."""
    input_path = tmp_path / "source.docx"
    _write_docx(input_path)

    extraction = extract_docx_tables(input_path)

    assert len(extraction.units) == 1
    assert extraction.cell_count == 3
    assert extraction.raw_cell_paragraph_count == 4
    assert extraction.skipped_empty_paragraph_refs == ("word/document.xml#/body/1/row/0/cell/0/paragraph/1",)
    table = extraction.units[0]
    assert table.body_index == 1
    assert len(table.rows) == 2
    assert [len(row.cells) for row in table.rows] == [2, 1]
    assert table.rows[0].cells[0].grid_span == 2
    assert table.rows[0].cells[0].vertical_merge == "restart"
    assert table.rows[1].cells[0].vertical_merge == "continue"
    assert table.rows[0].cells[0].text == "50 cm3"
    assert table.rows[0].cells[0].paragraphs[0].runs[1].vertical_alignment == "superscript"


def test_write_docx_tables_writes_jsonl_and_manifest(tmp_path: Path) -> None:
    """Table output and manifest describe the same extraction."""
    input_path = tmp_path / "source.docx"
    output_path = tmp_path / "tables.jsonl"
    manifest_path = tmp_path / "tables.manifest.json"
    _write_docx(input_path)
    extraction = extract_docx_tables(input_path)

    write_docx_tables(output_path, manifest_path, extraction)

    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert records == [unit.to_dict() for unit in extraction.units]
    assert manifest == extraction.manifest()
