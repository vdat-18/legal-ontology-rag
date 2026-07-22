"""Tests for native DOCX table extraction."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from legal_rag.docx_tables import (
    CELL_BLOCK_PARAGRAPH_KIND,
    CELL_BLOCK_TABLE_KIND,
    TableCellTable,
    extract_docx_tables,
    write_docx_tables,
)

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


_NESTED_DOCUMENT_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:tbl>
      <w:tr>
        <w:tc>
          <w:p><w:r><w:t>Above</w:t></w:r></w:p>
          <w:tbl>
            <w:tr>
              <w:tc><w:p><w:r><w:t>Inner A</w:t></w:r></w:p></w:tc>
              <w:tc><w:p><w:r><w:t>Inner B</w:t></w:r></w:p></w:tc>
            </w:tr>
          </w:tbl>
          <w:p><w:r><w:t>Below</w:t></w:r></w:p>
        </w:tc>
      </w:tr>
    </w:tbl>
    <w:sectPr/>
  </w:body>
</w:document>
"""


def _write_docx(path: Path, document_xml: str = _DOCUMENT_XML) -> None:
    """Write the minimal OOXML part required by the table extractor."""
    with zipfile.ZipFile(path, mode="w") as archive:
        archive.writestr("word/document.xml", document_xml)


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
    assert table.rows[0].cells[0].blocks[0].runs[1].vertical_alignment == "superscript"


def test_extract_docx_tables_preserves_nested_table_blocks(tmp_path: Path) -> None:
    """A nested table stays an ordered block inside its cell without flattening."""
    input_path = tmp_path / "nested.docx"
    _write_docx(input_path, _NESTED_DOCUMENT_XML)

    extraction = extract_docx_tables(input_path)

    assert len(extraction.units) == 1
    assert extraction.cell_count == 3
    cell = extraction.units[0].rows[0].cells[0]
    assert cell.text == "Above\nBelow"
    assert [block.to_dict()["kind"] for block in cell.blocks] == [
        CELL_BLOCK_PARAGRAPH_KIND,
        CELL_BLOCK_TABLE_KIND,
        CELL_BLOCK_PARAGRAPH_KIND,
    ]
    nested = cell.blocks[1]
    assert isinstance(nested, TableCellTable)
    assert nested.source_ref == "word/document.xml#/body/0/row/0/cell/0/table/0"
    assert [inner.text for inner in nested.rows[0].cells] == ["Inner A", "Inner B"]
    assert nested.rows[0].cells[0].blocks[0].source_ref == (
        "word/document.xml#/body/0/row/0/cell/0/table/0/row/0/cell/0/paragraph/0"
    )


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
