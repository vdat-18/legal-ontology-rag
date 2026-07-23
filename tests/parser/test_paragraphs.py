"""Tests for native DOCX paragraph and run extraction."""

from __future__ import annotations

import zipfile
from pathlib import Path

from legal_rag.parser.paragraphs import extract_docx_paragraphs

_DOCUMENT_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p>
      <w:r><w:t>từ 50 cm</w:t></w:r>
      <w:r><w:rPr><w:vertAlign w:val="superscript"/></w:rPr><w:t>3</w:t></w:r>
      <w:r><w:t xml:space="preserve"> trở lên</w:t></w:r>
    </w:p>
    <w:p><w:r><w:t>   </w:t></w:r></w:p>
    <w:tbl><w:tr><w:tc><w:p><w:r><w:t>table cell</w:t></w:r></w:p></w:tc></w:tr></w:tbl>
    <w:p><w:r><w:t>Sau bảng</w:t></w:r></w:p>
    <w:p>
      <w:r>
        <w:drawing>
          <w:txbxContent>
            <w:p><w:r><w:t>Nội dung textbox</w:t></w:r></w:p>
          </w:txbxContent>
        </w:drawing>
      </w:r>
    </w:p>
    <w:sectPr/>
  </w:body>
</w:document>
"""


def _write_docx(path: Path) -> None:
    """Write the minimal OOXML part required by the paragraph extractor."""
    with zipfile.ZipFile(path, mode="w") as archive:
        archive.writestr("word/document.xml", _DOCUMENT_XML)


def test_extract_docx_paragraphs_preserves_runs_and_source_order(tmp_path: Path) -> None:
    """Runs from one native paragraph remain in one paragraph unit."""
    input_path = tmp_path / "source.docx"
    _write_docx(input_path)

    extraction = extract_docx_paragraphs(input_path)

    assert [unit.sequence_index for unit in extraction.units] == [0, 2, 3]
    assert [unit.body_index for unit in extraction.units] == [0, 3, 4]
    assert [unit.text for unit in extraction.units] == ["từ 50 cm3 trở lên", "Sau bảng", "Nội dung textbox"]
    assert [run.text for run in extraction.units[0].runs] == ["từ 50 cm", "3", " trở lên"]
    assert extraction.units[0].runs[1].vertical_alignment == "superscript"
    assert [run.text for run in extraction.units[2].runs] == ["Nội dung textbox"]
