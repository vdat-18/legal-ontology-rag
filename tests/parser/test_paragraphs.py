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

_ANNOTATED_DOCUMENT_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document
    xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p>
      <w:r><w:t>Luật xử lý vi phạm hành chính</w:t></w:r>
      <w:r><w:fldChar w:fldCharType="begin"/></w:r>
      <w:r><w:instrText xml:space="preserve"> HYPERLINK \\l "_ftn1" </w:instrText></w:r>
      <w:r><w:fldChar w:fldCharType="separate"/></w:r>
      <w:r><w:t>[1]</w:t></w:r>
      <w:r><w:fldChar w:fldCharType="end"/></w:r>
    </w:p>
    <w:p>
      <w:r><w:t xml:space="preserve">Tham khảo: </w:t></w:r>
      <w:r><w:fldChar w:fldCharType="begin"/></w:r>
      <w:r><w:instrText xml:space="preserve"> HYPERLINK "https://example.com/reference" </w:instrText></w:r>
      <w:r><w:fldChar w:fldCharType="separate"/></w:r>
      <w:r><w:t>Tại đây</w:t></w:r>
      <w:r><w:fldChar w:fldCharType="end"/></w:r>
    </w:p>
    <w:p>
      <w:bookmarkStart w:id="600" w:name="_ftn1"/>
      <w:bookmarkEnd w:id="600"/>
      <w:r><w:t>[1] Nội dung chú thích</w:t></w:r>
    </w:p>
    <w:p>
      <w:hyperlink r:id="rId5"><w:r><w:t>Xem hướng dẫn</w:t></w:r></w:hyperlink>
      <w:r><w:t xml:space="preserve"> hoặc </w:t></w:r>
      <w:hyperlink w:anchor="_ftn1"><w:r><w:t>chú thích</w:t></w:r></w:hyperlink>
    </w:p>
    <w:p><w:bookmarkStart w:id="700" w:name="empty_target"/><w:bookmarkEnd w:id="700"/></w:p>
  </w:body>
</w:document>
"""

_DOCUMENT_RELATIONSHIPS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship
      Id="rId5"
      Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink"
      Target="https://example.com/guide"
      TargetMode="External"/>
</Relationships>
"""


def _write_docx(path: Path, document_xml: str = _DOCUMENT_XML, relationships_xml: str | None = None) -> None:
    """Write the minimal OOXML part required by the paragraph extractor."""
    with zipfile.ZipFile(path, mode="w") as archive:
        archive.writestr("word/document.xml", document_xml)
        if relationships_xml is not None:
            archive.writestr("word/_rels/document.xml.rels", relationships_xml)


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


def test_extract_docx_paragraphs_preserves_fields_hyperlinks_and_bookmarks(tmp_path: Path) -> None:
    """Native inline relationships remain attached to their exact visible spans."""
    input_path = tmp_path / "annotated.docx"
    _write_docx(input_path, _ANNOTATED_DOCUMENT_XML, _DOCUMENT_RELATIONSHIPS_XML)

    extraction = extract_docx_paragraphs(input_path)

    assert len(extraction.units) == 5
    reference = extraction.units[0]
    assert reference.text == "Luật xử lý vi phạm hành chính[1]"
    assert reference.fields[0].instruction == 'HYPERLINK \\l "_ftn1"'
    assert reference.fields[0].field_type == "HYPERLINK"
    assert reference.fields[0].result_text == "[1]"
    assert reference.fields[0].start_offset == len("Luật xử lý vi phạm hành chính")
    assert reference.fields[0].end_offset == len(reference.text)
    assert reference.fields[0].target is None
    assert reference.fields[0].anchor == "_ftn1"

    external_field = extraction.units[1].fields[0]
    assert external_field.result_text == "Tại đây"
    assert external_field.target == "https://example.com/reference"
    assert external_field.anchor is None

    footnote = extraction.units[2]
    assert footnote.bookmarks[0].bookmark_id == "600"
    assert footnote.bookmarks[0].name == "_ftn1"
    assert footnote.bookmarks[0].start_offset == 0
    assert footnote.bookmarks[0].end_offset == 0

    links = extraction.units[3]
    assert [hyperlink.result_text for hyperlink in links.hyperlinks] == ["Xem hướng dẫn", "chú thích"]
    assert links.hyperlinks[0].relationship_id == "rId5"
    assert links.hyperlinks[0].target == "https://example.com/guide"
    assert links.hyperlinks[0].external is True
    assert links.hyperlinks[1].anchor == "_ftn1"

    empty_bookmark = extraction.units[4]
    assert empty_bookmark.text == ""
    assert empty_bookmark.bookmarks[0].name == "empty_target"
