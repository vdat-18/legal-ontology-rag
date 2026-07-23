"""Tests for structural context selection around DOCX images."""

from __future__ import annotations

from legal_rag.parser.document_source import KIND_PARAGRAPH, KIND_TABLE, DocumentSource, DocumentSourceUnit
from legal_rag.parser.image_context import ImageContextRelation, build_image_context
from legal_rag.parser.images import DocxImage
from legal_rag.parser.tables import TableCell, TableCellTable, TableRow


def _image(*, body_index: int, parent_source_ref: str) -> DocxImage:
    """Build one image occurrence at a native body location."""
    return DocxImage(
        schema_version="1.0.0",
        image_id="image-id",
        source_name="source.docx",
        source_sha256="source-hash",
        sequence_index=0,
        body_index=body_index,
        source_ref=f"{parent_source_ref}/run/0/image/0",
        parent_source_ref=parent_source_ref,
        run_index=0,
        image_index=0,
        relationship_id="rId1",
        package_path="word/media/image.png",
        media_sha256="media-hash",
        mime_type="image/png",
        asset_name="media-hash.png",
        media_bytes=b"image",
    )


def _paragraph_unit(*, body_index: int, text: str) -> DocumentSourceUnit:
    """Build one paragraph source unit."""
    source_ref = f"word/document.xml#/body/{body_index}"
    return DocumentSourceUnit(
        schema_version="1.1.0",
        unit_id=f"paragraph-{body_index}",
        source_name="source.docx",
        source_sha256="source-hash",
        sequence_index=body_index,
        body_index=body_index,
        source_ref=source_ref,
        kind=KIND_PARAGRAPH,
        payload_sha256=f"payload-{body_index}",
        text=text,
        runs=(),
        rows=None,
    )


def _source(units: tuple[DocumentSourceUnit, ...]) -> DocumentSource:
    """Build a document source around selected units."""
    return DocumentSource(
        source_name="source.docx",
        source_sha256="source-hash",
        units=units,
    )


def test_table_image_uses_text_from_its_native_row() -> None:
    """A table image receives cell text from the same row, not adjacent units."""
    table_ref = "word/document.xml#/body/4"
    row_ref = f"{table_ref}/row/2"
    cells = (
        TableCell(0, f"{row_ref}/cell/0", 1, None, "Ô tô tải tự đổ", ()),
        TableCell(1, f"{row_ref}/cell/1", 1, None, "Thùng hàng có cơ cấu nâng", ()),
        TableCell(2, f"{row_ref}/cell/2", 1, None, "", ()),
    )
    table = DocumentSourceUnit(
        schema_version="1.1.0",
        unit_id="table-4",
        source_name="source.docx",
        source_sha256="source-hash",
        sequence_index=0,
        body_index=4,
        source_ref=table_ref,
        kind=KIND_TABLE,
        payload_sha256="payload-table",
        text=None,
        runs=None,
        rows=(TableRow(2, row_ref, cells),),
    )
    image = _image(body_index=4, parent_source_ref=f"{row_ref}/cell/2/paragraph/0")

    context = build_image_context(image, _source((table,)), max_characters=1000, surrounding_units=3)

    assert [entry.text for entry in context.entries] == ["Ô tô tải tự đổ", "Thùng hàng có cơ cấu nâng"]
    assert [entry.relation for entry in context.entries] == [
        ImageContextRelation.SAME_TABLE_ROW,
        ImageContextRelation.SAME_TABLE_ROW,
    ]
    assert [entry.distance for entry in context.entries] == [0, 0]
    assert context.source_refs == (f"{row_ref}/cell/0", f"{row_ref}/cell/1")


def test_image_only_paragraph_uses_immediate_surrounding_units() -> None:
    """An omitted empty image paragraph receives native units around its body index."""
    source = _source(
        (
            _paragraph_unit(body_index=1, text="Nội dung trước ảnh"),
            _paragraph_unit(body_index=3, text="Nội dung sau ảnh"),
        )
    )
    image = _image(body_index=2, parent_source_ref="word/document.xml#/body/2")

    context = build_image_context(image, source, max_characters=1000, surrounding_units=3)

    assert [entry.text for entry in context.entries] == ["Nội dung trước ảnh", "Nội dung sau ảnh"]
    assert [entry.relation for entry in context.entries] == [ImageContextRelation.BEFORE, ImageContextRelation.AFTER]
    assert [entry.distance for entry in context.entries] == [1, 1]


def test_nested_table_image_uses_the_innermost_row() -> None:
    """Nested-table provenance resolves to nested cells rather than the outer row."""
    table_ref = "word/document.xml#/body/4"
    outer_row_ref = f"{table_ref}/row/0"
    outer_cell_ref = f"{outer_row_ref}/cell/0"
    nested_table_ref = f"{outer_cell_ref}/table/0"
    nested_row_ref = f"{nested_table_ref}/row/0"
    nested_cells = (
        TableCell(0, f"{nested_row_ref}/cell/0", 1, None, "Nhãn trong bảng lồng", ()),
        TableCell(1, f"{nested_row_ref}/cell/1", 1, None, "", ()),
    )
    nested_table = TableCellTable(0, nested_table_ref, (TableRow(0, nested_row_ref, nested_cells),))
    outer_cell = TableCell(0, outer_cell_ref, 1, None, "Văn bản ngoài", (nested_table,))
    table = DocumentSourceUnit(
        schema_version="1.1.0",
        unit_id="table-4",
        source_name="source.docx",
        source_sha256="source-hash",
        sequence_index=0,
        body_index=4,
        source_ref=table_ref,
        kind=KIND_TABLE,
        payload_sha256="payload-table",
        text=None,
        runs=None,
        rows=(TableRow(0, outer_row_ref, (outer_cell,)),),
    )
    image = _image(body_index=4, parent_source_ref=f"{nested_row_ref}/cell/1/paragraph/0")

    context = build_image_context(image, _source((table,)), max_characters=1000, surrounding_units=3)

    assert [entry.text for entry in context.entries] == ["Nhãn trong bảng lồng"]
    assert context.source_refs == (f"{nested_row_ref}/cell/0",)


def test_text_bearing_image_paragraph_prioritizes_its_own_text() -> None:
    """Direct paragraph text precedes supplemental surrounding context."""
    source = _source(
        (
            _paragraph_unit(body_index=1, text="Trước"),
            _paragraph_unit(body_index=2, text="Chú thích cùng đoạn"),
            _paragraph_unit(body_index=3, text="Sau"),
        )
    )
    image = _image(body_index=2, parent_source_ref="word/document.xml#/body/2")

    context = build_image_context(image, source, max_characters=1000, surrounding_units=3)

    assert [entry.text for entry in context.entries] == ["Chú thích cùng đoạn", "Trước", "Sau"]
    assert [entry.relation for entry in context.entries] == [
        ImageContextRelation.SAME_PARAGRAPH,
        ImageContextRelation.BEFORE,
        ImageContextRelation.AFTER,
    ]
    assert [entry.distance for entry in context.entries] == [0, 1, 1]


def test_image_only_paragraph_uses_three_units_on_each_side() -> None:
    """A wider structural window includes captions beyond the immediate neighbor."""
    source = _source(
        tuple(_paragraph_unit(body_index=index, text=f"Đoạn {index}") for index in (*range(1, 4), *range(5, 8)))
    )
    image = _image(body_index=4, parent_source_ref="word/document.xml#/body/4")

    context = build_image_context(image, source, max_characters=1000, surrounding_units=3)

    assert [entry.text for entry in context.entries] == [
        "Đoạn 3",
        "Đoạn 5",
        "Đoạn 2",
        "Đoạn 6",
        "Đoạn 1",
        "Đoạn 7",
    ]
    assert [entry.relation for entry in context.entries] == [
        ImageContextRelation.BEFORE,
        ImageContextRelation.AFTER,
        ImageContextRelation.BEFORE,
        ImageContextRelation.AFTER,
        ImageContextRelation.BEFORE,
        ImageContextRelation.AFTER,
    ]
    assert [entry.distance for entry in context.entries] == [1, 1, 2, 2, 3, 3]


def test_context_limit_preserves_structural_priority() -> None:
    """Character limits truncate deterministically after nearest entries are prioritized."""
    source = _source(
        (
            _paragraph_unit(body_index=1, text="abcdef"),
            _paragraph_unit(body_index=3, text="ghijkl"),
        )
    )
    image = _image(body_index=2, parent_source_ref="word/document.xml#/body/2")

    context = build_image_context(image, source, max_characters=8, surrounding_units=3)

    assert [entry.text for entry in context.entries] == ["abcdef", "gh"]
    assert [entry.relation for entry in context.entries] == [ImageContextRelation.BEFORE, ImageContextRelation.AFTER]
    assert [entry.distance for entry in context.entries] == [1, 1]
