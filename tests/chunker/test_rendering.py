"""Tests for deterministic source-unit rendering and provenance."""

from __future__ import annotations

from legal_rag.chunker.models import SourceKind, SourceRecord, SourceScope
from legal_rag.chunker.rendering import render_source_scope


def _scope(*records: SourceRecord) -> SourceScope:
    """Build one root scope for rendering tests."""
    return SourceScope(
        scope_id="scope-id",
        source_name="source.docx",
        source_sha256="source-hash",
        record_scope="root",
        parent_attachment_id=None,
        attachment_member_path=None,
        attachment_member_sha256=None,
        records=records,
    )


def _record(kind: SourceKind, payload: dict[str, object], *, order_index: int = 0) -> SourceRecord:
    """Build one parser record with shared provenance."""
    return SourceRecord(
        unit_id=f"{kind}-unit",
        source_ref=f"source/{order_index}",
        kind=kind,
        order_index=order_index,
        payload=payload,
    )


def _heading(text: str, bookmark_name: str, order_index: int) -> SourceRecord:
    """Build one full-paragraph OOXML structural heading."""
    return _record(
        "paragraph",
        {
            "text": text,
            "bookmarks": [
                {
                    "name": bookmark_name,
                    "start_offset": 0,
                    "end_offset": len(text),
                }
            ],
        },
        order_index=order_index,
    )


def test_render_carries_parser_derived_legal_structure_to_later_paragraphs() -> None:
    """Later leaf text inherits chapter, section, and article headings from OOXML bookmarks."""
    scope = _scope(
        _heading("Chương II", "chuong_2", 0),
        _heading("HÀNH VI VI PHẠM", "chuong_2_name", 1),
        _heading("Mục 1. VI PHẠM QUY TẮC GIAO THÔNG", "muc_1_2", 2),
        _heading("Điều 6. Xử phạt người điều khiển xe ô tô", "dieu_6", 3),
        _record("paragraph", {"text": "a) Không chấp hành hiệu lệnh."}, order_index=4),
    )

    units = render_source_scope(scope)

    assert [(entry.level, entry.text) for entry in units[-1].structural_context] == [
        ("chapter", "Chương II\nHÀNH VI VI PHẠM"),
        ("section", "Mục 1. VI PHẠM QUY TẮC GIAO THÔNG"),
        ("article", "Điều 6. Xử phạt người điều khiển xe ô tô"),
    ]
    assert units[-1].structural_context[0].source_refs == ("source/0", "source/1")


def test_render_ignores_bookmarks_that_do_not_cover_a_complete_heading() -> None:
    """Inline bookmarks never become structural ancestry for later content."""
    record = _record(
        "paragraph",
        {
            "text": "Dẫn chiếu Điều 6 của Nghị định này.",
            "bookmarks": [{"name": "dieu_6", "start_offset": 9, "end_offset": 15}],
        },
    )

    units = render_source_scope(_scope(record))

    assert units[0].structural_context == ()


def test_render_accepts_structural_bookmarks_with_only_trailing_whitespace_outside() -> None:
    """DOCX heading bookmarks may exclude harmless trailing paragraph whitespace."""
    text = "Điều 12. Xử phạt các hành vi vi phạm khác "
    heading = _record(
        "paragraph",
        {
            "text": text,
            "bookmarks": [
                {
                    "name": "dieu_12",
                    "start_offset": 0,
                    "end_offset": len(text.rstrip()),
                }
            ],
        },
    )

    units = render_source_scope(_scope(heading))

    assert units[0].structural_context[0].text == text.rstrip()


def test_render_paragraph_and_attached_images_without_losing_order_or_provenance() -> None:
    """A text record is followed by each independently searchable image caption."""
    record = _record(
        "paragraph",
        {
            "text": "Điều 5. Mức xử phạt.",
            "images": [
                {
                    "image_id": "image-1",
                    "source_ref": "source/0/image/1",
                    "parent_source_ref": "source/0",
                    "description_vi": "Biển báo giới hạn tốc độ.",
                },
                {
                    "image_id": "image-2",
                    "source_ref": "source/0/image/2",
                    "parent_source_ref": "source/0",
                    "description_vi": "Sơ đồ vị trí đặt biển báo.",
                },
            ],
        },
    )

    units = render_source_scope(_scope(record))

    assert [unit.kind for unit in units] == ["paragraph", "image", "image"]
    assert [unit.text for unit in units] == [
        "Điều 5. Mức xử phạt.",
        "Biển báo giới hạn tốc độ.",
        "Sơ đồ vị trí đặt biển báo.",
    ]
    assert units[1].unit_ids == ("paragraph-unit", "image-1")
    assert units[1].source_refs == ("source/0/image/1", "source/0")


def test_render_nested_table_preserves_structure_text_and_source_refs() -> None:
    """Nested rows and cells remain explicit and traceable in rendered order."""
    record = _record(
        "table",
        {
            "images": [],
            "rows": [
                {
                    "row_index": 0,
                    "source_ref": "table/row/0",
                    "cells": [
                        {
                            "cell_index": 0,
                            "source_ref": "table/row/0/cell/0",
                            "blocks": [
                                {
                                    "kind": "paragraph",
                                    "source_ref": "table/row/0/cell/0/paragraph/0",
                                    "text": "Above",
                                },
                                {
                                    "kind": "table",
                                    "table_index": 0,
                                    "source_ref": "table/row/0/cell/0/table/0",
                                    "rows": [
                                        {
                                            "row_index": 0,
                                            "source_ref": "nested/row/0",
                                            "cells": [
                                                {
                                                    "cell_index": 0,
                                                    "source_ref": "nested/row/0/cell/0",
                                                    "blocks": [
                                                        {
                                                            "kind": "paragraph",
                                                            "source_ref": "nested/paragraph/0",
                                                            "text": "Inner A",
                                                        }
                                                    ],
                                                },
                                                {
                                                    "cell_index": 1,
                                                    "source_ref": "nested/row/0/cell/1",
                                                    "blocks": [
                                                        {
                                                            "kind": "paragraph",
                                                            "source_ref": "nested/paragraph/1",
                                                            "text": "Inner B",
                                                        }
                                                    ],
                                                },
                                            ],
                                        }
                                    ],
                                },
                                {
                                    "kind": "paragraph",
                                    "source_ref": "table/row/0/cell/0/paragraph/1",
                                    "text": "Below",
                                },
                            ],
                        }
                    ],
                }
            ],
        },
    )

    units = render_source_scope(_scope(record))

    assert len(units) == 1
    assert units[0].text == "\n".join(
        (
            "[TABLE 1.0.0]",
            "[ROW 0]",
            "[CELL 0]",
            "Above",
            "[TABLE 0]",
            "[ROW 0]",
            "[CELL 0]",
            "Inner A",
            "[CELL 1]",
            "Inner B",
            "[/TABLE 0]",
            "Below",
            "[/TABLE]",
        )
    )
    assert units[0].source_refs == (
        "source/0",
        "table/row/0",
        "table/row/0/cell/0",
        "table/row/0/cell/0/paragraph/0",
        "table/row/0/cell/0/table/0",
        "nested/row/0",
        "nested/row/0/cell/0",
        "nested/paragraph/0",
        "nested/row/0/cell/1",
        "nested/paragraph/1",
        "table/row/0/cell/0/paragraph/1",
    )


def test_render_spreadsheet_row_preserves_sheet_cell_value_and_formula() -> None:
    """Spreadsheet rendering retains native coordinates and formula provenance."""
    record = _record(
        "spreadsheet_row",
        {
            "images": [],
            "sheet_name": "Theo doi",
            "row_index": 1,
            "cells": [
                {"cell_ref": "A1", "value": "Biên bản", "formula": None},
                {"cell_ref": "C1", "value": "2", "formula": "1+1"},
            ],
        },
    )

    units = render_source_scope(_scope(record))

    assert len(units) == 1
    assert units[0].text == "\n".join(
        (
            "[SPREADSHEET_ROW 1.0.0]",
            "[SHEET Theo doi]",
            "[ROW 1]",
            "[CELL A1]",
            "Biên bản",
            "[CELL C1]",
            "2",
            "[FORMULA]",
            "1+1",
        )
    )
    assert units[0].unit_ids == ("spreadsheet_row-unit",)
    assert units[0].source_refs == ("source/0",)


def test_render_skips_empty_base_content_without_dropping_images() -> None:
    """A media-only position still produces its image unit."""
    record = _record(
        "image",
        {
            "images": [
                {
                    "image_id": "image-unit",
                    "source_ref": "source/0/image/0",
                    "parent_source_ref": "source/0",
                    "description_vi": "Mặt trước giấy phép lái xe.",
                }
            ]
        },
    )

    units = render_source_scope(_scope(record))

    assert len(units) == 1
    assert units[0].kind == "image"
    assert units[0].unit_ids == ("image-unit",)
    assert units[0].text == "Mặt trước giấy phép lái xe."
