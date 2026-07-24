"""Render parser records into stable searchable units without semantic inference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from legal_rag.chunker.models import NormalizedUnit, SourceRecord, SourceScope

TABLE_RENDERING_VERSION = "1.0.0"
SPREADSHEET_RENDERING_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class RenderedContent:
    """Rendered searchable text and the source references it contains."""

    text: str
    source_refs: tuple[str, ...]


def render_source_scope(scope: SourceScope) -> tuple[NormalizedUnit, ...]:
    """Render one ordered source scope into searchable units."""
    units: list[NormalizedUnit] = []
    for record in scope.records:
        if record.kind == "paragraph":
            text = record.payload["text"]
            if text.strip():
                units.append(
                    NormalizedUnit(
                        scope_id=scope.scope_id,
                        kind="paragraph",
                        sequence_index=record.order_index,
                        unit_ids=(record.unit_id,),
                        source_refs=(record.source_ref,),
                        text=text,
                        source_payload=record.payload,
                    )
                )
        elif record.kind == "table":
            table = render_table(record.source_ref, record.payload["rows"])
            if table is not None:
                units.append(
                    NormalizedUnit(
                        scope_id=scope.scope_id,
                        kind="table",
                        sequence_index=record.order_index,
                        unit_ids=(record.unit_id,),
                        source_refs=table.source_refs,
                        text=table.text,
                        source_payload=record.payload,
                    )
                )
        elif record.kind == "spreadsheet_row":
            spreadsheet_row = render_spreadsheet_row(
                record.source_ref,
                record.payload["sheet_name"],
                record.payload["row_index"],
                record.payload["cells"],
            )
            if spreadsheet_row is not None:
                units.append(
                    NormalizedUnit(
                        scope_id=scope.scope_id,
                        kind="spreadsheet_row",
                        sequence_index=record.order_index,
                        unit_ids=(record.unit_id,),
                        source_refs=spreadsheet_row.source_refs,
                        text=spreadsheet_row.text,
                        source_payload=record.payload,
                    )
                )
        elif record.kind != "image":
            msg = f"Unsupported normalized source kind: {record.kind}"
            raise ValueError(msg)

        units.extend(_render_images(scope.scope_id, record))
    return tuple(units)


def render_table(source_ref: str, rows: Sequence[Mapping[str, Any]]) -> RenderedContent | None:
    """Render table rows while retaining native row, cell, and block provenance."""
    lines, source_refs, has_content = _render_table_rows(rows)
    if not has_content:
        return None
    text = "\n".join((f"[TABLE {TABLE_RENDERING_VERSION}]", *lines, "[/TABLE]"))
    return RenderedContent(text, tuple(dict.fromkeys((source_ref, *source_refs))))


def _render_table_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[list[str], list[str], bool]:
    """Render table rows recursively in parser-provided source order."""
    lines: list[str] = []
    source_refs: list[str] = []
    has_content = False
    for row in rows:
        lines.append(f"[ROW {row['row_index']}]")
        source_refs.append(row["source_ref"])
        for cell in row["cells"]:
            lines.append(f"[CELL {cell['cell_index']}]")
            source_refs.append(cell["source_ref"])
            for block in cell["blocks"]:
                if block["kind"] == "paragraph":
                    text = block["text"]
                    if text.strip():
                        lines.append(text)
                        source_refs.append(block["source_ref"])
                        has_content = True
                elif block["kind"] == "table":
                    nested_lines, nested_refs, nested_has_content = _render_table_rows(block["rows"])
                    if nested_has_content:
                        table_index = block["table_index"]
                        lines.extend((f"[TABLE {table_index}]", *nested_lines, f"[/TABLE {table_index}]"))
                        source_refs.extend((block["source_ref"], *nested_refs))
                        has_content = True
                else:
                    msg = f"Unsupported table block kind: {block['kind']}"
                    raise ValueError(msg)
    return lines, source_refs, has_content


def render_spreadsheet_row(
    source_ref: str,
    sheet_name: str,
    row_index: int,
    cells: Sequence[Mapping[str, Any]],
) -> RenderedContent | None:
    """Render one spreadsheet row with cell references, values, and formulas."""
    lines = [
        f"[SPREADSHEET_ROW {SPREADSHEET_RENDERING_VERSION}]",
        f"[SHEET {sheet_name}]",
        f"[ROW {row_index}]",
    ]
    has_content = False
    for cell in cells:
        value = cell["value"]
        formula = cell["formula"]
        if (value is not None and value.strip()) or (formula is not None and formula.strip()):
            lines.append(f"[CELL {cell['cell_ref']}]")
            if value is not None and value.strip():
                lines.append(value)
            if formula is not None and formula.strip():
                lines.extend(("[FORMULA]", formula))
            has_content = True
    return RenderedContent("\n".join(lines), (source_ref,)) if has_content else None


def _render_images(scope_id: str, record: SourceRecord) -> tuple[NormalizedUnit, ...]:
    """Render every captioned image occurrence attached to one parser record."""
    units: list[NormalizedUnit] = []
    for image in record.payload.get("images", ()):
        description = image["description_vi"]
        if description.strip():
            units.append(
                NormalizedUnit(
                    scope_id=scope_id,
                    kind="image",
                    sequence_index=record.order_index,
                    unit_ids=tuple(dict.fromkeys((record.unit_id, image["image_id"]))),
                    source_refs=tuple(dict.fromkeys((image["source_ref"], image["parent_source_ref"]))),
                    text=description,
                    source_payload=image,
                )
            )
    return tuple(units)
