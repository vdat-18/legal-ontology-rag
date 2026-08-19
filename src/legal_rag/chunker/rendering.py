"""Render parser records into stable searchable units without semantic inference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from legal_rag.chunker.models import NormalizedUnit, SourceRecord, SourceScope, StructuralContextEntry, StructuralLevel

TABLE_RENDERING_VERSION = "1.0.0"
SPREADSHEET_RENDERING_VERSION = "1.0.0"
STRUCTURAL_CONTEXT_RENDERING_VERSION = "1.0.0"

_STRUCTURAL_BOOKMARK_PREFIXES: tuple[tuple[str, StructuralLevel], ...] = (
    ("chuong_", "chapter"),
    ("muc_", "section"),
    ("dieu_", "article"),
)


@dataclass(frozen=True, slots=True)
class RenderedContent:
    """Rendered searchable text and the source references it contains."""

    text: str
    source_refs: tuple[str, ...]


@dataclass(slots=True)
class _StructuralContextState:
    """Track active legal headings from parser-provided OOXML bookmarks."""

    chapter: StructuralContextEntry | None = None
    chapter_key: str | None = None
    section: StructuralContextEntry | None = None
    section_key: str | None = None
    article: StructuralContextEntry | None = None
    article_key: str | None = None

    def update(self, record: SourceRecord) -> None:
        """Apply a structural heading carried by one source record."""
        marker = _structural_marker(record)
        if marker is None:
            return
        level, key, is_title = marker
        if is_title:
            self._append_title(level, key, record)
            return

        entry = StructuralContextEntry(
            level=level, text=record.payload["text"].strip(), source_refs=(record.source_ref,)
        )
        if level == "chapter":
            self.chapter = entry
            self.chapter_key = key
            self.section = None
            self.section_key = None
            self.article = None
            self.article_key = None
        elif level == "section":
            self.section = entry
            self.section_key = key
            self.article = None
            self.article_key = None
        else:
            self.article = entry
            self.article_key = key

    def entries(self) -> tuple[StructuralContextEntry, ...]:
        """Return active headings in document hierarchy order."""
        return tuple(entry for entry in (self.chapter, self.section, self.article) if entry is not None)

    def _append_title(self, level: StructuralLevel, key: str, record: SourceRecord) -> None:
        """Append a separately bookmarked heading title to its active marker."""
        attribute = level
        key_attribute = f"{level}_key"
        entry = getattr(self, attribute)
        if entry is None or getattr(self, key_attribute) != key:
            return
        setattr(
            self,
            attribute,
            StructuralContextEntry(
                level=level,
                text=f"{entry.text}\n{record.payload['text'].strip()}",
                source_refs=(*entry.source_refs, record.source_ref),
            ),
        )


def render_source_scope(scope: SourceScope) -> tuple[NormalizedUnit, ...]:
    """Render one ordered source scope into searchable units."""
    units: list[NormalizedUnit] = []
    structural_context = _StructuralContextState()
    for record in scope.records:
        structural_context.update(record)
        context_entries = structural_context.entries()
        if record.kind == "paragraph":
            text = record.payload["text"]
            if text.strip():
                units.append(
                    NormalizedUnit(
                        kind="paragraph",
                        sequence_index=record.order_index,
                        unit_ids=(record.unit_id,),
                        source_refs=(record.source_ref,),
                        text=text,
                        source_payload=record.payload,
                        structural_context=context_entries,
                    )
                )
        elif record.kind == "table":
            table = render_table(record.source_ref, record.payload["rows"])
            if table is not None:
                units.append(
                    NormalizedUnit(
                        kind="table",
                        sequence_index=record.order_index,
                        unit_ids=(record.unit_id,),
                        source_refs=table.source_refs,
                        text=table.text,
                        source_payload=record.payload,
                        structural_context=context_entries,
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
                        kind="spreadsheet_row",
                        sequence_index=record.order_index,
                        unit_ids=(record.unit_id,),
                        source_refs=spreadsheet_row.source_refs,
                        text=spreadsheet_row.text,
                        source_payload=record.payload,
                        structural_context=context_entries,
                    )
                )
        elif record.kind != "image":
            msg = f"Unsupported normalized source kind: {record.kind}"
            raise ValueError(msg)

        units.extend(_render_images(record, context_entries))
    return tuple(units)


def render_structural_context(entries: Sequence[StructuralContextEntry]) -> str:
    """Render source-grounded heading ancestry separately from evidence text."""
    if not entries:
        return ""
    lines = (entry.text for entry in entries)
    return "\n".join((f"[LEGAL_STRUCTURE {STRUCTURAL_CONTEXT_RENDERING_VERSION}]", *lines, "[/LEGAL_STRUCTURE]"))


def render_structural_context_variants(
    contexts: Sequence[tuple[StructuralContextEntry, ...]],
) -> str:
    """Render every distinct ancestry when one chunk crosses a structural boundary."""
    unique_contexts = tuple(dict.fromkeys(context for context in contexts if context))
    article_contexts = tuple(
        context for context in unique_contexts if any(entry.level == "article" for entry in context)
    )
    selected_contexts = article_contexts or unique_contexts
    if len(selected_contexts) < 2:
        return render_structural_context(selected_contexts[0]) if selected_contexts else ""

    variants = []
    for index, context in enumerate(selected_contexts, start=1):
        lines = (entry.text for entry in context)
        variants.append("\n".join((f"[VARIANT {index}]", *lines, "[/VARIANT]")))
    return "\n".join(
        (
            f"[LEGAL_STRUCTURE_VARIANTS {STRUCTURAL_CONTEXT_RENDERING_VERSION}]",
            *variants,
            "[/LEGAL_STRUCTURE_VARIANTS]",
        )
    )


def _structural_marker(record: SourceRecord) -> tuple[StructuralLevel, str, bool] | None:
    """Read one full-paragraph structural marker from OOXML bookmark metadata."""
    if record.kind != "paragraph":
        return None
    text = record.payload["text"]
    for bookmark in record.payload.get("bookmarks", ()):
        if text[: bookmark["start_offset"]].strip() or text[bookmark["end_offset"] :].strip():
            continue
        name = bookmark["name"]
        for prefix, level in _STRUCTURAL_BOOKMARK_PREFIXES:
            if name.startswith(prefix):
                is_title = name.endswith("_name")
                return level, name.removesuffix("_name"), is_title
    return None


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


def _render_images(
    record: SourceRecord, structural_context: tuple[StructuralContextEntry, ...]
) -> tuple[NormalizedUnit, ...]:
    """Render every captioned image occurrence attached to one parser record."""
    units: list[NormalizedUnit] = []
    for image in record.payload.get("images", ()):
        description = image["description_vi"]
        if description.strip():
            units.append(
                NormalizedUnit(
                    kind="image",
                    sequence_index=record.order_index,
                    unit_ids=tuple(dict.fromkeys((record.unit_id, image["image_id"]))),
                    source_refs=tuple(dict.fromkeys((image["source_ref"], image["parent_source_ref"]))),
                    text=description,
                    source_payload=image,
                    structural_context=structural_context,
                )
            )
    return tuple(units)
