"""Build bounded structural document context for embedded images."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from legal_rag.parser.document_source import KIND_PARAGRAPH, DocumentSource, DocumentSourceUnit
from legal_rag.parser.images import DocxImage
from legal_rag.parser.tables import TableCellTable, TableRow


class ImageContextRelation(StrEnum):
    """Structural relationship between an image and one context entry."""

    SAME_PARAGRAPH = "SameParagraph"
    SAME_TABLE_ROW = "SameTableRow"
    BEFORE = "Before"
    AFTER = "After"


@dataclass(frozen=True, slots=True)
class ImageContextEntry:
    """One bounded structural context entry supplied to the caption model."""

    source_ref: str
    relation: ImageContextRelation
    distance: int
    text: str


@dataclass(frozen=True, slots=True)
class ImageContext:
    """Priority-ordered structural context entries for one image occurrence."""

    entries: tuple[ImageContextEntry, ...]

    @property
    def source_refs(self) -> tuple[str, ...]:
        """Return the source references disclosed to the model."""
        return tuple(entry.source_ref for entry in self.entries)


def build_image_context(
    image: DocxImage,
    source: DocumentSource,
    *,
    max_characters: int,
    surrounding_units: int,
) -> ImageContext:
    """Select direct and surrounding context using native document structure."""
    unit_index = next(
        (index for index, candidate in enumerate(source.units) if candidate.body_index == image.body_index),
        None,
    )
    primary_entries: tuple[ImageContextEntry, ...] = ()
    if unit_index is not None:
        unit = source.units[unit_index]
        primary_entries = _primary_entries(image, unit)

    surrounding_entries = _surrounding_entries(
        source.units,
        image.body_index,
        unit_index=unit_index,
        limit=surrounding_units,
    )
    return ImageContext(entries=_bounded_entries((*primary_entries, *surrounding_entries), max_characters))


def _primary_entries(image: DocxImage, unit: DocumentSourceUnit) -> tuple[ImageContextEntry, ...]:
    """Return context from the image's paragraph or innermost table row."""
    if unit.kind == KIND_PARAGRAPH:
        if unit.source_ref != image.parent_source_ref or not (unit.text or "").strip():
            return ()
        return (
            ImageContextEntry(
                source_ref=unit.source_ref,
                relation=ImageContextRelation.SAME_PARAGRAPH,
                distance=0,
                text=unit.text or "",
            ),
        )

    row = _find_containing_row(unit.rows or (), image.parent_source_ref)
    return tuple(
        ImageContextEntry(
            source_ref=cell.source_ref,
            relation=ImageContextRelation.SAME_TABLE_ROW,
            distance=0,
            text=cell.text,
        )
        for cell in row.cells
        if cell.text.strip()
    )


def _surrounding_entries(
    units: tuple[DocumentSourceUnit, ...],
    image_body_index: int,
    *,
    unit_index: int | None,
    limit: int,
) -> tuple[ImageContextEntry, ...]:
    """Return nearest preceding and following units in alternating proximity order."""
    insertion_index = (
        unit_index
        if unit_index is not None
        else next(
            (index for index, candidate in enumerate(units) if candidate.body_index > image_body_index),
            len(units),
        )
    )
    before_units = tuple(reversed(units[max(0, insertion_index - limit) : insertion_index]))
    after_start = insertion_index + 1 if unit_index is not None else insertion_index
    after_units = units[after_start : after_start + limit]

    entries: list[ImageContextEntry] = []
    for offset in range(limit):
        distance = offset + 1
        if offset < len(before_units):
            entry = _unit_entry(before_units[offset], ImageContextRelation.BEFORE, distance)
            if entry is not None:
                entries.append(entry)
        if offset < len(after_units):
            entry = _unit_entry(after_units[offset], ImageContextRelation.AFTER, distance)
            if entry is not None:
                entries.append(entry)
    return tuple(entries)


def _find_containing_row(rows: tuple[TableRow, ...], paragraph_ref: str) -> TableRow | None:
    """Find the innermost native row containing an image paragraph reference."""
    for row in rows:
        for cell in row.cells:
            paragraph_prefix = f"{cell.source_ref}/paragraph/"
            if paragraph_ref.startswith(paragraph_prefix):
                return row
            for block in cell.blocks:
                if isinstance(block, TableCellTable):
                    nested_row = _find_containing_row(block.rows, paragraph_ref)
                    if nested_row is not None:
                        return nested_row
    return None


def _unit_entry(
    unit: DocumentSourceUnit,
    relation: ImageContextRelation,
    distance: int,
) -> ImageContextEntry | None:
    """Represent one surrounding source unit without semantic classification."""
    if unit.kind == KIND_PARAGRAPH:
        text = unit.text or ""
    else:
        text = "\n".join(cell.text for row in unit.rows or () for cell in row.cells if cell.text.strip())
    if not text.strip():
        return None
    return ImageContextEntry(source_ref=unit.source_ref, relation=relation, distance=distance, text=text)


def _bounded_entries(
    entries: tuple[ImageContextEntry, ...],
    max_characters: int,
) -> tuple[ImageContextEntry, ...]:
    """Bound context deterministically while preserving structural priority."""
    bounded: list[ImageContextEntry] = []
    remaining = max_characters
    for entry in entries:
        if remaining == 0:
            break
        text = entry.text[:remaining]
        if text:
            bounded.append(ImageContextEntry(entry.source_ref, entry.relation, entry.distance, text))
            remaining -= len(text)
    return tuple(bounded)
