"""Build bounded structural document context for embedded images."""

from __future__ import annotations

from dataclasses import dataclass

from legal_rag.document_source import KIND_PARAGRAPH, DocumentSource, DocumentSourceUnit
from legal_rag.docx_images import DocxImage
from legal_rag.docx_tables import TableCellTable, TableRow


class ImageContextError(ValueError):
    """Raised when image and document provenance cannot be joined safely."""

    @classmethod
    def source_mismatch(cls) -> ImageContextError:
        """Build an error for image and source views of different DOCX bytes."""
        return cls("Image extraction and document source refer to different DOCX bytes")

    @classmethod
    def missing_table_row(cls, source_ref: str) -> ImageContextError:
        """Build an error when a table image cannot be placed in its native row."""
        return cls(f"Image occurrence cannot be placed in a table row: {source_ref}")

    @classmethod
    def invalid_limit(cls) -> ImageContextError:
        """Build an error for a non-positive context character limit."""
        return cls("Image context character limit must be positive")


@dataclass(frozen=True, slots=True)
class ImageContextEntry:
    """One source reference and its bounded text supplied to the caption model."""

    source_ref: str
    text: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serializable context entry."""
        return {"source_ref": self.source_ref, "text": self.text}


@dataclass(frozen=True, slots=True)
class ImageContext:
    """Ordered structural context entries for one image occurrence."""

    entries: tuple[ImageContextEntry, ...]

    @property
    def source_refs(self) -> tuple[str, ...]:
        """Return the source references disclosed to the model."""
        return tuple(entry.source_ref for entry in self.entries)

    def to_payload(self) -> dict[str, list[dict[str, str]]]:
        """Return the structured context payload sent with the image."""
        return {"structural_context": [entry.to_dict() for entry in self.entries]}


def build_image_context(
    image: DocxImage,
    source: DocumentSource,
    *,
    max_characters: int,
) -> ImageContext:
    """Select same-row or adjacent-unit context using native document structure."""
    if max_characters <= 0:
        raise ImageContextError.invalid_limit()
    if image.source_sha256 != source.source_sha256:
        raise ImageContextError.source_mismatch()

    unit = next((candidate for candidate in source.units if candidate.body_index == image.body_index), None)
    if unit is not None and unit.kind != KIND_PARAGRAPH:
        row = _find_containing_row(unit.rows or (), image.parent_source_ref)
        if row is None:
            raise ImageContextError.missing_table_row(image.source_ref)
        candidates = tuple(
            ImageContextEntry(source_ref=cell.source_ref, text=cell.text) for cell in row.cells if cell.text.strip()
        )
        return ImageContext(entries=_bounded_entries(candidates, max_characters))

    if unit is not None and unit.kind == KIND_PARAGRAPH and unit.source_ref == image.parent_source_ref:
        text = unit.text or ""
        candidates = (ImageContextEntry(source_ref=unit.source_ref, text=text),) if text.strip() else ()
        return ImageContext(entries=_bounded_entries(candidates, max_characters))

    insertion_index = next(
        (index for index, candidate in enumerate(source.units) if candidate.body_index > image.body_index),
        len(source.units),
    )
    adjacent_units = source.units[max(0, insertion_index - 1) : insertion_index + 1]
    candidates = tuple(entry for candidate in adjacent_units if (entry := _unit_entry(candidate)) is not None)
    return ImageContext(entries=_bounded_entries(candidates, max_characters))


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


def _unit_entry(unit: DocumentSourceUnit) -> ImageContextEntry | None:
    """Represent one adjacent source unit without semantic classification."""
    if unit.kind == KIND_PARAGRAPH:
        text = unit.text or ""
    else:
        text = "\n".join(cell.text for row in unit.rows or () for cell in row.cells if cell.text.strip())
    if not text.strip():
        return None
    return ImageContextEntry(source_ref=unit.source_ref, text=text)


def _bounded_entries(
    entries: tuple[ImageContextEntry, ...],
    max_characters: int,
) -> tuple[ImageContextEntry, ...]:
    """Bound context deterministically while preserving entry and source order."""
    bounded: list[ImageContextEntry] = []
    remaining = max_characters
    for entry in entries:
        if remaining == 0:
            break
        text = entry.text[:remaining]
        if text:
            bounded.append(ImageContextEntry(source_ref=entry.source_ref, text=text))
            remaining -= len(text)
    return tuple(bounded)
