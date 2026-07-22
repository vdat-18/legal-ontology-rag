"""Create a content-bearing view without mutating raw source blocks."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from legal_rag.parser.source_blocks import SourceBlock, SourceBlockValidationError, write_source_blocks

NORMALIZATION_MANIFEST_SCHEMA_VERSION = "1.0.0"


class ContentNormalizationError(ValueError):
    """Raised when source blocks cannot form one normalized document view."""

    @classmethod
    def mixed_sources(cls) -> ContentNormalizationError:
        """Build an error for blocks from more than one source document."""
        return cls("Content normalization requires blocks from exactly one source document")

    @classmethod
    def invalid_text(cls, docling_ref: str) -> ContentNormalizationError:
        """Build an error for a non-string Docling text value."""
        return cls(f"Docling text must be a string: {docling_ref}")

    @classmethod
    def no_content(cls) -> ContentNormalizationError:
        """Build an error for a source without content-bearing blocks."""
        return cls("Source blocks contain no content-bearing items")


@dataclass(frozen=True, slots=True)
class ContentBlockSelection:
    """Normalized blocks plus an audit trail of skipped empty text items."""

    source_name: str
    source_sha256: str
    raw_block_count: int
    blocks: tuple[SourceBlock, ...]
    skipped_empty_text_refs: tuple[str, ...]

    def manifest(self) -> dict[str, Any]:
        """Return the normalization audit manifest."""
        return {
            "schema_version": NORMALIZATION_MANIFEST_SCHEMA_VERSION,
            "source_name": self.source_name,
            "source_sha256": self.source_sha256,
            "raw_block_count": self.raw_block_count,
            "normalized_block_count": len(self.blocks),
            "skipped_empty_text_count": len(self.skipped_empty_text_refs),
            "skipped_empty_text_refs": list(self.skipped_empty_text_refs),
        }


def select_content_blocks(blocks: list[SourceBlock]) -> ContentBlockSelection:
    """Exclude only empty Docling text items from a normalized view."""
    if not blocks:
        raise SourceBlockValidationError.empty_blocks()

    source_names = {block.source_name for block in blocks}
    source_hashes = {block.source_sha256 for block in blocks}
    if len(source_names) != 1 or len(source_hashes) != 1:
        raise ContentNormalizationError.mixed_sources()

    selected: list[SourceBlock] = []
    skipped_refs: list[str] = []
    for block in blocks:
        text = block.payload.get("text")
        if text is None:
            selected.append(block)
            continue
        if not isinstance(text, str):
            raise ContentNormalizationError.invalid_text(block.docling_ref)
        if text.strip():
            selected.append(block)
        else:
            skipped_refs.append(block.docling_ref)

    if not selected:
        raise ContentNormalizationError.no_content()

    return ContentBlockSelection(
        source_name=next(iter(source_names)),
        source_sha256=next(iter(source_hashes)),
        raw_block_count=len(blocks),
        blocks=tuple(selected),
        skipped_empty_text_refs=tuple(skipped_refs),
    )


def write_normalized_source_blocks(
    output_path: Path,
    manifest_path: Path,
    selection: ContentBlockSelection,
) -> None:
    """Write normalized blocks and their normalization audit manifest."""
    write_source_blocks(output_path, list(selection.blocks))
    _write_json(manifest_path, selection.manifest())


def _write_json(output_path: Path, value: dict[str, Any]) -> None:
    """Atomically write one UTF-8 JSON document."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(value, temporary_file, ensure_ascii=False, indent=2, sort_keys=True)
            temporary_file.write("\n")
        temporary_path.replace(output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
