"""Convert a source document into deterministic Docling source blocks."""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

from docling.document_converter import DocumentConverter
from docling_core.types.doc import DoclingDocument
from docling_core.types.io import DocumentStream

SOURCE_BLOCK_SCHEMA_VERSION = "1.0.0"


class SourceDocumentNotFoundError(FileNotFoundError):
    """Raised when the requested source document does not exist."""

    def __init__(self, path: Path) -> None:
        """Initialize the error with the missing path."""
        super().__init__(f"Source document does not exist: {path}")


class SourceDocumentTypeError(IsADirectoryError):
    """Raised when the requested source path is not a file."""

    def __init__(self, path: Path) -> None:
        """Initialize the error with the invalid path."""
        super().__init__(f"Source document is not a file: {path}")


class SourceBlockValidationError(ValueError):
    """Raised when Docling output cannot satisfy the source-block contract."""

    @classmethod
    def duplicate_docling_ref(cls, docling_ref: str) -> SourceBlockValidationError:
        """Build an error for a repeated Docling reference."""
        return cls(f"Docling produced a duplicate item reference: {docling_ref}")

    @classmethod
    def empty_document(cls, source_name: str) -> SourceBlockValidationError:
        """Build an error for a document without source items."""
        return cls(f"Docling document contains no source items: {source_name}")

    @classmethod
    def empty_blocks(cls) -> SourceBlockValidationError:
        """Build an error for an empty output collection."""
        return cls("At least one source block is required")

    @classmethod
    def duplicate_block_ids(cls) -> SourceBlockValidationError:
        """Build an error for repeated deterministic block IDs."""
        return cls("Source blocks contain duplicate block IDs")


@dataclass(frozen=True, slots=True)
class SourceBlock:
    """One lossless Docling item with deterministic provenance metadata."""

    schema_version: str
    block_id: str
    source_name: str
    source_sha256: str
    parser_name: str
    parser_version: str
    parser_core_version: str
    docling_ref: str
    docling_item_type: str
    docling_label: str
    depth: int
    payload_sha256: str
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable source-block record."""
        return {
            "schema_version": self.schema_version,
            "block_id": self.block_id,
            "source_name": self.source_name,
            "source_sha256": self.source_sha256,
            "parser_name": self.parser_name,
            "parser_version": self.parser_version,
            "parser_core_version": self.parser_core_version,
            "docling_ref": self.docling_ref,
            "docling_item_type": self.docling_item_type,
            "docling_label": self.docling_label,
            "depth": self.depth,
            "payload_sha256": self.payload_sha256,
            "payload": self.payload,
        }


def extract_source_blocks(input_path: Path) -> list[SourceBlock]:
    """Convert a source file and return its deterministic Docling blocks."""
    if not input_path.exists():
        raise SourceDocumentNotFoundError(input_path)
    if not input_path.is_file():
        raise SourceDocumentTypeError(input_path)

    source_bytes = input_path.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    source = DocumentStream(name=input_path.name, stream=io.BytesIO(source_bytes))
    document = DocumentConverter().convert(source).document
    return build_source_blocks(document, source_name=input_path.name, source_sha256=source_sha256)


def build_source_blocks(
    document: DoclingDocument,
    *,
    source_name: str,
    source_sha256: str,
) -> list[SourceBlock]:
    """Serialize every Docling item without adding legal interpretation."""
    parser_name = "docling"
    parser_version = version(parser_name)
    parser_core_version = version("docling-core")
    blocks: list[SourceBlock] = []
    seen_refs: set[str] = set()

    for item, depth in document.iterate_items(with_groups=True, traverse_pictures=True):
        docling_ref = str(item.self_ref)
        if docling_ref in seen_refs:
            raise SourceBlockValidationError.duplicate_docling_ref(docling_ref)
        seen_refs.add(docling_ref)

        payload = item.model_dump(mode="json", exclude_none=True)
        payload_sha256 = _sha256_json(payload)
        block_id = _sha256_json(
            {
                "docling_ref": docling_ref,
                "parser_core_version": parser_core_version,
                "parser_version": parser_version,
                "payload_sha256": payload_sha256,
                "source_sha256": source_sha256,
            }
        )
        blocks.append(
            SourceBlock(
                schema_version=SOURCE_BLOCK_SCHEMA_VERSION,
                block_id=block_id,
                source_name=source_name,
                source_sha256=source_sha256,
                parser_name=parser_name,
                parser_version=parser_version,
                parser_core_version=parser_core_version,
                docling_ref=docling_ref,
                docling_item_type=type(item).__name__,
                docling_label=str(item.label),
                depth=depth,
                payload_sha256=payload_sha256,
                payload=payload,
            )
        )

    if not blocks:
        raise SourceBlockValidationError.empty_document(source_name)
    return blocks


def write_source_blocks(output_path: Path, blocks: list[SourceBlock]) -> None:
    """Atomically write source blocks as UTF-8 JSON Lines."""
    if not blocks:
        raise SourceBlockValidationError.empty_blocks()
    block_ids = [block.block_id for block in blocks]
    if len(block_ids) != len(set(block_ids)):
        raise SourceBlockValidationError.duplicate_block_ids()

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
            for block in blocks:
                record = json.dumps(block.to_dict(), ensure_ascii=False, separators=(",", ":"), sort_keys=True)
                temporary_file.write(f"{record}\n")
        temporary_path.replace(output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _sha256_json(value: object) -> str:
    """Hash a value using canonical UTF-8 JSON serialization."""
    serialized = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
