"""Merge native DOCX paragraphs and tables into one ordered source view."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from legal_rag.parser.paragraphs import ParagraphRun, ParagraphUnit, extract_docx_paragraphs
from legal_rag.parser.tables import TableRow, TableUnit, extract_docx_tables

DOCUMENT_SOURCE_SCHEMA_VERSION = "1.1.0"
DOCUMENT_SOURCE_MANIFEST_SCHEMA_VERSION = "1.0.0"

KIND_PARAGRAPH = "paragraph"
KIND_TABLE = "table"


class DocumentSourceValidationError(ValueError):
    """Raised when merged document source units break their persisted contract."""

    @classmethod
    def source_mismatch(cls) -> DocumentSourceValidationError:
        """Build an error for paragraph and table views of different source bytes."""
        return cls("Paragraph and table extractions read different source bytes")

    @classmethod
    def duplicate_unit_ids(cls) -> DocumentSourceValidationError:
        """Build an error for repeated document-source unit IDs."""
        return cls("Document source units contain duplicate unit IDs")

    @classmethod
    def duplicate_source_refs(cls) -> DocumentSourceValidationError:
        """Build an error for repeated OOXML source references."""
        return cls("Document source units contain duplicate source references")

    @classmethod
    def invalid_order(cls) -> DocumentSourceValidationError:
        """Build an error for units outside strictly increasing document order."""
        return cls("Document source units are not in strictly increasing document order")


@dataclass(frozen=True, slots=True)
class DocumentSourceUnit:
    """One native paragraph or table placed in native document order."""

    schema_version: str
    unit_id: str
    source_name: str
    source_sha256: str
    sequence_index: int
    body_index: int
    source_ref: str
    kind: str
    payload_sha256: str
    text: str | None
    runs: tuple[ParagraphRun, ...] | None
    rows: tuple[TableRow, ...] | None

    @classmethod
    def from_paragraph(cls, unit: ParagraphUnit, *, sequence_index: int) -> DocumentSourceUnit:
        """Wrap one native paragraph unit at its merged document position."""
        return cls(
            schema_version=DOCUMENT_SOURCE_SCHEMA_VERSION,
            unit_id=unit.unit_id,
            source_name=unit.source_name,
            source_sha256=unit.source_sha256,
            sequence_index=sequence_index,
            body_index=unit.body_index,
            source_ref=unit.source_ref,
            kind=KIND_PARAGRAPH,
            payload_sha256=unit.payload_sha256,
            text=unit.text,
            runs=unit.runs,
            rows=None,
        )

    @classmethod
    def from_table(cls, unit: TableUnit, *, sequence_index: int) -> DocumentSourceUnit:
        """Wrap one native table unit at its merged document position."""
        return cls(
            schema_version=DOCUMENT_SOURCE_SCHEMA_VERSION,
            unit_id=unit.unit_id,
            source_name=unit.source_name,
            source_sha256=unit.source_sha256,
            sequence_index=sequence_index,
            body_index=unit.body_index,
            source_ref=unit.source_ref,
            kind=KIND_TABLE,
            payload_sha256=unit.payload_sha256,
            text=None,
            runs=None,
            rows=unit.rows,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable document-source record."""
        record: dict[str, Any] = {
            "schema_version": self.schema_version,
            "unit_id": self.unit_id,
            "source_name": self.source_name,
            "source_sha256": self.source_sha256,
            "sequence_index": self.sequence_index,
            "body_index": self.body_index,
            "source_ref": self.source_ref,
            "kind": self.kind,
            "payload_sha256": self.payload_sha256,
        }
        if self.kind == KIND_PARAGRAPH:
            record["text"] = self.text
            record["runs"] = [run.to_dict() for run in self.runs or ()]
        else:
            record["rows"] = [row.to_dict() for row in self.rows or ()]
        return record


@dataclass(frozen=True, slots=True)
class DocumentSource:
    """Ordered native units plus an audit trail of skipped empty paragraphs."""

    source_name: str
    source_sha256: str
    units: tuple[DocumentSourceUnit, ...]
    paragraph_count: int
    table_count: int
    cell_count: int
    skipped_empty_paragraph_refs: tuple[str, ...]
    skipped_empty_table_paragraph_refs: tuple[str, ...]

    def manifest(self) -> dict[str, Any]:
        """Return the document-source audit manifest."""
        return {
            "schema_version": DOCUMENT_SOURCE_MANIFEST_SCHEMA_VERSION,
            "source_name": self.source_name,
            "source_sha256": self.source_sha256,
            "unit_count": len(self.units),
            "paragraph_count": self.paragraph_count,
            "table_count": self.table_count,
            "cell_count": self.cell_count,
            "skipped_empty_paragraph_count": len(self.skipped_empty_paragraph_refs),
            "skipped_empty_paragraph_refs": list(self.skipped_empty_paragraph_refs),
            "skipped_empty_table_paragraph_count": len(self.skipped_empty_table_paragraph_refs),
            "skipped_empty_table_paragraph_refs": list(self.skipped_empty_table_paragraph_refs),
        }


def extract_document_source(input_path: Path) -> DocumentSource:
    """Merge native DOCX paragraphs and tables into one native-order source view."""
    paragraphs = extract_docx_paragraphs(input_path)
    tables = extract_docx_tables(input_path)
    if paragraphs.source_sha256 != tables.source_sha256:
        raise DocumentSourceValidationError.source_mismatch()

    merged: list[ParagraphUnit | TableUnit] = [*paragraphs.units, *tables.units]
    merged.sort(key=lambda unit: unit.body_index)

    units: list[DocumentSourceUnit] = []
    for sequence_index, unit in enumerate(merged):
        if isinstance(unit, ParagraphUnit):
            units.append(DocumentSourceUnit.from_paragraph(unit, sequence_index=sequence_index))
        else:
            units.append(DocumentSourceUnit.from_table(unit, sequence_index=sequence_index))

    return DocumentSource(
        source_name=paragraphs.source_name,
        source_sha256=paragraphs.source_sha256,
        units=tuple(units),
        paragraph_count=len(paragraphs.units),
        table_count=len(tables.units),
        cell_count=tables.cell_count,
        skipped_empty_paragraph_refs=paragraphs.skipped_empty_paragraph_refs,
        skipped_empty_table_paragraph_refs=tables.skipped_empty_paragraph_refs,
    )


def write_document_source(output_path: Path, source: DocumentSource, *, manifest_path: Path | None = None) -> None:
    """Atomically write the merged source view and, when requested, its audit manifest."""
    _validate_units(source.units)
    records = (unit.to_dict() for unit in source.units)
    _write_jsonl(output_path, records)
    if manifest_path is not None:
        _write_json(manifest_path, source.manifest())


def _validate_units(units: tuple[DocumentSourceUnit, ...]) -> None:
    """Validate identifiers, references, and retained native document order."""
    unit_ids = [unit.unit_id for unit in units]
    if len(unit_ids) != len(set(unit_ids)):
        raise DocumentSourceValidationError.duplicate_unit_ids()
    source_refs = [unit.source_ref for unit in units]
    if len(source_refs) != len(set(source_refs)):
        raise DocumentSourceValidationError.duplicate_source_refs()
    body_indices = [unit.body_index for unit in units]
    if body_indices != sorted(body_indices) or len(body_indices) != len(set(body_indices)):
        raise DocumentSourceValidationError.invalid_order()
    sequence_indices = [unit.sequence_index for unit in units]
    if sequence_indices != list(range(len(units))):
        raise DocumentSourceValidationError.invalid_order()


def _write_jsonl(output_path: Path, records: Iterable[dict[str, Any]]) -> None:
    """Atomically write JSON-serializable records as UTF-8 JSON Lines."""
    serialized = (
        json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n" for record in records
    )
    _write_text(output_path, serialized)


def _write_json(output_path: Path, value: dict[str, Any]) -> None:
    """Atomically write one UTF-8 JSON document."""
    serialized = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _write_text(output_path, (serialized,))


def _write_text(output_path: Path, parts: Iterable[str]) -> None:
    """Atomically write text parts to one output path."""
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
            temporary_file.writelines(parts)
        temporary_path.replace(output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
