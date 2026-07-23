"""Merge native DOCX paragraphs and tables into one ordered source view."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from legal_rag.parser.docx_annotations import DocxBookmark, DocxField, DocxHyperlink
from legal_rag.parser.docx_package import DocxPackage
from legal_rag.parser.paragraphs import (
    ParagraphRun,
    ParagraphUnit,
    extract_package_paragraphs,
)
from legal_rag.parser.tables import TableRow, TableUnit, extract_package_tables

DOCUMENT_SOURCE_SCHEMA_VERSION = "1.2.0"

KIND_PARAGRAPH = "paragraph"
KIND_TABLE = "table"


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
    fields: tuple[DocxField, ...] = ()
    hyperlinks: tuple[DocxHyperlink, ...] = ()
    bookmarks: tuple[DocxBookmark, ...] = ()

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
            fields=unit.fields,
            hyperlinks=unit.hyperlinks,
            bookmarks=unit.bookmarks,
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
            record["fields"] = [field.to_dict() for field in self.fields]
            record["hyperlinks"] = [hyperlink.to_dict() for hyperlink in self.hyperlinks]
            record["bookmarks"] = [bookmark.to_dict() for bookmark in self.bookmarks]
        else:
            record["rows"] = [row.to_dict() for row in self.rows or ()]
        return record


@dataclass(frozen=True, slots=True)
class DocumentSource:
    """Ordered native units extracted from one document."""

    source_name: str
    source_sha256: str
    units: tuple[DocumentSourceUnit, ...]


def extract_document_source(input_path: Path) -> DocumentSource:
    """Merge native DOCX paragraphs and tables into one native-order source view."""
    with DocxPackage.open(input_path) as package:
        return extract_package_document_source(package)


def extract_package_document_source(package: DocxPackage) -> DocumentSource:
    """Merge paragraphs and tables from one already loaded DOCX package."""
    paragraphs = extract_package_paragraphs(package)
    tables = extract_package_tables(package)
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
    )
