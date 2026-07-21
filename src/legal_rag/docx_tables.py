"""Extract native top-level table structure from DOCX OOXML."""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from defusedxml import ElementTree as SafeElementTree

from legal_rag.docx_paragraphs import ParagraphRun, extract_paragraph_runs

TABLE_UNIT_SCHEMA_VERSION = "1.0.0"
TABLE_MANIFEST_SCHEMA_VERSION = "1.0.0"

_DOCUMENT_PART = "word/document.xml"
_WORDPROCESSINGML_NAMESPACE = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_WORDPROCESSINGML_PREFIX = f"{{{_WORDPROCESSINGML_NAMESPACE}}}"
_BODY_TAG = f"{_WORDPROCESSINGML_PREFIX}body"
_TABLE_TAG = f"{_WORDPROCESSINGML_PREFIX}tbl"
_ROW_TAG = f"{_WORDPROCESSINGML_PREFIX}tr"
_CELL_TAG = f"{_WORDPROCESSINGML_PREFIX}tc"
_CELL_PROPERTIES_TAG = f"{_WORDPROCESSINGML_PREFIX}tcPr"
_PARAGRAPH_TAG = f"{_WORDPROCESSINGML_PREFIX}p"
_GRID_SPAN_TAG = f"{_WORDPROCESSINGML_PREFIX}gridSpan"
_VERTICAL_MERGE_TAG = f"{_WORDPROCESSINGML_PREFIX}vMerge"
_VALUE_ATTRIBUTE = f"{_WORDPROCESSINGML_PREFIX}val"
_DEFAULT_GRID_SPAN = 1
_VERTICAL_MERGE_CONTINUE = "continue"


class DocxTableExtractionError(ValueError):
    """Raised when native DOCX tables cannot be extracted safely."""

    @classmethod
    def invalid_archive(cls, path: Path) -> DocxTableExtractionError:
        """Build an error for a file that is not a valid DOCX archive."""
        return cls(f"Source document is not a valid DOCX archive: {path}")

    @classmethod
    def missing_document_part(cls, path: Path) -> DocxTableExtractionError:
        """Build an error for a DOCX without its main document part."""
        return cls(f"DOCX is missing {_DOCUMENT_PART}: {path}")

    @classmethod
    def malformed_document_part(cls, path: Path) -> DocxTableExtractionError:
        """Build an error for malformed main-document XML."""
        return cls(f"Malformed DOCX main document XML: {path}")

    @classmethod
    def missing_body(cls, path: Path) -> DocxTableExtractionError:
        """Build an error for a DOCX without a document body."""
        return cls(f"DOCX main document has no body: {path}")

    @classmethod
    def nested_table(cls, source_ref: str) -> DocxTableExtractionError:
        """Build an error for a nested table that needs an explicit schema."""
        return cls(f"Nested DOCX table is not supported: {source_ref}")

    @classmethod
    def invalid_grid_span(cls, source_ref: str) -> DocxTableExtractionError:
        """Build an error for a malformed OOXML grid span."""
        return cls(f"DOCX table cell has an invalid grid span: {source_ref}")


class TableUnitValidationError(ValueError):
    """Raised when table units cannot satisfy their persisted contract."""

    @classmethod
    def duplicate_unit_ids(cls) -> TableUnitValidationError:
        """Build an error for repeated table-unit IDs."""
        return cls("Table units contain duplicate unit IDs")

    @classmethod
    def duplicate_source_refs(cls) -> TableUnitValidationError:
        """Build an error for repeated OOXML table references."""
        return cls("Table units contain duplicate source references")

    @classmethod
    def invalid_body_order(cls) -> TableUnitValidationError:
        """Build an error for table units outside DOCX body order."""
        return cls("Table units are not in strictly increasing body order")


@dataclass(frozen=True, slots=True)
class TableCellParagraph:
    """One content-bearing paragraph within a DOCX table cell."""

    paragraph_index: int
    source_ref: str
    text: str
    runs: tuple[ParagraphRun, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable table-cell paragraph."""
        return {
            "paragraph_index": self.paragraph_index,
            "source_ref": self.source_ref,
            "text": self.text,
            "runs": [run.to_dict() for run in self.runs],
        }


@dataclass(frozen=True, slots=True)
class TableCell:
    """One native DOCX table cell with merge metadata and paragraphs."""

    cell_index: int
    source_ref: str
    grid_span: int
    vertical_merge: str | None
    text: str
    paragraphs: tuple[TableCellParagraph, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable table cell."""
        return {
            "cell_index": self.cell_index,
            "source_ref": self.source_ref,
            "grid_span": self.grid_span,
            "vertical_merge": self.vertical_merge,
            "text": self.text,
            "paragraphs": [paragraph.to_dict() for paragraph in self.paragraphs],
        }


@dataclass(frozen=True, slots=True)
class TableRow:
    """One native DOCX table row in source order."""

    row_index: int
    source_ref: str
    cells: tuple[TableCell, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable table row."""
        return {
            "row_index": self.row_index,
            "source_ref": self.source_ref,
            "cells": [cell.to_dict() for cell in self.cells],
        }


@dataclass(frozen=True, slots=True)
class TableUnit:
    """One top-level DOCX table with native row and cell boundaries."""

    schema_version: str
    unit_id: str
    source_name: str
    source_sha256: str
    table_index: int
    body_index: int
    source_ref: str
    rows: tuple[TableRow, ...]
    payload_sha256: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable table unit."""
        return {
            "schema_version": self.schema_version,
            "unit_id": self.unit_id,
            "source_name": self.source_name,
            "source_sha256": self.source_sha256,
            "table_index": self.table_index,
            "body_index": self.body_index,
            "source_ref": self.source_ref,
            "rows": [row.to_dict() for row in self.rows],
            "payload_sha256": self.payload_sha256,
        }


@dataclass(frozen=True, slots=True)
class TableExtraction:
    """Table units plus an audit trail of skipped empty cell paragraphs."""

    source_name: str
    source_sha256: str
    units: tuple[TableUnit, ...]
    cell_count: int
    raw_cell_paragraph_count: int
    skipped_empty_paragraph_refs: tuple[str, ...]

    def manifest(self) -> dict[str, Any]:
        """Return the table-extraction audit manifest."""
        return {
            "schema_version": TABLE_MANIFEST_SCHEMA_VERSION,
            "source_name": self.source_name,
            "source_sha256": self.source_sha256,
            "table_unit_count": len(self.units),
            "cell_count": self.cell_count,
            "raw_cell_paragraph_count": self.raw_cell_paragraph_count,
            "cell_paragraph_count": self.raw_cell_paragraph_count - len(self.skipped_empty_paragraph_refs),
            "skipped_empty_paragraph_count": len(self.skipped_empty_paragraph_refs),
            "skipped_empty_paragraph_refs": list(self.skipped_empty_paragraph_refs),
        }


def extract_docx_tables(input_path: Path) -> TableExtraction:
    """Extract top-level tables directly from native DOCX OOXML."""
    source_bytes = input_path.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    document_xml = _read_document_part(input_path, source_bytes)
    try:
        document = SafeElementTree.fromstring(document_xml)
    except ElementTree.ParseError as error:
        raise DocxTableExtractionError.malformed_document_part(input_path) from error

    body = document.find(_BODY_TAG)
    if body is None:
        raise DocxTableExtractionError.missing_body(input_path)

    units: list[TableUnit] = []
    skipped_refs: list[str] = []
    cell_count = 0
    paragraph_count = 0
    for body_index, element in enumerate(body):
        if element.tag != _TABLE_TAG:
            continue

        table_index = len(units)
        source_ref = f"{_DOCUMENT_PART}#/body/{body_index}"
        rows, table_cell_count, table_paragraph_count = _extract_rows(element, source_ref, skipped_refs)
        cell_count += table_cell_count
        paragraph_count += table_paragraph_count
        payload = {
            "body_index": body_index,
            "rows": [row.to_dict() for row in rows],
            "source_ref": source_ref,
            "table_index": table_index,
        }
        payload_sha256 = _sha256_json(payload)
        unit_id = _sha256_json(
            {
                "payload_sha256": payload_sha256,
                "source_ref": source_ref,
                "source_sha256": source_sha256,
            }
        )
        units.append(
            TableUnit(
                schema_version=TABLE_UNIT_SCHEMA_VERSION,
                unit_id=unit_id,
                source_name=input_path.name,
                source_sha256=source_sha256,
                table_index=table_index,
                body_index=body_index,
                source_ref=source_ref,
                rows=rows,
                payload_sha256=payload_sha256,
            )
        )

    return TableExtraction(
        source_name=input_path.name,
        source_sha256=source_sha256,
        units=tuple(units),
        cell_count=cell_count,
        raw_cell_paragraph_count=paragraph_count,
        skipped_empty_paragraph_refs=tuple(skipped_refs),
    )


def write_docx_tables(output_path: Path, manifest_path: Path, extraction: TableExtraction) -> None:
    """Atomically write table units and their audit manifest."""
    _validate_units(extraction.units)
    records = (unit.to_dict() for unit in extraction.units)
    _write_jsonl(output_path, records)
    _write_json(manifest_path, extraction.manifest())


def _extract_rows(
    table: ElementTree.Element,
    table_ref: str,
    skipped_refs: list[str],
) -> tuple[tuple[TableRow, ...], int, int]:
    """Extract rows and return their cell and paragraph counts."""
    nested_tables = [element for element in table.iter(_TABLE_TAG) if element is not table]
    if nested_tables:
        raise DocxTableExtractionError.nested_table(table_ref)

    rows: list[TableRow] = []
    cell_count = 0
    paragraph_count = 0
    for row_index, row in enumerate(table.iter(_ROW_TAG)):
        row_ref = f"{table_ref}/row/{row_index}"
        cells: list[TableCell] = []
        for cell_index, cell in enumerate(row.iter(_CELL_TAG)):
            cell_ref = f"{row_ref}/cell/{cell_index}"
            paragraphs, raw_count = _extract_cell_paragraphs(cell, cell_ref, skipped_refs)
            paragraph_count += raw_count
            cell_count += 1
            grid_span, vertical_merge = _cell_merge_metadata(cell, cell_ref)
            cells.append(
                TableCell(
                    cell_index=cell_index,
                    source_ref=cell_ref,
                    grid_span=grid_span,
                    vertical_merge=vertical_merge,
                    text="\n".join(paragraph.text for paragraph in paragraphs),
                    paragraphs=paragraphs,
                )
            )
        rows.append(TableRow(row_index=row_index, source_ref=row_ref, cells=tuple(cells)))
    return tuple(rows), cell_count, paragraph_count


def _extract_cell_paragraphs(
    cell: ElementTree.Element,
    cell_ref: str,
    skipped_refs: list[str],
) -> tuple[tuple[TableCellParagraph, ...], int]:
    """Extract content-bearing paragraphs from one table cell."""
    paragraphs: list[TableCellParagraph] = []
    raw_count = 0
    for paragraph_index, paragraph in enumerate(cell.iter(_PARAGRAPH_TAG)):
        raw_count += 1
        paragraph_ref = f"{cell_ref}/paragraph/{paragraph_index}"
        runs = tuple(extract_paragraph_runs(paragraph))
        text = "".join(run.text for run in runs)
        if not text.strip():
            skipped_refs.append(paragraph_ref)
            continue
        paragraphs.append(
            TableCellParagraph(
                paragraph_index=paragraph_index,
                source_ref=paragraph_ref,
                text=text,
                runs=runs,
            )
        )
    return tuple(paragraphs), raw_count


def _cell_merge_metadata(cell: ElementTree.Element, cell_ref: str) -> tuple[int, str | None]:
    """Return native horizontal and vertical merge metadata."""
    properties = cell.find(_CELL_PROPERTIES_TAG)
    if properties is None:
        return _DEFAULT_GRID_SPAN, None

    grid_span = _DEFAULT_GRID_SPAN
    grid_span_element = properties.find(_GRID_SPAN_TAG)
    if grid_span_element is not None:
        value = grid_span_element.get(_VALUE_ATTRIBUTE)
        try:
            grid_span = int(value) if value is not None else _DEFAULT_GRID_SPAN
        except ValueError as error:
            raise DocxTableExtractionError.invalid_grid_span(cell_ref) from error
        if grid_span < _DEFAULT_GRID_SPAN:
            raise DocxTableExtractionError.invalid_grid_span(cell_ref)

    vertical_merge = None
    vertical_merge_element = properties.find(_VERTICAL_MERGE_TAG)
    if vertical_merge_element is not None:
        vertical_merge = vertical_merge_element.get(_VALUE_ATTRIBUTE) or _VERTICAL_MERGE_CONTINUE
    return grid_span, vertical_merge


def _read_document_part(input_path: Path, source_bytes: bytes) -> bytes:
    """Read the main OOXML document part from DOCX bytes."""
    try:
        with zipfile.ZipFile(io.BytesIO(source_bytes)) as archive:
            try:
                return archive.read(_DOCUMENT_PART)
            except KeyError as error:
                raise DocxTableExtractionError.missing_document_part(input_path) from error
    except zipfile.BadZipFile as error:
        raise DocxTableExtractionError.invalid_archive(input_path) from error


def _validate_units(units: tuple[TableUnit, ...]) -> None:
    """Validate identifiers, references, and retained DOCX body order."""
    unit_ids = [unit.unit_id for unit in units]
    if len(unit_ids) != len(set(unit_ids)):
        raise TableUnitValidationError.duplicate_unit_ids()
    source_refs = [unit.source_ref for unit in units]
    if len(source_refs) != len(set(source_refs)):
        raise TableUnitValidationError.duplicate_source_refs()
    body_indices = [unit.body_index for unit in units]
    if body_indices != sorted(body_indices) or len(body_indices) != len(set(body_indices)):
        raise TableUnitValidationError.invalid_body_order()


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


def _sha256_json(value: object) -> str:
    """Hash a value using canonical UTF-8 JSON serialization."""
    serialized = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
