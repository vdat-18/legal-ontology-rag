"""Extract native top-level table structure from DOCX OOXML."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from legal_rag.parser.docx_package import DocxPackage
from legal_rag.parser.hashing import sha256_json
from legal_rag.parser.paragraphs import ParagraphRun, extract_paragraph_runs

TABLE_UNIT_SCHEMA_VERSION = "1.1.0"

CELL_BLOCK_PARAGRAPH_KIND = "paragraph"
CELL_BLOCK_TABLE_KIND = "table"

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
    def missing_body(cls, path: Path) -> DocxTableExtractionError:
        """Build an error for a DOCX without a document body."""
        return cls(f"DOCX main document has no body: {path}")

    @classmethod
    def invalid_grid_span(cls, source_ref: str) -> DocxTableExtractionError:
        """Build an error for a malformed OOXML grid span."""
        return cls(f"DOCX table cell has an invalid grid span: {source_ref}")


@dataclass(frozen=True, slots=True)
class TableCellParagraph:
    """One content-bearing paragraph within a DOCX table cell."""

    paragraph_index: int
    source_ref: str
    text: str
    runs: tuple[ParagraphRun, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable table-cell paragraph block."""
        return {
            "kind": CELL_BLOCK_PARAGRAPH_KIND,
            "paragraph_index": self.paragraph_index,
            "source_ref": self.source_ref,
            "text": self.text,
            "runs": [run.to_dict() for run in self.runs],
        }


@dataclass(frozen=True, slots=True)
class TableCellTable:
    """One nested DOCX table placed inside a table cell in source order."""

    table_index: int
    source_ref: str
    rows: tuple[TableRow, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable nested-table block."""
        return {
            "kind": CELL_BLOCK_TABLE_KIND,
            "table_index": self.table_index,
            "source_ref": self.source_ref,
            "rows": [row.to_dict() for row in self.rows],
        }


@dataclass(frozen=True, slots=True)
class TableCell:
    """One native DOCX table cell with merge metadata and ordered content blocks."""

    cell_index: int
    source_ref: str
    grid_span: int
    vertical_merge: str | None
    text: str
    blocks: tuple[TableCellParagraph | TableCellTable, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable table cell."""
        return {
            "cell_index": self.cell_index,
            "source_ref": self.source_ref,
            "grid_span": self.grid_span,
            "vertical_merge": self.vertical_merge,
            "text": self.text,
            "blocks": [block.to_dict() for block in self.blocks],
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
    """Native table units extracted from one DOCX."""

    source_name: str
    source_sha256: str
    units: tuple[TableUnit, ...]


def extract_docx_tables(input_path: Path) -> TableExtraction:
    """Extract top-level tables directly from native DOCX OOXML."""
    with DocxPackage.open(input_path) as package:
        return extract_package_tables(package)


def extract_package_tables(package: DocxPackage) -> TableExtraction:
    """Extract tables from an already loaded DOCX package."""
    input_path = package.input_path

    body = package.document.find(_BODY_TAG)
    if body is None:
        raise DocxTableExtractionError.missing_body(input_path)

    units: list[TableUnit] = []
    for body_index, element in enumerate(body):
        if element.tag != _TABLE_TAG:
            continue

        table_index = len(units)
        source_ref = f"{_DOCUMENT_PART}#/body/{body_index}"
        rows = _extract_rows(element, source_ref)
        payload = {
            "body_index": body_index,
            "rows": [row.to_dict() for row in rows],
            "source_ref": source_ref,
            "table_index": table_index,
        }
        payload_sha256 = sha256_json(payload)
        unit_id = sha256_json(
            {
                "payload_sha256": payload_sha256,
                "source_ref": source_ref,
                "source_sha256": package.source_sha256,
            }
        )
        units.append(
            TableUnit(
                schema_version=TABLE_UNIT_SCHEMA_VERSION,
                unit_id=unit_id,
                source_name=package.source_name,
                source_sha256=package.source_sha256,
                table_index=table_index,
                body_index=body_index,
                source_ref=source_ref,
                rows=rows,
                payload_sha256=payload_sha256,
            )
        )

    return TableExtraction(
        source_name=package.source_name,
        source_sha256=package.source_sha256,
        units=tuple(units),
    )


def _extract_rows(
    table: ElementTree.Element,
    table_ref: str,
) -> tuple[TableRow, ...]:
    """Extract native rows from one table."""
    rows: list[TableRow] = []
    for row_index, row in enumerate(child for child in table if child.tag == _ROW_TAG):
        row_ref = f"{table_ref}/row/{row_index}"
        cells: list[TableCell] = []
        for cell_index, cell in enumerate(child for child in row if child.tag == _CELL_TAG):
            cell_ref = f"{row_ref}/cell/{cell_index}"
            blocks, text = _extract_cell_blocks(cell, cell_ref)
            grid_span, vertical_merge = _cell_merge_metadata(cell, cell_ref)
            cells.append(
                TableCell(
                    cell_index=cell_index,
                    source_ref=cell_ref,
                    grid_span=grid_span,
                    vertical_merge=vertical_merge,
                    text=text,
                    blocks=blocks,
                )
            )
        rows.append(TableRow(row_index=row_index, source_ref=row_ref, cells=tuple(cells)))
    return tuple(rows)


def _extract_cell_blocks(
    cell: ElementTree.Element,
    cell_ref: str,
) -> tuple[tuple[TableCellParagraph | TableCellTable, ...], str]:
    """Extract ordered paragraph and nested-table blocks from one table cell."""
    blocks: list[TableCellParagraph | TableCellTable] = []
    paragraph_texts: list[str] = []
    paragraph_ordinal = 0
    table_ordinal = 0
    for child in cell:
        if child.tag == _PARAGRAPH_TAG:
            paragraph_ref = f"{cell_ref}/paragraph/{paragraph_ordinal}"
            paragraph_index = paragraph_ordinal
            paragraph_ordinal += 1
            runs = tuple(extract_paragraph_runs(child))
            text = "".join(run.text for run in runs)
            if not text.strip():
                continue
            paragraph_texts.append(text)
            blocks.append(
                TableCellParagraph(
                    paragraph_index=paragraph_index,
                    source_ref=paragraph_ref,
                    text=text,
                    runs=runs,
                )
            )
        elif child.tag == _TABLE_TAG:
            nested_ref = f"{cell_ref}/table/{table_ordinal}"
            table_index = table_ordinal
            table_ordinal += 1
            nested_rows = _extract_rows(child, nested_ref)
            blocks.append(
                TableCellTable(
                    table_index=table_index,
                    source_ref=nested_ref,
                    rows=nested_rows,
                )
            )
    return tuple(blocks), "\n".join(paragraph_texts)


def _cell_merge_metadata(cell: ElementTree.Element, cell_ref: str) -> tuple[int, str | None]:
    """Return native horizontal and vertical merge metadata."""
    properties = cell.find(_CELL_PROPERTIES_TAG)
    if properties is None:
        return _DEFAULT_GRID_SPAN, None

    grid_span = _DEFAULT_GRID_SPAN
    grid_span_element = properties.find(_GRID_SPAN_TAG)
    if grid_span_element is not None:
        value = grid_span_element.get(_VALUE_ATTRIBUTE)
        grid_span = int(value) if value is not None else _DEFAULT_GRID_SPAN
        if grid_span < _DEFAULT_GRID_SPAN:
            raise DocxTableExtractionError.invalid_grid_span(cell_ref)

    vertical_merge = None
    vertical_merge_element = properties.find(_VERTICAL_MERGE_TAG)
    if vertical_merge_element is not None:
        vertical_merge = vertical_merge_element.get(_VALUE_ATTRIBUTE) or _VERTICAL_MERGE_CONTINUE
    return grid_span, vertical_merge
