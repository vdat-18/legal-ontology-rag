"""Extract searchable rows from XLSX worksheet OOXML."""

from __future__ import annotations

import hashlib
import io
import posixpath
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from defusedxml import ElementTree as SafeElementTree

from legal_rag.parser.hashing import sha256_json

SPREADSHEET_ROW_SCHEMA_VERSION = "1.0.0"
KIND_SPREADSHEET_ROW = "spreadsheet_row"

_WORKBOOK_PART = "xl/workbook.xml"
_WORKBOOK_RELATIONSHIPS_PART = "xl/_rels/workbook.xml.rels"
_SHARED_STRINGS_PART = "xl/sharedStrings.xml"
_SPREADSHEET_PREFIX = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_RELATIONSHIP_ATTRIBUTE_PREFIX = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
_PACKAGE_RELATIONSHIPS_PREFIX = "{http://schemas.openxmlformats.org/package/2006/relationships}"

_SHEET_TAG = f"{_SPREADSHEET_PREFIX}sheet"
_SHEET_DATA_TAG = f"{_SPREADSHEET_PREFIX}sheetData"
_ROW_TAG = f"{_SPREADSHEET_PREFIX}row"
_CELL_TAG = f"{_SPREADSHEET_PREFIX}c"
_VALUE_TAG = f"{_SPREADSHEET_PREFIX}v"
_FORMULA_TAG = f"{_SPREADSHEET_PREFIX}f"
_TEXT_TAG = f"{_SPREADSHEET_PREFIX}t"
_RELATIONSHIP_TAG = f"{_PACKAGE_RELATIONSHIPS_PREFIX}Relationship"
_RELATIONSHIP_ID_ATTRIBUTE = f"{_RELATIONSHIP_ATTRIBUTE_PREFIX}id"


class SpreadsheetExtractionError(ValueError):
    """Raised when an XLSX package cannot be read safely."""


@dataclass(frozen=True, slots=True)
class SpreadsheetCell:
    """One populated worksheet cell with its native reference."""

    cell_ref: str
    value: str | None
    formula: str | None

    def to_dict(self) -> dict[str, str | None]:
        """Return a JSON-serializable cell record."""
        return {"cell_ref": self.cell_ref, "value": self.value, "formula": self.formula}


@dataclass(frozen=True, slots=True)
class SpreadsheetRow:
    """One content-bearing worksheet row."""

    schema_version: str
    unit_id: str
    source_name: str
    source_sha256: str
    sequence_index: int
    sheet_name: str
    row_index: int
    source_ref: str
    payload_sha256: str
    text: str
    cells: tuple[SpreadsheetCell, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable spreadsheet-row record."""
        return {
            "schema_version": self.schema_version,
            "unit_id": self.unit_id,
            "source_name": self.source_name,
            "source_sha256": self.source_sha256,
            "sequence_index": self.sequence_index,
            "source_ref": self.source_ref,
            "kind": KIND_SPREADSHEET_ROW,
            "payload_sha256": self.payload_sha256,
            "sheet_name": self.sheet_name,
            "row_index": self.row_index,
            "text": self.text,
            "cells": [cell.to_dict() for cell in self.cells],
        }


def extract_xlsx_rows(source_bytes: bytes, source_name: str) -> tuple[SpreadsheetRow, ...]:
    """Extract populated rows from an XLSX byte stream in workbook order."""
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    try:
        archive = zipfile.ZipFile(io.BytesIO(source_bytes))
    except zipfile.BadZipFile as error:
        message = f"Invalid XLSX package: {source_name}"
        raise SpreadsheetExtractionError(message) from error
    with archive:
        workbook = SafeElementTree.fromstring(_read_part(archive, _WORKBOOK_PART, source_name))
        relationships = _workbook_relationships(archive, source_name)
        shared_strings = _shared_strings(archive)
        rows: list[SpreadsheetRow] = []
        for sheet in workbook.iter(_SHEET_TAG):
            sheet_name = sheet.get("name")
            relationship_id = sheet.get(_RELATIONSHIP_ID_ATTRIBUTE)
            if sheet_name is None or relationship_id is None or relationship_id not in relationships:
                message = f"Malformed XLSX workbook metadata: {source_name}"
                raise SpreadsheetExtractionError(message)
            worksheet_path = _resolve_workbook_target(relationships[relationship_id], source_name)
            worksheet = SafeElementTree.fromstring(_read_part(archive, worksheet_path, source_name))
            sheet_data = worksheet.find(f".//{_SHEET_DATA_TAG}")
            if sheet_data is None:
                continue
            for row in sheet_data.findall(_ROW_TAG):
                cells = tuple(_extract_cell(cell, shared_strings, source_name) for cell in row.findall(_CELL_TAG))
                cells = tuple(cell for cell in cells if cell.value is not None or cell.formula is not None)
                if not cells:
                    continue
                row_index = int(row.get("r") or len(rows) + 1)
                source_ref = f"{worksheet_path}#/row/{row_index}"
                text = "\t".join(cell.value or cell.formula or "" for cell in cells)
                payload_sha256 = sha256_json(
                    {
                        "cells": [cell.to_dict() for cell in cells],
                        "row_index": row_index,
                        "sheet_name": sheet_name,
                        "source_ref": source_ref,
                    }
                )
                rows.append(
                    SpreadsheetRow(
                        schema_version=SPREADSHEET_ROW_SCHEMA_VERSION,
                        unit_id=sha256_json(
                            {
                                "payload_sha256": payload_sha256,
                                "source_ref": source_ref,
                                "source_sha256": source_sha256,
                            }
                        ),
                        source_name=source_name,
                        source_sha256=source_sha256,
                        sequence_index=len(rows),
                        sheet_name=sheet_name,
                        row_index=row_index,
                        source_ref=source_ref,
                        payload_sha256=payload_sha256,
                        text=text,
                        cells=cells,
                    )
                )
    return tuple(rows)


def _extract_cell(
    cell: Any,
    shared_strings: tuple[str, ...],
    source_name: str,
) -> SpreadsheetCell:
    """Read one worksheet cell without interpreting formatting semantics."""
    cell_ref = cell.get("r")
    if cell_ref is None:
        message = f"XLSX cell has no reference: {source_name}"
        raise SpreadsheetExtractionError(message)
    formula_element = cell.find(_FORMULA_TAG)
    formula = formula_element.text if formula_element is not None else None
    cell_type = cell.get("t")
    if cell_type == "inlineStr":
        value = "".join(text.text or "" for text in cell.iter(_TEXT_TAG))
    else:
        value_element = cell.find(_VALUE_TAG)
        value = value_element.text if value_element is not None else None
        if cell_type == "s" and value is not None:
            try:
                value = shared_strings[int(value)]
            except (IndexError, ValueError) as error:
                message = f"Invalid XLSX shared-string reference: {source_name}"
                raise SpreadsheetExtractionError(message) from error
    return SpreadsheetCell(cell_ref=cell_ref, value=value, formula=formula)


def _workbook_relationships(archive: zipfile.ZipFile, source_name: str) -> dict[str, str]:
    """Return workbook relationship targets keyed by relationship ID."""
    root = SafeElementTree.fromstring(_read_part(archive, _WORKBOOK_RELATIONSHIPS_PART, source_name))
    relationships: dict[str, str] = {}
    for relationship in root.findall(_RELATIONSHIP_TAG):
        relationship_id = relationship.get("Id")
        target = relationship.get("Target")
        if relationship_id is not None and target is not None:
            relationships[relationship_id] = target
    return relationships


def _shared_strings(archive: zipfile.ZipFile) -> tuple[str, ...]:
    """Return workbook shared strings, including rich-text fragments."""
    try:
        root = SafeElementTree.fromstring(archive.read(_SHARED_STRINGS_PART))
    except KeyError:
        return ()
    return tuple("".join(text.text or "" for text in item.iter(_TEXT_TAG)) for item in root)


def _resolve_workbook_target(target: str, source_name: str) -> str:
    """Resolve a worksheet relationship target within the XLSX package."""
    package_path = posixpath.normpath(posixpath.join(posixpath.dirname(_WORKBOOK_PART), target))
    path = PurePosixPath(package_path)
    if path.is_absolute() or ".." in path.parts:
        message = f"Unsafe XLSX worksheet target in {source_name}: {target}"
        raise SpreadsheetExtractionError(message)
    return package_path


def _read_part(archive: zipfile.ZipFile, part: str, source_name: str) -> bytes:
    """Read one required XLSX package part."""
    try:
        return archive.read(part)
    except KeyError as error:
        message = f"XLSX package is missing {part}: {source_name}"
        raise SpreadsheetExtractionError(message) from error
