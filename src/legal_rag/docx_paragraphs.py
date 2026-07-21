"""Extract content-bearing paragraphs from native DOCX OOXML."""

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

PARAGRAPH_UNIT_SCHEMA_VERSION = "1.0.0"
PARAGRAPH_MANIFEST_SCHEMA_VERSION = "1.0.0"

_DOCUMENT_PART = "word/document.xml"
_WORDPROCESSINGML_NAMESPACE = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_WORDPROCESSINGML_PREFIX = f"{{{_WORDPROCESSINGML_NAMESPACE}}}"
_BODY_TAG = f"{_WORDPROCESSINGML_PREFIX}body"
_PARAGRAPH_TAG = f"{_WORDPROCESSINGML_PREFIX}p"
_RUN_TAG = f"{_WORDPROCESSINGML_PREFIX}r"
_RUN_PROPERTIES_TAG = f"{_WORDPROCESSINGML_PREFIX}rPr"
_TEXT_TAG = f"{_WORDPROCESSINGML_PREFIX}t"
_TAB_TAG = f"{_WORDPROCESSINGML_PREFIX}tab"
_BREAK_TAG = f"{_WORDPROCESSINGML_PREFIX}br"
_CARRIAGE_RETURN_TAG = f"{_WORDPROCESSINGML_PREFIX}cr"
_NO_BREAK_HYPHEN_TAG = f"{_WORDPROCESSINGML_PREFIX}noBreakHyphen"
_SOFT_HYPHEN_TAG = f"{_WORDPROCESSINGML_PREFIX}softHyphen"
_VERTICAL_ALIGNMENT_TAG = f"{_WORDPROCESSINGML_PREFIX}vertAlign"
_VALUE_ATTRIBUTE = f"{_WORDPROCESSINGML_PREFIX}val"


class DocxParagraphExtractionError(ValueError):
    """Raised when native DOCX paragraphs cannot be extracted safely."""

    @classmethod
    def invalid_archive(cls, path: Path) -> DocxParagraphExtractionError:
        """Build an error for a file that is not a valid DOCX archive."""
        return cls(f"Source document is not a valid DOCX archive: {path}")

    @classmethod
    def missing_document_part(cls, path: Path) -> DocxParagraphExtractionError:
        """Build an error for a DOCX without its main document part."""
        return cls(f"DOCX is missing {_DOCUMENT_PART}: {path}")

    @classmethod
    def malformed_document_part(cls, path: Path) -> DocxParagraphExtractionError:
        """Build an error for malformed main-document XML."""
        return cls(f"Malformed DOCX main document XML: {path}")

    @classmethod
    def missing_body(cls, path: Path) -> DocxParagraphExtractionError:
        """Build an error for a DOCX without a document body."""
        return cls(f"DOCX main document has no body: {path}")

    @classmethod
    def no_content(cls, path: Path) -> DocxParagraphExtractionError:
        """Build an error for a DOCX without content-bearing body paragraphs."""
        return cls(f"DOCX contains no content-bearing body paragraphs: {path}")


class ParagraphUnitValidationError(ValueError):
    """Raised when paragraph units cannot satisfy their persisted contract."""

    @classmethod
    def duplicate_unit_ids(cls) -> ParagraphUnitValidationError:
        """Build an error for repeated paragraph-unit IDs."""
        return cls("Paragraph units contain duplicate unit IDs")

    @classmethod
    def duplicate_source_refs(cls) -> ParagraphUnitValidationError:
        """Build an error for repeated OOXML source references."""
        return cls("Paragraph units contain duplicate source references")

    @classmethod
    def invalid_sequence(cls) -> ParagraphUnitValidationError:
        """Build an error for paragraph units outside source order."""
        return cls("Paragraph units are not in strictly increasing source order")


@dataclass(frozen=True, slots=True)
class ParagraphRun:
    """One visible OOXML run and its direct vertical alignment."""

    run_index: int
    text: str
    vertical_alignment: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable run record."""
        return {
            "run_index": self.run_index,
            "text": self.text,
            "vertical_alignment": self.vertical_alignment,
        }


@dataclass(frozen=True, slots=True)
class ParagraphUnit:
    """One content-bearing top-level DOCX paragraph with exact run boundaries."""

    schema_version: str
    unit_id: str
    source_name: str
    source_sha256: str
    sequence_index: int
    body_index: int
    source_ref: str
    text: str
    runs: tuple[ParagraphRun, ...]
    payload_sha256: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable paragraph-unit record."""
        return {
            "schema_version": self.schema_version,
            "unit_id": self.unit_id,
            "source_name": self.source_name,
            "source_sha256": self.source_sha256,
            "sequence_index": self.sequence_index,
            "body_index": self.body_index,
            "source_ref": self.source_ref,
            "text": self.text,
            "runs": [run.to_dict() for run in self.runs],
            "payload_sha256": self.payload_sha256,
        }


@dataclass(frozen=True, slots=True)
class ParagraphExtraction:
    """Paragraph units plus an audit trail of skipped empty paragraphs."""

    source_name: str
    source_sha256: str
    raw_paragraph_count: int
    units: tuple[ParagraphUnit, ...]
    skipped_empty_paragraph_refs: tuple[str, ...]

    def manifest(self) -> dict[str, Any]:
        """Return the paragraph-extraction audit manifest."""
        return {
            "schema_version": PARAGRAPH_MANIFEST_SCHEMA_VERSION,
            "source_name": self.source_name,
            "source_sha256": self.source_sha256,
            "raw_paragraph_count": self.raw_paragraph_count,
            "paragraph_unit_count": len(self.units),
            "skipped_empty_paragraph_count": len(self.skipped_empty_paragraph_refs),
            "skipped_empty_paragraph_refs": list(self.skipped_empty_paragraph_refs),
        }


def extract_docx_paragraphs(input_path: Path) -> ParagraphExtraction:
    """Extract top-level body paragraphs directly from native DOCX OOXML."""
    source_bytes = input_path.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    document_xml = _read_document_part(input_path, source_bytes)
    try:
        document = SafeElementTree.fromstring(document_xml)
    except ElementTree.ParseError as error:
        raise DocxParagraphExtractionError.malformed_document_part(input_path) from error

    body = document.find(_BODY_TAG)
    if body is None:
        raise DocxParagraphExtractionError.missing_body(input_path)

    units: list[ParagraphUnit] = []
    skipped_refs: list[str] = []
    paragraph_count = 0
    for body_index, element in enumerate(body):
        if element.tag != _PARAGRAPH_TAG:
            continue

        sequence_index = paragraph_count
        paragraph_count += 1
        source_ref = f"{_DOCUMENT_PART}#/body/{body_index}"
        runs = tuple(_extract_runs(element))
        text = "".join(run.text for run in runs)
        if not text.strip():
            skipped_refs.append(source_ref)
            continue

        payload = {
            "body_index": body_index,
            "runs": [run.to_dict() for run in runs],
            "sequence_index": sequence_index,
            "source_ref": source_ref,
            "text": text,
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
            ParagraphUnit(
                schema_version=PARAGRAPH_UNIT_SCHEMA_VERSION,
                unit_id=unit_id,
                source_name=input_path.name,
                source_sha256=source_sha256,
                sequence_index=sequence_index,
                body_index=body_index,
                source_ref=source_ref,
                text=text,
                runs=runs,
                payload_sha256=payload_sha256,
            )
        )

    if not units:
        raise DocxParagraphExtractionError.no_content(input_path)
    return ParagraphExtraction(
        source_name=input_path.name,
        source_sha256=source_sha256,
        raw_paragraph_count=paragraph_count,
        units=tuple(units),
        skipped_empty_paragraph_refs=tuple(skipped_refs),
    )


def write_docx_paragraphs(
    output_path: Path,
    manifest_path: Path,
    extraction: ParagraphExtraction,
) -> None:
    """Atomically write paragraph units and their audit manifest."""
    _validate_units(extraction.units)
    records = (unit.to_dict() for unit in extraction.units)
    _write_jsonl(output_path, records)
    _write_json(manifest_path, extraction.manifest())


def _read_document_part(input_path: Path, source_bytes: bytes) -> bytes:
    """Read the main OOXML document part from DOCX bytes."""
    try:
        with zipfile.ZipFile(io.BytesIO(source_bytes)) as archive:
            try:
                return archive.read(_DOCUMENT_PART)
            except KeyError as error:
                raise DocxParagraphExtractionError.missing_document_part(input_path) from error
    except zipfile.BadZipFile as error:
        raise DocxParagraphExtractionError.invalid_archive(input_path) from error


def _extract_runs(paragraph: ElementTree.Element) -> list[ParagraphRun]:
    """Return visible runs in their native OOXML order."""
    runs: list[ParagraphRun] = []
    for run_index, run in enumerate(paragraph.iter(_RUN_TAG)):
        text = _visible_run_text(run)
        if not text:
            continue
        properties = run.find(_RUN_PROPERTIES_TAG)
        alignment = None
        if properties is not None:
            vertical_alignment = properties.find(_VERTICAL_ALIGNMENT_TAG)
            if vertical_alignment is not None:
                alignment = vertical_alignment.get(_VALUE_ATTRIBUTE)
        runs.append(ParagraphRun(run_index=run_index, text=text, vertical_alignment=alignment))
    return runs


def _visible_run_text(run: ElementTree.Element) -> str:
    """Materialize visible OOXML text controls without semantic interpretation."""
    parts: list[str] = []
    for element in run.iter():
        if element.tag == _TEXT_TAG:
            parts.append(element.text or "")
        elif element.tag == _TAB_TAG:
            parts.append("\t")
        elif element.tag in {_BREAK_TAG, _CARRIAGE_RETURN_TAG}:
            parts.append("\n")
        elif element.tag == _NO_BREAK_HYPHEN_TAG:
            parts.append("\u2011")
        elif element.tag == _SOFT_HYPHEN_TAG:
            parts.append("\u00ad")
    return "".join(parts)


def _validate_units(units: tuple[ParagraphUnit, ...]) -> None:
    """Validate identifiers, references, and retained source order."""
    unit_ids = [unit.unit_id for unit in units]
    if len(unit_ids) != len(set(unit_ids)):
        raise ParagraphUnitValidationError.duplicate_unit_ids()
    source_refs = [unit.source_ref for unit in units]
    if len(source_refs) != len(set(source_refs)):
        raise ParagraphUnitValidationError.duplicate_source_refs()
    sequence_indices = [unit.sequence_index for unit in units]
    if sequence_indices != sorted(sequence_indices) or len(sequence_indices) != len(set(sequence_indices)):
        raise ParagraphUnitValidationError.invalid_sequence()


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
