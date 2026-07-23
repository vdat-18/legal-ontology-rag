"""Extract content-bearing paragraphs from native DOCX OOXML."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from legal_rag.parser.docx_package import DocxPackage
from legal_rag.parser.hashing import sha256_json

PARAGRAPH_UNIT_SCHEMA_VERSION = "1.0.0"

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

    def __init__(self, path: Path) -> None:
        """Identify a DOCX without a main document body."""
        super().__init__(f"DOCX main document has no body: {path}")


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
    """Content-bearing paragraph units extracted from one DOCX."""

    source_name: str
    source_sha256: str
    units: tuple[ParagraphUnit, ...]


def extract_docx_paragraphs(input_path: Path) -> ParagraphExtraction:
    """Extract top-level body paragraphs directly from native DOCX OOXML."""
    with DocxPackage.open(input_path) as package:
        return extract_package_paragraphs(package)


def extract_package_paragraphs(package: DocxPackage) -> ParagraphExtraction:
    """Extract paragraphs from an already loaded DOCX package."""
    input_path = package.input_path

    body = package.document.find(_BODY_TAG)
    if body is None:
        raise DocxParagraphExtractionError(input_path)

    units: list[ParagraphUnit] = []
    paragraph_count = 0
    for body_index, element in enumerate(body):
        if element.tag != _PARAGRAPH_TAG:
            continue

        sequence_index = paragraph_count
        paragraph_count += 1
        source_ref = f"{_DOCUMENT_PART}#/body/{body_index}"
        runs = tuple(extract_paragraph_runs(element))
        text = "".join(run.text for run in runs)
        if not text.strip():
            continue

        payload = {
            "body_index": body_index,
            "runs": [run.to_dict() for run in runs],
            "sequence_index": sequence_index,
            "source_ref": source_ref,
            "text": text,
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
            ParagraphUnit(
                schema_version=PARAGRAPH_UNIT_SCHEMA_VERSION,
                unit_id=unit_id,
                source_name=package.source_name,
                source_sha256=package.source_sha256,
                sequence_index=sequence_index,
                body_index=body_index,
                source_ref=source_ref,
                text=text,
                runs=runs,
                payload_sha256=payload_sha256,
            )
        )

    return ParagraphExtraction(
        source_name=package.source_name,
        source_sha256=package.source_sha256,
        units=tuple(units),
    )


def extract_paragraph_runs(paragraph: ElementTree.Element) -> list[ParagraphRun]:
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
    """Materialize text owned by one run without re-reading nested runs."""
    parts: list[str] = []

    def walk(element: ElementTree.Element) -> None:
        for child in element:
            if child.tag == _RUN_TAG:
                continue
            if child.tag == _TEXT_TAG:
                parts.append(child.text or "")
            elif child.tag == _TAB_TAG:
                parts.append("\t")
            elif child.tag in {_BREAK_TAG, _CARRIAGE_RETURN_TAG}:
                parts.append("\n")
            elif child.tag == _NO_BREAK_HYPHEN_TAG:
                parts.append("\u2011")
            elif child.tag == _SOFT_HYPHEN_TAG:
                parts.append("\u00ad")
            walk(child)

    walk(run)
    return "".join(parts)
