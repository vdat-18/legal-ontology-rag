"""Tests for loading the shared DOCX package boundary."""

from __future__ import annotations

import zipfile
from collections.abc import Callable
from pathlib import Path

import pytest

from legal_rag.parser.attachments import DocxAttachmentExtraction, extract_docx_attachments
from legal_rag.parser.docx_package import (
    DOCUMENT_PART,
    DocxPackage,
    InvalidDocxPackageError,
    MalformedDocxPartError,
    MissingDocxPartError,
)
from legal_rag.parser.images import DocxImageExtraction, extract_docx_images
from legal_rag.parser.paragraphs import ParagraphExtraction, extract_docx_paragraphs
from legal_rag.parser.tables import TableExtraction, extract_docx_tables


def test_open_rejects_invalid_archive(tmp_path: Path) -> None:
    """Non-ZIP input raises the package-specific archive error."""
    input_path = tmp_path / "invalid.docx"
    input_path.write_bytes(b"not a DOCX archive")

    with pytest.raises(InvalidDocxPackageError):
        DocxPackage.open(input_path)


@pytest.mark.parametrize(
    "extractor",
    [extract_docx_paragraphs, extract_docx_tables, extract_docx_images, extract_docx_attachments],
)
def test_extractors_preserve_package_errors(
    tmp_path: Path,
    extractor: Callable[
        [Path],
        ParagraphExtraction | TableExtraction | DocxImageExtraction | DocxAttachmentExtraction,
    ],
) -> None:
    """Standalone extractors do not rename package boundary errors."""
    input_path = tmp_path / "invalid.docx"
    input_path.write_bytes(b"not a DOCX archive")

    with pytest.raises(InvalidDocxPackageError):
        extractor(input_path)


def test_open_rejects_missing_document_part(tmp_path: Path) -> None:
    """A ZIP without the main OOXML document part is rejected."""
    input_path = tmp_path / "missing-part.docx"
    with zipfile.ZipFile(input_path, mode="w") as archive:
        archive.writestr("placeholder", b"")

    with pytest.raises(MissingDocxPartError):
        DocxPackage.open(input_path)


@pytest.mark.parametrize(
    "document_xml",
    [
        b"<invalid>",
        b'<!DOCTYPE doc [<!ENTITY value "blocked">]><doc>&value;</doc>',
    ],
)
def test_open_rejects_unsafe_or_malformed_document_xml(
    tmp_path: Path,
    document_xml: bytes,
) -> None:
    """Malformed and unsafe XML use the same package boundary error."""
    input_path = tmp_path / "invalid-xml.docx"
    with zipfile.ZipFile(input_path, mode="w") as archive:
        archive.writestr(DOCUMENT_PART, document_xml)

    with pytest.raises(MalformedDocxPartError):
        DocxPackage.open(input_path)
