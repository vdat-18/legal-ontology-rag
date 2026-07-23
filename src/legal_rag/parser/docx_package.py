"""Load one DOCX package for shared parser extraction stages."""

from __future__ import annotations

import hashlib
import io
import zipfile
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from xml.etree import ElementTree

from defusedxml import ElementTree as SafeElementTree
from defusedxml.common import DefusedXmlException

DOCUMENT_PART = "word/document.xml"


class DocxPackageError(ValueError):
    """Raised when a DOCX package cannot be loaded safely."""


class InvalidDocxPackageError(DocxPackageError):
    """Raised when source bytes are not a valid ZIP package."""

    def __init__(self, path: Path) -> None:
        """Identify the invalid source archive."""
        super().__init__(f"Source document is not a valid DOCX archive: {path}")


class MissingDocxPartError(DocxPackageError):
    """Raised when a required package part is absent."""

    def __init__(self, path: Path) -> None:
        """Identify the package missing its main document part."""
        super().__init__(f"DOCX is missing {DOCUMENT_PART}: {path}")


class MalformedDocxPartError(DocxPackageError):
    """Raised when a required XML package part is malformed."""

    def __init__(self, path: Path) -> None:
        """Identify the package containing malformed document XML."""
        super().__init__(f"Malformed DOCX package part {DOCUMENT_PART}: {path}")


@dataclass(slots=True)
class DocxPackage:
    """One open DOCX archive with immutable source identity and parsed document XML."""

    input_path: Path
    source_sha256: str
    document: ElementTree.Element
    archive: zipfile.ZipFile
    _buffer: io.BytesIO

    @property
    def source_name(self) -> str:
        """Return the source filename persisted in parser records."""
        return self.input_path.name

    @classmethod
    def open(cls, input_path: Path) -> DocxPackage:
        """Read, identify, open, and parse one DOCX package."""
        source_bytes = input_path.read_bytes()
        source_sha256 = hashlib.sha256(source_bytes).hexdigest()
        buffer = io.BytesIO(source_bytes)
        with ExitStack() as resources:
            resources.enter_context(buffer)
            try:
                archive = resources.enter_context(zipfile.ZipFile(buffer))
            except zipfile.BadZipFile as error:
                raise InvalidDocxPackageError(input_path) from error
            try:
                document_bytes = archive.read(DOCUMENT_PART)
            except KeyError as error:
                raise MissingDocxPartError(input_path) from error
            try:
                document = SafeElementTree.fromstring(document_bytes)
            except (ElementTree.ParseError, DefusedXmlException) as error:
                raise MalformedDocxPartError(input_path) from error
            package = cls(
                input_path=input_path,
                source_sha256=source_sha256,
                document=document,
                archive=archive,
                _buffer=buffer,
            )
            resources.pop_all()
            return package

    def close(self) -> None:
        """Close the archive and its backing in-memory buffer."""
        self.archive.close()
        self._buffer.close()

    def __enter__(self) -> DocxPackage:
        """Return this loaded package for a managed extraction scope."""
        return self

    def __exit__(
        self,
        _exception_type: type[BaseException] | None,
        _exception: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        """Close package resources when extraction leaves its managed scope."""
        self.close()
