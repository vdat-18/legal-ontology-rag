"""Extract embedded OLE package attachments from DOCX files."""

from __future__ import annotations

import hashlib
import io
import mimetypes
import posixpath
import struct
import zipfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from xml.etree import ElementTree

import olefile
from defusedxml import ElementTree as SafeElementTree

from legal_rag.parser.docx_package import DocxPackage
from legal_rag.parser.hashing import sha256_json

ATTACHMENT_SCHEMA_VERSION = "1.0.0"
ATTACHMENT_ASSET_DIRECTORY = "assets"

_DOCUMENT_PART = "word/document.xml"
_RELATIONSHIPS_PART = "word/_rels/document.xml.rels"
_OLE_NATIVE_STREAM = "\x01Ole10Native"
_OLE_RELATIONSHIP_TYPE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/oleObject"

_WORDPROCESSINGML_PREFIX = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_OFFICE_PREFIX = "{urn:schemas-microsoft-com:office:office}"
_RELATIONSHIP_ATTRIBUTE_PREFIX = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
_PACKAGE_RELATIONSHIPS_PREFIX = "{http://schemas.openxmlformats.org/package/2006/relationships}"

_BODY_TAG = f"{_WORDPROCESSINGML_PREFIX}body"
_PARAGRAPH_TAG = f"{_WORDPROCESSINGML_PREFIX}p"
_RUN_TAG = f"{_WORDPROCESSINGML_PREFIX}r"
_TABLE_TAG = f"{_WORDPROCESSINGML_PREFIX}tbl"
_ROW_TAG = f"{_WORDPROCESSINGML_PREFIX}tr"
_CELL_TAG = f"{_WORDPROCESSINGML_PREFIX}tc"
_OLE_OBJECT_TAG = f"{_OFFICE_PREFIX}OLEObject"
_RELATIONSHIP_TAG = f"{_PACKAGE_RELATIONSHIPS_PREFIX}Relationship"

_RELATIONSHIP_ID_ATTRIBUTE = f"{_RELATIONSHIP_ATTRIBUTE_PREFIX}id"


class DocxAttachmentExtractionError(ValueError):
    """Raised when an embedded DOCX attachment cannot be extracted safely."""

    @classmethod
    def missing_body(cls, path: Path) -> DocxAttachmentExtractionError:
        """Build an error for a document without a main body."""
        return cls(f"DOCX main document has no body: {path}")

    @classmethod
    def missing_relationship_id(cls, source_ref: str) -> DocxAttachmentExtractionError:
        """Build an error for an OLE object without a relationship ID."""
        return cls(f"DOCX OLE object has no relationship ID: {source_ref}")

    @classmethod
    def unknown_relationship(cls, relationship_id: str) -> DocxAttachmentExtractionError:
        """Build an error for an unknown attachment relationship."""
        return cls(f"DOCX attachment references an unknown relationship: {relationship_id}")

    @classmethod
    def external_relationship(cls, relationship_id: str) -> DocxAttachmentExtractionError:
        """Build an error for an externally linked attachment."""
        return cls(f"DOCX attachment relationship is external: {relationship_id}")

    @classmethod
    def invalid_relationship_type(cls, relationship_id: str) -> DocxAttachmentExtractionError:
        """Build an error for a relationship that is not an OLE object."""
        return cls(f"DOCX attachment relationship is not an OLE object: {relationship_id}")

    @classmethod
    def missing_part(cls, package_path: str) -> DocxAttachmentExtractionError:
        """Build an error for a missing OLE package part."""
        return cls(f"DOCX attachment part is missing: {package_path}")

    @classmethod
    def missing_native_stream(cls) -> DocxAttachmentExtractionError:
        """Build an error for an OLE object without a packaged file."""
        return cls("DOCX OLE object has no Ole10Native package stream")

    @classmethod
    def unreadable_ole(cls) -> DocxAttachmentExtractionError:
        """Build an error for an invalid OLE compound file."""
        return cls("DOCX attachment is not a readable OLE package")

    @classmethod
    def malformed_native_stream(cls) -> DocxAttachmentExtractionError:
        """Build an error for a malformed Ole10Native package stream."""
        return cls("DOCX attachment has a malformed Ole10Native package stream")

    @classmethod
    def unsafe_target(cls, relationship_id: str, target: str) -> DocxAttachmentExtractionError:
        """Build an error for a relationship target outside the DOCX package."""
        return cls(f"DOCX attachment relationship has an unsafe target ({relationship_id}): {target}")


@dataclass(frozen=True, slots=True)
class _Relationship:
    """One main-document OOXML relationship."""

    relationship_type: str
    target: str
    external: bool


@dataclass(frozen=True, slots=True)
class DocxAttachment:
    """One embedded attachment with content-addressed payload bytes."""

    schema_version: str
    attachment_id: str
    source_name: str
    source_sha256: str
    body_index: int
    source_ref: str
    parent_source_ref: str
    relationship_id: str
    package_path: str
    ole_sha256: str
    filename: str
    mime_type: str
    size_bytes: int
    payload_sha256: str
    asset_path: str
    payload_bytes: bytes = field(repr=False)

    def to_dict(self) -> dict[str, object]:
        """Return attachment metadata without embedding binary payload bytes."""
        return {
            "schema_version": self.schema_version,
            "attachment_id": self.attachment_id,
            "source_name": self.source_name,
            "source_sha256": self.source_sha256,
            "body_index": self.body_index,
            "source_ref": self.source_ref,
            "parent_source_ref": self.parent_source_ref,
            "relationship_id": self.relationship_id,
            "package_path": self.package_path,
            "ole_sha256": self.ole_sha256,
            "filename": self.filename,
            "mime_type": self.mime_type,
            "size_bytes": self.size_bytes,
            "payload_sha256": self.payload_sha256,
            "asset_path": self.asset_path,
        }


@dataclass(frozen=True, slots=True)
class DocxAttachmentExtraction:
    """Ordered attachments extracted from one DOCX package."""

    source_name: str
    source_sha256: str
    attachments: tuple[DocxAttachment, ...]


def extract_docx_attachments(input_path: Path) -> DocxAttachmentExtraction:
    """Extract embedded OLE package attachments from one DOCX file."""
    with DocxPackage.open(input_path) as package:
        return extract_package_attachments(package)


def extract_package_attachments(package: DocxPackage) -> DocxAttachmentExtraction:
    """Extract attachments from an already loaded DOCX package."""
    relationships = _read_relationships(package.archive)
    body = package.document.find(_BODY_TAG)
    if body is None:
        raise DocxAttachmentExtractionError.missing_body(package.input_path)
    attachments = _extract_body_attachments(
        package.archive,
        body,
        package.source_name,
        package.source_sha256,
        relationships,
    )
    return DocxAttachmentExtraction(package.source_name, package.source_sha256, tuple(attachments))


def _extract_body_attachments(
    archive: zipfile.ZipFile,
    body: ElementTree.Element,
    source_name: str,
    source_sha256: str,
    relationships: Mapping[str, _Relationship],
) -> list[DocxAttachment]:
    """Walk supported body structures in native document order."""
    attachments: list[DocxAttachment] = []
    for body_index, element in enumerate(body):
        parent_ref = f"{_DOCUMENT_PART}#/body/{body_index}"
        if element.tag == _PARAGRAPH_TAG:
            _append_paragraph_attachments(
                archive,
                element,
                parent_ref,
                body_index,
                source_name,
                source_sha256,
                relationships,
                attachments,
            )
        elif element.tag == _TABLE_TAG:
            _append_table_attachments(
                archive,
                element,
                parent_ref,
                body_index,
                source_name,
                source_sha256,
                relationships,
                attachments,
            )
    return attachments


def _append_table_attachments(
    archive: zipfile.ZipFile,
    table: ElementTree.Element,
    table_ref: str,
    body_index: int,
    source_name: str,
    source_sha256: str,
    relationships: Mapping[str, _Relationship],
    attachments: list[DocxAttachment],
) -> None:
    """Append attachments from one table, including nested tables."""
    for row_index, row in enumerate(child for child in table if child.tag == _ROW_TAG):
        row_ref = f"{table_ref}/row/{row_index}"
        for cell_index, cell in enumerate(child for child in row if child.tag == _CELL_TAG):
            cell_ref = f"{row_ref}/cell/{cell_index}"
            paragraph_ordinal = 0
            table_ordinal = 0
            for child in cell:
                if child.tag == _PARAGRAPH_TAG:
                    paragraph_ref = f"{cell_ref}/paragraph/{paragraph_ordinal}"
                    paragraph_ordinal += 1
                    _append_paragraph_attachments(
                        archive,
                        child,
                        paragraph_ref,
                        body_index,
                        source_name,
                        source_sha256,
                        relationships,
                        attachments,
                    )
                elif child.tag == _TABLE_TAG:
                    nested_ref = f"{cell_ref}/table/{table_ordinal}"
                    table_ordinal += 1
                    _append_table_attachments(
                        archive,
                        child,
                        nested_ref,
                        body_index,
                        source_name,
                        source_sha256,
                        relationships,
                        attachments,
                    )


def _append_paragraph_attachments(
    archive: zipfile.ZipFile,
    paragraph: ElementTree.Element,
    parent_ref: str,
    body_index: int,
    source_name: str,
    source_sha256: str,
    relationships: Mapping[str, _Relationship],
    attachments: list[DocxAttachment],
) -> None:
    """Append embedded OLE packages from one native paragraph."""
    for run_index, run in enumerate(paragraph.iter(_RUN_TAG)):
        for attachment_index, ole_object in enumerate(_iter_run_ole_objects(run)):
            relationship_id = ole_object.get(_RELATIONSHIP_ID_ATTRIBUTE)
            if relationship_id is None:
                raise DocxAttachmentExtractionError.missing_relationship_id(parent_ref)
            source_ref = f"{parent_ref}/run/{run_index}/attachment/{attachment_index}"
            attachments.append(
                _build_attachment(
                    archive,
                    relationship_id,
                    source_name,
                    source_sha256,
                    body_index,
                    source_ref,
                    parent_ref,
                    relationships,
                )
            )


def _iter_run_ole_objects(run: ElementTree.Element) -> Iterable[ElementTree.Element]:
    """Yield OLE objects owned by one run without entering nested runs."""

    def walk(element: ElementTree.Element) -> Iterable[ElementTree.Element]:
        for child in element:
            if child.tag == _RUN_TAG:
                continue
            if child.tag == _OLE_OBJECT_TAG and child.get("Type") == "Embed":
                yield child
            yield from walk(child)

    yield from walk(run)


def _build_attachment(
    archive: zipfile.ZipFile,
    relationship_id: str,
    source_name: str,
    source_sha256: str,
    body_index: int,
    source_ref: str,
    parent_ref: str,
    relationships: Mapping[str, _Relationship],
) -> DocxAttachment:
    """Resolve one OLE relationship and extract its native payload."""
    relationship = relationships.get(relationship_id)
    if relationship is None:
        raise DocxAttachmentExtractionError.unknown_relationship(relationship_id)
    if relationship.external:
        raise DocxAttachmentExtractionError.external_relationship(relationship_id)
    if relationship.relationship_type != _OLE_RELATIONSHIP_TYPE:
        raise DocxAttachmentExtractionError.invalid_relationship_type(relationship_id)

    package_path = _resolve_package_path(relationship_id, relationship.target)
    try:
        ole_bytes = archive.read(package_path)
    except KeyError as error:
        raise DocxAttachmentExtractionError.missing_part(package_path) from error
    filename, payload_bytes = _read_ole_package(ole_bytes)
    payload_sha256 = hashlib.sha256(payload_bytes).hexdigest()
    asset_path = PurePosixPath(ATTACHMENT_ASSET_DIRECTORY, payload_sha256).as_posix()
    mime_type = mimetypes.MimeTypes().guess_type(filename, strict=False)[0] or "application/octet-stream"
    attachment_id = sha256_json(
        {
            "payload_sha256": payload_sha256,
            "source_ref": source_ref,
            "source_sha256": source_sha256,
        }
    )
    return DocxAttachment(
        schema_version=ATTACHMENT_SCHEMA_VERSION,
        attachment_id=attachment_id,
        source_name=source_name,
        source_sha256=source_sha256,
        body_index=body_index,
        source_ref=source_ref,
        parent_source_ref=parent_ref,
        relationship_id=relationship_id,
        package_path=package_path,
        ole_sha256=hashlib.sha256(ole_bytes).hexdigest(),
        filename=filename,
        mime_type=mime_type,
        size_bytes=len(payload_bytes),
        payload_sha256=payload_sha256,
        asset_path=asset_path,
        payload_bytes=payload_bytes,
    )


def _read_ole_package(ole_bytes: bytes) -> tuple[str, bytes]:
    """Read the native file stored inside one OLE Package object."""
    try:
        with olefile.OleFileIO(io.BytesIO(ole_bytes)) as compound_file:
            native_bytes = (
                compound_file.openstream(_OLE_NATIVE_STREAM).read()
                if compound_file.exists(_OLE_NATIVE_STREAM)
                else None
            )
    except OSError as error:
        raise DocxAttachmentExtractionError.unreadable_ole() from error
    if native_bytes is None:
        raise DocxAttachmentExtractionError.missing_native_stream()
    return _parse_ole10_native(native_bytes)


def _parse_ole10_native(native_bytes: bytes) -> tuple[str, bytes]:
    """Parse the documented Ole10Native Package stream structure."""
    total_size = _read_uint32(native_bytes, 0)
    if total_size != len(native_bytes) - 4:
        raise DocxAttachmentExtractionError.malformed_native_stream()
    offset = 6
    filename_bytes, offset = _read_null_terminated(native_bytes, offset)
    _, offset = _read_null_terminated(native_bytes, offset)
    offset += 8
    _, offset = _read_null_terminated(native_bytes, offset)
    payload_size = _read_uint32(native_bytes, offset)
    offset += 4
    payload_end = offset + payload_size
    if payload_end > len(native_bytes):
        raise DocxAttachmentExtractionError.malformed_native_stream()
    try:
        filename = PureWindowsPath(filename_bytes.decode("windows-1252")).name
    except UnicodeDecodeError as error:  # pragma: no cover - windows-1252 maps every byte
        raise DocxAttachmentExtractionError.malformed_native_stream() from error
    if not filename:
        raise DocxAttachmentExtractionError.malformed_native_stream()
    return filename, native_bytes[offset:payload_end]


def _read_null_terminated(data: bytes, offset: int) -> tuple[bytes, int]:
    """Read one null-terminated Ole10Native byte string."""
    try:
        end = data.index(0, offset)
    except ValueError as error:
        raise DocxAttachmentExtractionError.malformed_native_stream() from error
    return data[offset:end], end + 1


def _read_uint32(data: bytes, offset: int) -> int:
    """Read one little-endian unsigned integer from an Ole10Native stream."""
    try:
        return struct.unpack_from("<I", data, offset)[0]
    except struct.error as error:
        raise DocxAttachmentExtractionError.malformed_native_stream() from error


def _read_relationships(archive: zipfile.ZipFile) -> dict[str, _Relationship]:
    """Read main-document relationships keyed by relationship ID."""
    try:
        relationship_bytes = archive.read(_RELATIONSHIPS_PART)
    except KeyError:
        return {}
    root = SafeElementTree.fromstring(relationship_bytes)
    relationships: dict[str, _Relationship] = {}
    for element in root.findall(_RELATIONSHIP_TAG):
        relationship_id = element.get("Id")
        relationship_type = element.get("Type")
        target = element.get("Target")
        if relationship_id is None or relationship_type is None or target is None:
            continue
        relationships[relationship_id] = _Relationship(
            relationship_type=relationship_type,
            target=target,
            external=element.get("TargetMode") == "External",
        )
    return relationships


def _resolve_package_path(relationship_id: str, target: str) -> str:
    """Resolve a main-document relationship target within the DOCX package."""
    package_path = posixpath.normpath(posixpath.join(posixpath.dirname(_DOCUMENT_PART), target))
    path = PurePosixPath(package_path)
    if path.is_absolute() or ".." in path.parts:
        raise DocxAttachmentExtractionError.unsafe_target(relationship_id, target)
    return package_path
