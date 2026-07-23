"""Extract embedded images and native OOXML provenance from DOCX files."""

from __future__ import annotations

import hashlib
import posixpath
import zipfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree

from defusedxml import ElementTree as SafeElementTree

from legal_rag.parser.docx_package import DocxPackage
from legal_rag.parser.hashing import sha256_json

IMAGE_SCHEMA_VERSION = "1.0.0"

_DOCUMENT_PART = "word/document.xml"
_RELATIONSHIPS_PART = "word/_rels/document.xml.rels"
_CONTENT_TYPES_PART = "[Content_Types].xml"

_WORDPROCESSINGML_PREFIX = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_DRAWINGML_PREFIX = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
_VML_PREFIX = "{urn:schemas-microsoft-com:vml}"
_OFFICE_PREFIX = "{urn:schemas-microsoft-com:office:office}"
_RELATIONSHIP_ATTRIBUTE_PREFIX = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
_PACKAGE_RELATIONSHIPS_PREFIX = "{http://schemas.openxmlformats.org/package/2006/relationships}"
_CONTENT_TYPES_PREFIX = "{http://schemas.openxmlformats.org/package/2006/content-types}"

_BODY_TAG = f"{_WORDPROCESSINGML_PREFIX}body"
_PARAGRAPH_TAG = f"{_WORDPROCESSINGML_PREFIX}p"
_RUN_TAG = f"{_WORDPROCESSINGML_PREFIX}r"
_TABLE_TAG = f"{_WORDPROCESSINGML_PREFIX}tbl"
_ROW_TAG = f"{_WORDPROCESSINGML_PREFIX}tr"
_CELL_TAG = f"{_WORDPROCESSINGML_PREFIX}tc"
_BLIP_TAG = f"{_DRAWINGML_PREFIX}blip"
_IMAGE_DATA_TAG = f"{_VML_PREFIX}imagedata"
_SHAPE_TAG = f"{_VML_PREFIX}shape"
_OLE_OBJECT_TAG = f"{_OFFICE_PREFIX}OLEObject"
_RELATIONSHIP_TAG = f"{_PACKAGE_RELATIONSHIPS_PREFIX}Relationship"
_CONTENT_TYPE_DEFAULT_TAG = f"{_CONTENT_TYPES_PREFIX}Default"
_CONTENT_TYPE_OVERRIDE_TAG = f"{_CONTENT_TYPES_PREFIX}Override"

_RELATIONSHIP_ID_ATTRIBUTE = f"{_RELATIONSHIP_ATTRIBUTE_PREFIX}id"
_RELATIONSHIP_EMBED_ATTRIBUTE = f"{_RELATIONSHIP_ATTRIBUTE_PREFIX}embed"


class DocxImageExtractionError(ValueError):
    """Raised when embedded DOCX images cannot be extracted safely."""

    @classmethod
    def missing_body(cls, path: Path) -> DocxImageExtractionError:
        """Build an error for a document without a main body."""
        return cls(f"DOCX main document has no body: {path}")

    @classmethod
    def unknown_relationship(cls, relationship_id: str) -> DocxImageExtractionError:
        """Build an error for an image with no relationship target."""
        return cls(f"DOCX image references an unknown relationship: {relationship_id}")

    @classmethod
    def external_relationship(cls, relationship_id: str) -> DocxImageExtractionError:
        """Build an error for an externally linked image."""
        return cls(f"DOCX image relationship is external and cannot be archived: {relationship_id}")

    @classmethod
    def unsafe_target(cls, relationship_id: str, target: str) -> DocxImageExtractionError:
        """Build an error for a relationship target outside the package."""
        return cls(f"DOCX image relationship has an unsafe target ({relationship_id}): {target}")

    @classmethod
    def unknown_content_type(cls, package_path: str) -> DocxImageExtractionError:
        """Build an error for media without a declared content type."""
        return cls(f"DOCX image media has no declared content type: {package_path}")

    @classmethod
    def invalid_content_type(cls, package_path: str, content_type: str) -> DocxImageExtractionError:
        """Build an error for a relationship that does not resolve to image media."""
        return cls(f"DOCX image media has a non-image content type ({content_type}): {package_path}")


@dataclass(frozen=True, slots=True)
class _Relationship:
    """One internal OOXML relationship target."""

    target: str
    external: bool


@dataclass(frozen=True, slots=True)
class DocxImage:
    """One embedded image occurrence with content-addressed media bytes."""

    schema_version: str
    image_id: str
    source_name: str
    source_sha256: str
    sequence_index: int
    body_index: int
    source_ref: str
    parent_source_ref: str
    run_index: int
    image_index: int
    relationship_id: str
    package_path: str
    media_sha256: str
    mime_type: str
    asset_name: str
    media_bytes: bytes


@dataclass(frozen=True, slots=True)
class DocxImageExtraction:
    """Ordered image occurrences and their source audit metadata."""

    source_name: str
    source_sha256: str
    images: tuple[DocxImage, ...]


def extract_docx_images(input_path: Path) -> DocxImageExtraction:
    """Extract main-document images with native paragraph and table provenance."""
    with DocxPackage.open(input_path) as package:
        return extract_package_images(package)


def extract_package_images(package: DocxPackage) -> DocxImageExtraction:
    """Extract images from an already loaded DOCX package."""
    relationships = _read_relationships(package.archive)
    content_types = _read_content_types(package.archive)
    body = package.document.find(_BODY_TAG)
    if body is None:
        raise DocxImageExtractionError.missing_body(package.input_path)
    images = _extract_body_images(
        package.archive,
        body,
        package.source_name,
        package.source_sha256,
        relationships,
        content_types,
    )
    return DocxImageExtraction(
        source_name=package.source_name,
        source_sha256=package.source_sha256,
        images=tuple(images),
    )


def _extract_body_images(
    archive: zipfile.ZipFile,
    body: ElementTree.Element,
    source_name: str,
    source_sha256: str,
    relationships: Mapping[str, _Relationship],
    content_types: Mapping[str, str],
) -> list[DocxImage]:
    """Walk supported body structures in the same order as native source extraction."""
    images: list[DocxImage] = []
    for body_index, element in enumerate(body):
        parent_ref = f"{_DOCUMENT_PART}#/body/{body_index}"
        if element.tag == _PARAGRAPH_TAG:
            _append_paragraph_images(
                archive,
                element,
                parent_ref,
                body_index,
                source_name,
                source_sha256,
                relationships,
                content_types,
                images,
            )
        elif element.tag == _TABLE_TAG:
            _append_table_images(
                archive,
                element,
                parent_ref,
                body_index,
                source_name,
                source_sha256,
                relationships,
                content_types,
                images,
            )
    return images


def _append_table_images(
    archive: zipfile.ZipFile,
    table: ElementTree.Element,
    table_ref: str,
    body_index: int,
    source_name: str,
    source_sha256: str,
    relationships: Mapping[str, _Relationship],
    content_types: Mapping[str, str],
    images: list[DocxImage],
) -> None:
    """Append images from one table, including recursively nested tables."""
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
                    _append_paragraph_images(
                        archive,
                        child,
                        paragraph_ref,
                        body_index,
                        source_name,
                        source_sha256,
                        relationships,
                        content_types,
                        images,
                    )
                elif child.tag == _TABLE_TAG:
                    nested_ref = f"{cell_ref}/table/{table_ordinal}"
                    table_ordinal += 1
                    _append_table_images(
                        archive,
                        child,
                        nested_ref,
                        body_index,
                        source_name,
                        source_sha256,
                        relationships,
                        content_types,
                        images,
                    )


def _append_paragraph_images(
    archive: zipfile.ZipFile,
    paragraph: ElementTree.Element,
    parent_ref: str,
    body_index: int,
    source_name: str,
    source_sha256: str,
    relationships: Mapping[str, _Relationship],
    content_types: Mapping[str, str],
    images: list[DocxImage],
) -> None:
    """Append VML and DrawingML images from one native paragraph."""
    for run_index, run in enumerate(paragraph.iter(_RUN_TAG)):
        attachment_shape_ids = {
            element.get("ShapeID")
            for element in run.iter(_OLE_OBJECT_TAG)
            if element.get("Type") == "Embed" and element.get("ShapeID") is not None
        }
        image_index = 0
        for element in _iter_run_image_elements(run, attachment_shape_ids):
            relationship_id = _image_relationship_id(element)
            if relationship_id is None:  # pragma: no cover - iterator yields only supported elements
                continue
            sequence_index = len(images)
            source_ref = f"{parent_ref}/run/{run_index}/image/{image_index}"
            image_index += 1
            images.append(
                _build_image(
                    archive,
                    relationship_id,
                    source_name,
                    source_sha256,
                    sequence_index,
                    body_index,
                    source_ref,
                    parent_ref,
                    run_index,
                    image_index - 1,
                    relationships,
                    content_types,
                )
            )


def _iter_run_image_elements(
    run: ElementTree.Element,
    excluded_shape_ids: set[str],
) -> Iterable[ElementTree.Element]:
    """Yield non-attachment images owned by one run."""

    def walk(element: ElementTree.Element) -> Iterable[ElementTree.Element]:
        for child in element:
            if child.tag == _RUN_TAG:
                continue
            if child.tag == _SHAPE_TAG and child.get("id") in excluded_shape_ids:
                continue
            if child.tag in {_IMAGE_DATA_TAG, _BLIP_TAG}:
                yield child
            yield from walk(child)

    yield from walk(run)


def _image_relationship_id(element: ElementTree.Element) -> str | None:
    """Return an embedded relationship ID for a supported image element."""
    if element.tag == _IMAGE_DATA_TAG:
        return element.get(_RELATIONSHIP_ID_ATTRIBUTE)
    if element.tag == _BLIP_TAG:
        return element.get(_RELATIONSHIP_EMBED_ATTRIBUTE)
    return None


def _build_image(
    archive: zipfile.ZipFile,
    relationship_id: str,
    source_name: str,
    source_sha256: str,
    sequence_index: int,
    body_index: int,
    source_ref: str,
    parent_ref: str,
    run_index: int,
    image_index: int,
    relationships: Mapping[str, _Relationship],
    content_types: Mapping[str, str],
) -> DocxImage:
    """Resolve one image relationship and build deterministic occurrence metadata."""
    relationship = relationships.get(relationship_id)
    if relationship is None:
        raise DocxImageExtractionError.unknown_relationship(relationship_id)
    if relationship.external:
        raise DocxImageExtractionError.external_relationship(relationship_id)

    package_path = _resolve_package_path(relationship_id, relationship.target)
    media_bytes = archive.read(package_path)
    mime_type = _content_type(package_path, content_types)
    media_sha256 = hashlib.sha256(media_bytes).hexdigest()
    suffix = PurePosixPath(package_path).suffix.lower()
    asset_name = f"{media_sha256}{suffix}"
    image_id = sha256_json(
        {
            "media_sha256": media_sha256,
            "source_ref": source_ref,
            "source_sha256": source_sha256,
        }
    )
    return DocxImage(
        schema_version=IMAGE_SCHEMA_VERSION,
        image_id=image_id,
        source_name=source_name,
        source_sha256=source_sha256,
        sequence_index=sequence_index,
        body_index=body_index,
        source_ref=source_ref,
        parent_source_ref=parent_ref,
        run_index=run_index,
        image_index=image_index,
        relationship_id=relationship_id,
        package_path=package_path,
        media_sha256=media_sha256,
        mime_type=mime_type,
        asset_name=asset_name,
        media_bytes=media_bytes,
    )


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
        target = element.get("Target")
        if relationship_id is None or target is None:
            continue
        relationships[relationship_id] = _Relationship(
            target=target,
            external=element.get("TargetMode") == "External",
        )
    return relationships


def _read_content_types(archive: zipfile.ZipFile) -> dict[str, str]:
    """Read OOXML default and override content-type declarations."""
    root = SafeElementTree.fromstring(archive.read(_CONTENT_TYPES_PART))
    content_types: dict[str, str] = {}
    for element in root.findall(_CONTENT_TYPE_DEFAULT_TAG):
        extension = element.get("Extension")
        content_type = element.get("ContentType")
        if extension and content_type:
            content_types[f"*.{extension.lower()}"] = content_type
    for element in root.findall(_CONTENT_TYPE_OVERRIDE_TAG):
        part_name = element.get("PartName")
        content_type = element.get("ContentType")
        if part_name and content_type:
            content_types[part_name.removeprefix("/")] = content_type
    return content_types


def _resolve_package_path(relationship_id: str, target: str) -> str:
    """Resolve a main-document relationship target within the DOCX package."""
    package_path = posixpath.normpath(posixpath.join(posixpath.dirname(_DOCUMENT_PART), target))
    path = PurePosixPath(package_path)
    if path.is_absolute() or ".." in path.parts:
        raise DocxImageExtractionError.unsafe_target(relationship_id, target)
    return package_path


def _content_type(package_path: str, content_types: Mapping[str, str]) -> str:
    """Resolve and validate an image content type from OOXML declarations."""
    suffix = PurePosixPath(package_path).suffix.lower()
    content_type = content_types.get(package_path) or content_types.get(f"*{suffix}")
    if content_type is None:
        raise DocxImageExtractionError.unknown_content_type(package_path)
    if not content_type.startswith("image/"):
        raise DocxImageExtractionError.invalid_content_type(package_path, content_type)
    return content_type
