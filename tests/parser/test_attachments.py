"""Tests for embedded DOCX attachment extraction."""

from __future__ import annotations

import hashlib
import struct
import zipfile
from pathlib import Path

import pytest

from legal_rag.parser import attachments
from legal_rag.parser.attachments import (
    DocxAttachmentExtractionError,
    extract_docx_attachments,
)
from legal_rag.parser.images import extract_docx_images

_DOCUMENT_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document
    xmlns:o="urn:schemas-microsoft-com:office:office"
    xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    xmlns:v="urn:schemas-microsoft-com:vml"
    xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:tbl>
      <w:tr>
        <w:tc>
          <w:p>
            <w:r>
              <w:object>
                <v:shape id="attachment-shape">
                  <v:imagedata r:id="rIdIcon"/>
                </v:shape>
                <o:OLEObject Type="Embed" ProgID="Package" ShapeID="attachment-shape" r:id="rIdAttachment"/>
              </w:object>
            </w:r>
          </w:p>
        </w:tc>
      </w:tr>
    </w:tbl>
  </w:body>
</w:document>
"""

_RELATIONSHIPS_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship
      Id="rIdAttachment"
      Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/oleObject"
      Target="embeddings/oleObject1.bin"/>
  <Relationship Id="rIdIcon" Type="image" Target="media/icon.png"/>
</Relationships>
"""

_CONTENT_TYPES_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="png" ContentType="image/png"/>
  <Default Extension="xml" ContentType="application/xml"/>
</Types>
"""

_OLE_BYTES = b"OLE wrapper"
_PAYLOAD_BYTES = b"PK attachment payload"


def _write_docx(path: Path, *, relationships_xml: str = _RELATIONSHIPS_XML) -> None:
    """Write a DOCX fixture with one OLE attachment and its display icon."""
    with zipfile.ZipFile(path, mode="w") as archive:
        archive.writestr("word/document.xml", _DOCUMENT_XML)
        archive.writestr("word/_rels/document.xml.rels", relationships_xml)
        archive.writestr("[Content_Types].xml", _CONTENT_TYPES_XML)
        archive.writestr("word/embeddings/oleObject1.bin", _OLE_BYTES)
        archive.writestr("word/media/icon.png", b"icon")


def test_extract_docx_attachments_preserves_payload_and_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An OLE relationship produces one content-addressed attachment record."""
    input_path = tmp_path / "source.docx"
    _write_docx(input_path)

    def read_ole_package(ole_bytes: bytes) -> tuple[str, bytes]:
        assert ole_bytes == _OLE_BYTES
        return "Bieu mau.zip", _PAYLOAD_BYTES

    monkeypatch.setattr(attachments, "_read_ole_package", read_ole_package)

    extraction = extract_docx_attachments(input_path)

    attachment = extraction.attachments[0]
    expected_hash = hashlib.sha256(_PAYLOAD_BYTES).hexdigest()
    assert attachment.filename == "Bieu mau.zip"
    assert attachment.mime_type == "application/zip"
    assert attachment.payload_sha256 == expected_hash
    assert attachment.asset_path == f"assets/{expected_hash}"
    assert attachment.payload_bytes == _PAYLOAD_BYTES
    assert attachment.source_ref == ("word/document.xml#/body/0/row/0/cell/0/paragraph/0/run/0/attachment/0")


def test_ole_attachment_icon_is_not_extracted_as_an_image(tmp_path: Path) -> None:
    """A VML shape linked to an OLE object does not consume image-caption quota."""
    input_path = tmp_path / "source.docx"
    _write_docx(input_path)

    extraction = extract_docx_images(input_path)

    assert extraction.images == ()


def test_parse_ole10_native_returns_only_the_declared_payload() -> None:
    """Trailing OLE metadata is excluded using the native payload length field."""
    body = (
        struct.pack("<H", 2)
        + b"Bieu mau.zip\0"
        + b"C:\\Source\\Bieu mau.zip\0"
        + struct.pack("<II", 0, 0)
        + b"C:\\Temp\\Bieu mau.zip\0"
        + struct.pack("<I", len(_PAYLOAD_BYTES))
        + _PAYLOAD_BYTES
        + "Bieu mau.zip".encode("utf-16-le")
    )
    native_bytes = struct.pack("<I", len(body)) + body

    filename, payload = attachments._parse_ole10_native(native_bytes)

    assert filename == "Bieu mau.zip"
    assert payload == _PAYLOAD_BYTES


def test_parse_ole10_native_rejects_a_truncated_payload() -> None:
    """A payload length beyond the OLE stream fails at the document boundary."""
    body = (
        struct.pack("<H", 2)
        + b"attachment.zip\0"
        + b"source\0"
        + struct.pack("<II", 0, 0)
        + b"temp\0"
        + struct.pack("<I", len(_PAYLOAD_BYTES) + 1)
        + _PAYLOAD_BYTES
    )
    native_bytes = struct.pack("<I", len(body)) + body

    with pytest.raises(DocxAttachmentExtractionError, match="malformed Ole10Native"):
        attachments._parse_ole10_native(native_bytes)


def test_extract_docx_attachments_rejects_external_relationships(tmp_path: Path) -> None:
    """Externally linked OLE objects are never fetched as embedded evidence."""
    relationships_xml = _RELATIONSHIPS_XML.replace(
        'Target="embeddings/oleObject1.bin"/>',
        'Target="https://example.invalid/attachment.zip" TargetMode="External"/>',
    )
    input_path = tmp_path / "source.docx"
    _write_docx(input_path, relationships_xml=relationships_xml)

    with pytest.raises(DocxAttachmentExtractionError, match="external"):
        extract_docx_attachments(input_path)
