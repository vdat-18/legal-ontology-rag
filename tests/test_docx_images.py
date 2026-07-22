"""Tests for native DOCX image extraction and persisted assets."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest

from legal_rag.docx_images import (
    DocxImageExtractionError,
    extract_docx_images,
)

_DOCUMENT_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document
    xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
    xmlns:o="urn:schemas-microsoft-com:office:office"
    xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"
    xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    xmlns:v="urn:schemas-microsoft-com:vml"
    xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing">
  <w:body>
    <w:p>
      <w:r>
        <w:pict><v:shape><v:imagedata r:id="rIdVml" o:title="Biển báo"/></v:shape></w:pict>
      </w:r>
    </w:p>
    <w:tbl>
      <w:tr>
        <w:tc>
          <w:p><w:r><w:t>Cell text</w:t></w:r></w:p>
          <w:tbl>
            <w:tr>
              <w:tc>
                <w:p>
                  <w:r>
                    <w:drawing>
                      <wp:inline>
                        <wp:docPr id="1" name="Vehicle" descr="Xe minh họa"/>
                        <a:graphic><a:graphicData><pic:pic><pic:blipFill>
                          <a:blip r:embed="rIdDrawing"/>
                        </pic:blipFill></pic:pic></a:graphicData></a:graphic>
                      </wp:inline>
                    </w:drawing>
                  </w:r>
                </w:p>
              </w:tc>
            </w:tr>
          </w:tbl>
        </w:tc>
      </w:tr>
    </w:tbl>
    <w:sectPr/>
  </w:body>
</w:document>
"""

_RELATIONSHIPS_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rIdVml" Type="image" Target="media/sign.png"/>
  <Relationship Id="rIdDrawing" Type="image" Target="media/vehicle.jpeg"/>
</Relationships>
"""

_CONTENT_TYPES_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="png" ContentType="image/png"/>
  <Default Extension="jpeg" ContentType="image/jpeg"/>
  <Default Extension="xml" ContentType="application/xml"/>
</Types>
"""

_SIGN_BYTES = b"png image bytes"
_VEHICLE_BYTES = b"jpeg image bytes"


def _write_docx(path: Path, *, relationships_xml: str = _RELATIONSHIPS_XML) -> None:
    """Write a DOCX fixture with VML and nested-table DrawingML images."""
    with zipfile.ZipFile(path, mode="w") as archive:
        archive.writestr("word/document.xml", _DOCUMENT_XML)
        archive.writestr("word/_rels/document.xml.rels", relationships_xml)
        archive.writestr("[Content_Types].xml", _CONTENT_TYPES_XML)
        archive.writestr("word/media/sign.png", _SIGN_BYTES)
        archive.writestr("word/media/vehicle.jpeg", _VEHICLE_BYTES)


def test_extract_docx_images_preserves_native_order_and_provenance(tmp_path: Path) -> None:
    """VML and DrawingML occurrences retain exact paragraph and table locations."""
    input_path = tmp_path / "source.docx"
    _write_docx(input_path)

    extraction = extract_docx_images(input_path)

    assert [image.sequence_index for image in extraction.images] == [0, 1]
    assert [image.mime_type for image in extraction.images] == ["image/png", "image/jpeg"]
    assert [image.alt_text for image in extraction.images] == ["Biển báo", "Xe minh họa"]
    assert extraction.images[0].source_ref == "word/document.xml#/body/0/run/0/image/0"
    assert extraction.images[1].source_ref == (
        "word/document.xml#/body/1/row/0/cell/0/table/0/row/0/cell/0/paragraph/0/run/0/image/0"
    )
    assert extraction.images[1].parent_source_ref.endswith("/paragraph/0")
    assert extraction.images[0].media_bytes == _SIGN_BYTES


def test_extract_docx_images_rejects_external_media(tmp_path: Path) -> None:
    """Externally linked images are not fetched or treated as archived evidence."""
    relationships_xml = _RELATIONSHIPS_XML.replace(
        'Id="rIdVml" Type="image" Target="media/sign.png"',
        'Id="rIdVml" Type="image" Target="https://example.invalid/sign.png" TargetMode="External"',
    )
    input_path = tmp_path / "source.docx"
    _write_docx(input_path, relationships_xml=relationships_xml)

    with pytest.raises(DocxImageExtractionError, match="external"):
        extract_docx_images(input_path)


def test_asset_names_are_content_addressed(tmp_path: Path) -> None:
    """Asset filenames derive only from exact bytes and their package extension."""
    input_path = tmp_path / "source.docx"
    _write_docx(input_path)

    extraction = extract_docx_images(input_path)

    expected = hashlib.sha256(_SIGN_BYTES).hexdigest()
    assert extraction.images[0].asset_name == f"{expected}.png"


def test_document_without_images_does_not_require_relationships_part(tmp_path: Path) -> None:
    """A valid text-only DOCX produces an empty image sidecar."""
    document_xml = _DOCUMENT_XML.replace(
        '<w:pict><v:shape><v:imagedata r:id="rIdVml" o:title="Biển báo"/></v:shape></w:pict>',
        "<w:t>Text only</w:t>",
    ).replace('<a:blip r:embed="rIdDrawing"/>', "")
    input_path = tmp_path / "text-only.docx"
    with zipfile.ZipFile(input_path, mode="w") as archive:
        archive.writestr("word/document.xml", document_xml)
        archive.writestr("[Content_Types].xml", _CONTENT_TYPES_XML)

    extraction = extract_docx_images(input_path)

    assert extraction.images == ()
