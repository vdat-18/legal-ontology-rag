"""Tests for parser-output persistence."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from legal_rag.parser.attachments import ATTACHMENT_SCHEMA_VERSION, DocxAttachment
from legal_rag.parser.document_source import KIND_PARAGRAPH, DocumentSource, DocumentSourceUnit
from legal_rag.parser.models import AttachmentChildDocument, ParsedDocument
from legal_rag.parser.writer import write_parsed_document


@pytest.mark.parametrize("separator", ["\u0085", "\u2028", "\u2029"])
def test_write_parsed_document_escapes_unicode_line_separators(tmp_path: Path, separator: str) -> None:
    """Unicode line separators remain data without splitting one JSONL record."""
    text = f"Trước{separator}Sau"
    source_unit = DocumentSourceUnit(
        schema_version="1.1.0",
        unit_id="unit-0",
        source_name="source.docx",
        source_sha256="source-hash",
        sequence_index=0,
        body_index=0,
        source_ref="word/document.xml#/body/0",
        kind=KIND_PARAGRAPH,
        payload_sha256="payload-hash",
        text=text,
        runs=(),
        rows=None,
    )
    document = ParsedDocument.from_source(DocumentSource("source.docx", "source-hash", (source_unit,)), ())
    output_path = tmp_path / "parsed.jsonl"

    write_parsed_document(output_path, document)

    serialized = output_path.read_text(encoding="utf-8")
    lines = serialized.splitlines()
    assert len(lines) == 1
    assert separator not in serialized
    assert json.loads(lines[0])["text"] == text


def test_write_parsed_document_persists_attachment_assets(tmp_path: Path) -> None:
    """Attachment bytes are stored beside JSONL under their content hash."""
    payload = b"attachment payload"
    payload_hash = sha256(payload).hexdigest()
    source_unit = DocumentSourceUnit(
        schema_version="1.1.0",
        unit_id="unit-0",
        source_name="source.docx",
        source_sha256="source-hash",
        sequence_index=0,
        body_index=0,
        source_ref="word/document.xml#/body/0",
        kind=KIND_PARAGRAPH,
        payload_sha256="source-payload-hash",
        text="Tệp đính kèm",
        runs=(),
        rows=None,
    )
    attachment = DocxAttachment(
        schema_version=ATTACHMENT_SCHEMA_VERSION,
        attachment_id="attachment-0",
        source_name="source.docx",
        source_sha256="source-hash",
        body_index=0,
        source_ref="word/document.xml#/body/0/run/0/attachment/0",
        parent_source_ref="word/document.xml#/body/0",
        relationship_id="rId1",
        package_path="word/embeddings/oleObject1.bin",
        ole_sha256="ole-hash",
        filename="Phu luc.zip",
        mime_type="application/zip",
        size_bytes=len(payload),
        payload_sha256=payload_hash,
        asset_path=f"assets/{payload_hash}",
        payload_bytes=payload,
    )
    document = ParsedDocument.from_source(
        DocumentSource("source.docx", "source-hash", (source_unit,)),
        (),
        (attachment,),
    )
    output_path = tmp_path / "parsed.jsonl"

    write_parsed_document(output_path, document)

    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert record["attachments"][0]["asset_path"] == f"assets/{payload_hash}"
    assert (tmp_path / "assets" / payload_hash).read_bytes() == payload


def test_write_parsed_document_places_child_records_after_their_parent(tmp_path: Path) -> None:
    """Child records follow the exact parent unit that owns their attachment."""
    payload = b"attachment payload"
    payload_hash = sha256(payload).hexdigest()
    source_unit = DocumentSourceUnit(
        schema_version="1.1.0",
        unit_id="unit-0",
        source_name="source.docx",
        source_sha256="source-hash",
        sequence_index=0,
        body_index=0,
        source_ref="word/document.xml#/body/0",
        kind=KIND_PARAGRAPH,
        payload_sha256="source-payload-hash",
        text="Tep dinh kem",
        runs=(),
        rows=None,
    )
    attachment = DocxAttachment(
        schema_version=ATTACHMENT_SCHEMA_VERSION,
        attachment_id="attachment-0",
        source_name="source.docx",
        source_sha256="source-hash",
        body_index=0,
        source_ref="word/document.xml#/body/0/run/0/attachment/0",
        parent_source_ref="word/document.xml#/body/0",
        relationship_id="rId1",
        package_path="word/embeddings/oleObject1.bin",
        ole_sha256="ole-hash",
        filename="forms.zip",
        mime_type="application/zip",
        size_bytes=len(payload),
        payload_sha256=payload_hash,
        asset_path=f"assets/{payload_hash}",
        payload_bytes=payload,
    )
    child_document = AttachmentChildDocument(
        parent_attachment_id="attachment-0",
        member_index=0,
        member_path="form.docx",
        member_sha256="member-hash",
        source_format="docx",
        converted_from_format=None,
        native_records=(
            {
                "schema_version": "1.1.0",
                "unit_id": "child-unit",
                "source_name": "source.docx",
                "source_sha256": "native-hash",
                "sequence_index": 0,
                "source_ref": "word/document.xml#/body/0",
                "kind": "paragraph",
                "payload_sha256": "child-payload-hash",
                "text": "Noi dung con",
                "runs": [],
                "images": [],
                "attachments": [],
            },
        ),
    )
    document = ParsedDocument.from_source(
        DocumentSource("source.docx", "source-hash", (source_unit,)),
        (),
        (attachment,),
        (child_document,),
    )
    output_path = tmp_path / "parsed.jsonl"

    write_parsed_document(output_path, document)

    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert [record.get("record_scope", "root") for record in records] == ["root", "attachment_child"]
    assert records[1]["parent_attachment_id"] == "attachment-0"
