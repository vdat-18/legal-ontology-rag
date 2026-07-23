"""Tests for bounded attachment child-document ingestion."""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import pytest

from legal_rag.parser import attachment_documents
from legal_rag.parser.attachment_documents import (
    AttachmentDocumentIngestionError,
    ingest_attachment_documents,
)
from legal_rag.parser.attachments import ATTACHMENT_SCHEMA_VERSION, DocxAttachment
from legal_rag.parser.document_source import extract_document_source
from legal_rag.parser.models import ParsedDocument
from legal_rag.settings import Settings


def _settings(**overrides: int) -> Settings:
    """Build parser settings without environment access."""
    values = {
        "llm_base_url": "http://localhost/v1",
        "llm_api_key": "test-key",
        "llm_model": "test-model",
        "llm_timeout_seconds": 30,
        **overrides,
    }
    return Settings(**values)


def _docx_bytes(text: str = "Noi dung tai lieu con") -> bytes:
    """Build a minimal content-bearing DOCX package."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w") as archive:
        archive.writestr(
            "word/document.xml",
            f"""<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
              <w:body><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:body>
            </w:document>""",
        )
        archive.writestr("[Content_Types].xml", "<Types/>")
    return buffer.getvalue()


def _attachment(payload: bytes, filename: str = "forms.zip") -> DocxAttachment:
    """Build one extracted attachment around test payload bytes."""
    payload_hash = hashlib.sha256(payload).hexdigest()
    return DocxAttachment(
        schema_version=ATTACHMENT_SCHEMA_VERSION,
        attachment_id="attachment-0",
        source_name="parent.docx",
        source_sha256="parent-hash",
        body_index=0,
        source_ref="word/document.xml#/body/0/run/0/attachment/0",
        parent_source_ref="word/document.xml#/body/0",
        relationship_id="rId1",
        package_path="word/embeddings/oleObject1.bin",
        ole_sha256="ole-hash",
        filename=filename,
        mime_type="application/zip",
        size_bytes=len(payload),
        payload_sha256=payload_hash,
        asset_path=f"assets/{payload_hash}",
        payload_bytes=payload,
    )


def _parse_docx(path: Path) -> ParsedDocument:
    """Parse native child text without invoking an external caption model."""
    return ParsedDocument.from_source(extract_document_source(path), ())


def test_ingest_attachment_documents_parses_docx_members_with_provenance() -> None:
    """A supported ZIP member becomes searchable child records."""
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, mode="w") as archive:
        archive.writestr("forms/report.docx", _docx_bytes())
        archive.writestr("notes.txt", "ignored")

    documents = ingest_attachment_documents(
        (_attachment(archive_buffer.getvalue()),),
        _settings(),
        _parse_docx,
        renderer_executable=None,
    )

    assert len(documents) == 1
    assert documents[0].member_path == "forms/report.docx"
    records = documents[0].records("parent.docx", "parent-hash")
    assert records[0]["text"] == "Noi dung tai lieu con"
    assert records[0]["record_scope"] == "attachment_child"
    assert records[0]["parent_attachment_id"] == "attachment-0"
    assert records[0]["source_name"] == "forms/report.docx"
    assert records[0]["source_schema_version"] == "1.1.0"
    assert records[0]["child_parser_schema_version"] == "1.2.0"


def test_ingest_attachment_documents_rejects_unsafe_member_paths() -> None:
    """ZIP traversal syntax fails before member bytes reach a parser."""
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, mode="w") as archive:
        archive.writestr("../report.docx", _docx_bytes())

    with pytest.raises(AttachmentDocumentIngestionError, match="unsafe member path"):
        ingest_attachment_documents(
            (_attachment(archive_buffer.getvalue()),),
            _settings(),
            _parse_docx,
            renderer_executable=None,
        )


def test_ingest_attachment_documents_enforces_member_count_limit() -> None:
    """Archive expansion stops at the configured resource boundary."""
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, mode="w") as archive:
        archive.writestr("one.docx", _docx_bytes("one"))
        archive.writestr("two.docx", _docx_bytes("two"))

    with pytest.raises(AttachmentDocumentIngestionError, match="member limit"):
        ingest_attachment_documents(
            (_attachment(archive_buffer.getvalue()),),
            _settings(attachment_archive_max_members=1),
            _parse_docx,
            renderer_executable=None,
        )


def test_detect_format_checks_ole_before_embedded_zip_signatures(monkeypatch: pytest.MonkeyPatch) -> None:
    """A legacy DOC containing ZIP-like bytes remains an OLE Word document."""

    class FakeOleFile:
        """Expose the structural WordDocument stream used for format identity."""

        def __enter__(self) -> FakeOleFile:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def listdir(self, *, streams: bool, storages: bool) -> list[list[str]]:
            assert streams is True
            assert storages is False
            return [["WordDocument"]]

    monkeypatch.setattr(attachment_documents.olefile, "isOleFile", lambda source: True)
    monkeypatch.setattr(attachment_documents.olefile, "OleFileIO", lambda source: FakeOleFile())
    monkeypatch.setattr(attachment_documents.zipfile, "is_zipfile", lambda source: True)

    assert attachment_documents._detect_format(b"compound file with embedded zip") == "doc"


def test_legacy_documents_are_converted_in_one_batch_per_format(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Multiple legacy Word members share one bounded LibreOffice invocation."""
    attachment = _attachment(b"payload", "forms.doc")
    sources = [
        attachment_documents._ChildSource(attachment, index, f"form-{index}.doc", bytes([index]), "doc")
        for index in range(2)
    ]
    batches: list[int] = []

    def convert_batch(
        renderer: Path,
        profile_directory: Path,
        output_directory: Path,
        input_paths: list[Path],
        source_format: str,
        target_format: str,
        timeout_seconds: float,
    ) -> None:
        assert renderer == tmp_path.resolve()
        assert profile_directory.is_dir()
        assert source_format == "doc"
        assert target_format == "docx"
        assert timeout_seconds == 120
        batches.append(len(input_paths))
        for input_path in input_paths:
            (output_directory / f"{input_path.stem}.docx").write_bytes(_docx_bytes(input_path.stem))

    monkeypatch.setattr(attachment_documents, "_run_office_conversion", convert_batch)

    converted = attachment_documents._convert_legacy_documents(sources, _settings(), tmp_path)

    assert batches == [2]
    assert set(converted) == {0, 1}
