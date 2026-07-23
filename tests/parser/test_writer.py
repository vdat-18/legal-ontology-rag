"""Tests for parser-output persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from legal_rag.parser.document_source import KIND_PARAGRAPH, DocumentSource, DocumentSourceUnit
from legal_rag.parser.models import ParsedDocument
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
