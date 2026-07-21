"""Tests for deterministic Docling source-block extraction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from docling_core.types.doc import DocItemLabel, DoclingDocument
from docling_core.types.io import DocumentStream

from legal_rag import source_blocks
from legal_rag.source_blocks import build_source_blocks, extract_source_blocks, write_source_blocks


def _document() -> DoclingDocument:
    """Create a small Docling document without semantic assumptions."""
    document = DoclingDocument(name="source")
    group = document.add_group(name="section")
    document.add_text(label=DocItemLabel.TEXT, text="Nội dung nguồn", parent=group)
    return document


def test_build_source_blocks_preserves_docling_items_deterministically() -> None:
    """The same document and provenance produce identical records."""
    first = build_source_blocks(_document(), source_name="source.docx", source_sha256="source-hash")
    second = build_source_blocks(_document(), source_name="source.docx", source_sha256="source-hash")

    assert first == second
    assert len({block.block_id for block in first}) == len(first)
    text_block = next(block for block in first if block.docling_ref == "#/texts/0")
    assert text_block.docling_item_type == "TextItem"
    assert text_block.docling_label == "text"
    assert text_block.payload["text"] == "Nội dung nguồn"


def test_extract_source_blocks_hashes_the_exact_converted_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Extraction hashes the same bytes passed to Docling."""
    source_bytes = b"source bytes"
    input_path = tmp_path / "source.docx"
    input_path.write_bytes(source_bytes)
    converted_bytes: list[bytes] = []

    class FakeConverter:
        """Capture the Docling stream at the converter boundary."""

        def convert(self, source: DocumentStream) -> SimpleNamespace:
            """Return a converted document for the captured source bytes."""
            converted_bytes.append(source.stream.read())
            return SimpleNamespace(document=_document())

    monkeypatch.setattr(source_blocks, "DocumentConverter", FakeConverter)

    blocks = extract_source_blocks(input_path)

    expected_sha256 = hashlib.sha256(source_bytes).hexdigest()
    assert converted_bytes == [source_bytes]
    assert {block.source_sha256 for block in blocks} == {expected_sha256}
    assert {block.source_name for block in blocks} == {input_path.name}


def test_extract_source_blocks_rejects_missing_input(tmp_path: Path) -> None:
    """A missing source fails before Docling conversion."""
    missing_path = tmp_path / "missing.docx"

    with pytest.raises(FileNotFoundError, match="Source document does not exist"):
        extract_source_blocks(missing_path)


def test_write_source_blocks_writes_valid_jsonl(tmp_path: Path) -> None:
    """Every output line is one complete source-block record."""
    blocks = build_source_blocks(_document(), source_name="source.docx", source_sha256="source-hash")
    output_path = tmp_path / "nested" / "blocks.jsonl"

    write_source_blocks(output_path, blocks)

    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert records == [block.to_dict() for block in blocks]
