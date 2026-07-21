"""Tests for the non-destructive source-block normalization view."""

from __future__ import annotations

import json
from pathlib import Path

from docling_core.types.doc import DocItemLabel, DoclingDocument

from legal_rag.normalization import select_content_blocks, write_normalized_source_blocks
from legal_rag.source_blocks import build_source_blocks


def _blocks():
    """Build source blocks containing structural, content, and empty items."""
    document = DoclingDocument(name="source")
    group = document.add_group(name="section")
    document.add_text(label=DocItemLabel.TEXT, text="Nội dung nguồn", parent=group)
    document.add_text(label=DocItemLabel.TEXT, text="", parent=document.body)
    document.add_text(label=DocItemLabel.TEXT, text="   \n", parent=document.body)
    return build_source_blocks(document, source_name="source.docx", source_sha256="source-hash")


def test_select_content_blocks_keeps_raw_blocks_unchanged() -> None:
    """Normalization skips empty text while retaining structural items."""
    raw_blocks = _blocks()
    raw_records = [block.to_dict() for block in raw_blocks]

    selection = select_content_blocks(raw_blocks)

    assert [block.to_dict() for block in raw_blocks] == raw_records
    assert selection.raw_block_count == len(raw_blocks)
    assert selection.skipped_empty_text_refs == ("#/texts/1", "#/texts/2")
    assert {block.docling_ref for block in selection.blocks} == {"#/body", "#/groups/0", "#/texts/0"}


def test_write_normalized_source_blocks_writes_manifest(tmp_path: Path) -> None:
    """Normalized JSONL and manifest report the same selection."""
    selection = select_content_blocks(_blocks())
    output_path = tmp_path / "normalized.jsonl"
    manifest_path = tmp_path / "manifest.json"

    write_normalized_source_blocks(output_path, manifest_path, selection)

    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(records) == len(selection.blocks)
    assert manifest == selection.manifest()
    assert manifest["skipped_empty_text_count"] == 2
