"""Tests for atomic chunk-output persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from legal_rag.chunker.models import Chunk
from legal_rag.chunker.writer import write_chunks


def _chunk(text: str) -> Chunk:
    """Build one final chunk for persistence tests."""
    return Chunk(
        chunk_id="chunk-id",
        content_sha256="content-hash",
        chunker_config_sha256="config-hash",
        scope_id="scope-id",
        chunk_index=0,
        kind="paragraph_group",
        source_name="source.docx",
        source_sha256="source-hash",
        record_scope="root",
        parent_attachment_id=None,
        attachment_member_path=None,
        attachment_member_sha256=None,
        unit_ids=("unit-0",),
        source_refs=("word/document.xml#/body/0",),
        sequence_start=0,
        sequence_end=0,
        text=text,
        context_text="",
        embedding_text=text,
        text_token_count=5,
        embedding_token_count=5,
        previous_chunk_id=None,
        next_chunk_id=None,
    )


@pytest.mark.parametrize("separator", ["\u0085", "\u2028", "\u2029"])
def test_write_chunks_atomically_overwrites_with_one_physical_line(tmp_path: Path, separator: str) -> None:
    """Successful writes replace stale output and escape JSONL line separators."""
    output_path = tmp_path / "chunks.jsonl"
    output_path.write_text("stale output", encoding="utf-8")
    text = f"Trước{separator}Sau"

    write_chunks(output_path, (_chunk(text),))

    serialized = output_path.read_text(encoding="utf-8")
    lines = serialized.splitlines()
    assert len(lines) == 1
    assert separator not in serialized
    assert json.loads(lines[0])["text"] == text


def test_write_chunks_preserves_old_output_and_original_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Serialization failure leaves the prior output intact and removes the temporary file."""
    output_path = tmp_path / "chunks.jsonl"
    output_path.write_text("previous output", encoding="utf-8")
    expected_error = RuntimeError()

    def fail_serialization(chunk: Chunk) -> dict[str, object]:
        raise expected_error

    monkeypatch.setattr(Chunk, "to_dict", fail_serialization)

    with pytest.raises(RuntimeError) as raised:
        write_chunks(output_path, (_chunk("Nội dung"),))

    assert raised.value is expected_error
    assert output_path.read_text(encoding="utf-8") == "previous output"
    assert list(tmp_path.iterdir()) == [output_path]
