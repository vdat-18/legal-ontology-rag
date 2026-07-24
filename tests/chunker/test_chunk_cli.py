"""Tests for the end-to-end chunker command."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from legal_rag import chunk_cli
from legal_rag.embedding.profiles import EmbeddingProfile
from legal_rag.embedding.tokenizer import EmbeddingTokenizer


class CharacterTokenizer:
    """Fast deterministic tokenizer for CLI tests."""

    is_fast = True

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        truncation: bool = False,
        max_length: int | None = None,
    ) -> list[int]:
        """Encode each source character as one token."""
        token_ids = list(range(len(text)))
        if add_special_tokens:
            token_ids = [0, *token_ids, 1]
        return token_ids[:max_length] if truncation and max_length is not None else token_ids

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        """Return wrapper-token counts for single or paired input."""
        return 3 if pair else 2


def _tokenizer() -> EmbeddingTokenizer:
    """Build a tokenizer without model downloads."""
    profile = EmbeddingProfile(
        profile_id="test",
        model_id="test/model",
        model_revision="model-revision",
        tokenizer_id="test/tokenizer",
        tokenizer_revision="tokenizer-revision",
        dimensions=1024,
        max_input_tokens=4096,
        query_instruction=None,
    )
    return EmbeddingTokenizer(profile, CharacterTokenizer())


def _record(unit_id: str, sequence_index: int) -> dict[str, object]:
    """Build one parser paragraph record accepted by the chunker boundary."""
    return {
        "schema_version": "2.0.0",
        "unit_id": unit_id,
        "source_name": "source.docx",
        "source_sha256": "source-hash",
        "source_ref": f"word/document.xml#/body/{sequence_index}",
        "kind": "paragraph",
        "sequence_index": sequence_index,
        "text": unit_id,
    }


def _configure_cli(
    monkeypatch: pytest.MonkeyPatch,
    input_path: Path,
    output_path: Path,
) -> None:
    """Set deterministic CLI arguments, environment, and tokenizer loading."""
    monkeypatch.setenv("EMBEDDING_PROFILE", "qwen3_0_6b")
    monkeypatch.setenv("CHUNK_MAX_TOKENS", "64")
    monkeypatch.setenv("CHUNK_SPLIT_OVERLAP_TOKENS", "8")
    monkeypatch.setattr(chunk_cli.EmbeddingTokenizer, "from_profile", lambda profile: _tokenizer())
    monkeypatch.setattr(sys, "argv", ["legal-chunk", str(input_path), str(output_path)])


def test_chunk_cli_runs_end_to_end_and_overwrites_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The command reads parser JSONL, packs chunks, and replaces stale output."""
    input_path = tmp_path / "source.source.jsonl"
    output_path = tmp_path / "source.chunks.jsonl"
    records = (_record("Điều 1", 0), _record("Khoản 1", 1))
    input_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    output_path.write_text("stale output", encoding="utf-8")
    _configure_cli(monkeypatch, input_path, output_path)

    chunk_cli.main()

    chunks = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(chunks) == 1
    assert chunks[0]["schema_version"] == "1.0.0"
    assert chunks[0]["text"] == "Điều 1\n\nKhoản 1"
    assert chunks[0]["unit_ids"] == ["Điều 1", "Khoản 1"]


def test_chunk_cli_preserves_output_and_json_error_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Malformed input propagates its native error before the old output is replaced."""
    input_path = tmp_path / "invalid.source.jsonl"
    output_path = tmp_path / "source.chunks.jsonl"
    input_path.write_text("{invalid}\n", encoding="utf-8")
    output_path.write_text("previous output", encoding="utf-8")
    _configure_cli(monkeypatch, input_path, output_path)

    with pytest.raises(json.JSONDecodeError):
        chunk_cli.main()

    assert output_path.read_text(encoding="utf-8") == "previous output"
