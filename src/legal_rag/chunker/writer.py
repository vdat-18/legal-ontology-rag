"""Atomic persistence for final retrieval chunks."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from legal_rag.chunker.models import Chunk
from legal_rag.io_utils import write_jsonl_atomic


def write_chunks(output_path: Path, chunks: Sequence[Chunk]) -> None:
    """Atomically replace one UTF-8 JSONL output with deterministic chunk records."""
    write_jsonl_atomic(output_path, (chunk.to_dict() for chunk in chunks))
