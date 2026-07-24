"""Atomic persistence for final retrieval chunks."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Sequence
from pathlib import Path

from legal_rag.chunker.models import Chunk

_JSONL_LINE_SEPARATOR_ESCAPES = {
    ord("\u0085"): "\\u0085",
    ord("\u2028"): "\\u2028",
    ord("\u2029"): "\\u2029",
}


def write_chunks(output_path: Path, chunks: Sequence[Chunk]) -> None:
    """Atomically replace one UTF-8 JSONL output with deterministic chunk records."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            for chunk in chunks:
                serialized = json.dumps(chunk.to_dict(), ensure_ascii=False, separators=(",", ":"), sort_keys=True)
                temporary_file.write(serialized.translate(_JSONL_LINE_SEPARATOR_ESCAPES) + "\n")
        temporary_path.replace(output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
