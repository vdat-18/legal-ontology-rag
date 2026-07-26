"""Atomic JSON Lines persistence shared across pipeline stages."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TextIO

_JSONL_LINE_SEPARATOR_ESCAPES = {
    0x0085: "\\u0085",
    0x2028: "\\u2028",
    0x2029: "\\u2029",
}


def write_jsonl_atomic(output_path: Path, records: Iterable[Mapping[str, object]]) -> None:
    """Atomically replace one UTF-8 JSONL output with the given records."""
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
            for record in records:
                _write_jsonl_record(temporary_file, record)
        temporary_path.replace(output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _write_jsonl_record(temporary_file: TextIO, record: Mapping[str, object]) -> None:
    """Serialize one record as exactly one physical JSONL line."""
    serialized = json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    temporary_file.write(serialized.translate(_JSONL_LINE_SEPARATOR_ESCAPES) + "\n")
