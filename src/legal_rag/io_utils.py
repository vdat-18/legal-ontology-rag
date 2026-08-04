"""Atomic JSON Lines persistence shared across pipeline stages."""

from __future__ import annotations

import json
import tempfile
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TextIO

_JSONL_LINE_SEPARATOR_ESCAPES = {
    0x0085: "\\u0085",
    0x2028: "\\u2028",
    0x2029: "\\u2029",
}

_REPLACE_RETRY_DELAYS_SECONDS = (0.1, 0.2, 0.4, 0.8, 1.6)


def _replace_with_retry(source: Path, destination: Path) -> None:
    """Replace destination with source, retrying past transient Windows file locks.

    On Windows, antivirus/indexer/editor processes can briefly hold an open handle on the
    destination file, making os.replace raise PermissionError (WinError 5) even though nothing
    is actually wrong with the write. This is especially likely for outputs rewritten in a tight
    loop, such as per-pair checkpoint files. Retrying with a short backoff clears almost all of
    these without masking a genuine permissions problem, since we still raise after exhausting it.
    """
    for delay in (*_REPLACE_RETRY_DELAYS_SECONDS, None):
        try:
            source.replace(destination)
        except PermissionError:
            if delay is None:
                raise
            time.sleep(delay)
        else:
            return


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
        _replace_with_retry(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def write_json_atomic(output_path: Path, record: Mapping[str, object]) -> None:
    """Atomically replace one UTF-8 JSON file with a stable serialized record."""
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
            json.dump(record, temporary_file, ensure_ascii=False, indent=2, sort_keys=True)
            temporary_file.write("\n")
        _replace_with_retry(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _write_jsonl_record(temporary_file: TextIO, record: Mapping[str, object]) -> None:
    """Serialize one record as exactly one physical JSONL line."""
    serialized = json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    temporary_file.write(serialized.translate(_JSONL_LINE_SEPARATOR_ESCAPES) + "\n")
