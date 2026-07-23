"""Persistence for final parser output."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from legal_rag.parser.models import ParsedDocument

_JSONL_LINE_SEPARATOR_ESCAPES = {
    ord("\u0085"): "\\u0085",
    ord("\u2028"): "\\u2028",
    ord("\u2029"): "\\u2029",
}


def write_parsed_document(output_path: Path, document: ParsedDocument) -> None:
    """Atomically write enriched source units as UTF-8 JSON Lines."""
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
            for unit in document.units:
                record = json.dumps(unit.to_dict(), ensure_ascii=False, separators=(",", ":"), sort_keys=True)
                temporary_file.write(record.translate(_JSONL_LINE_SEPARATOR_ESCAPES) + "\n")
        temporary_path.replace(output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
