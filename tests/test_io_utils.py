"""Tests for the shared atomic JSONL writer."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from legal_rag.io_utils import write_jsonl_atomic

# NEL (U+0085), LINE SEPARATOR (U+2028), PARAGRAPH SEPARATOR (U+2029): the
# three characters splitlines() treats as line breaks but JSON does not escape.
_LINE_SEPARATOR_0085 = chr(0x0085)
_LINE_SEPARATOR_2028 = chr(0x2028)
_LINE_SEPARATOR_2029 = chr(0x2029)


@pytest.mark.parametrize("separator", [_LINE_SEPARATOR_0085, _LINE_SEPARATOR_2028, _LINE_SEPARATOR_2029])
def test_write_jsonl_atomic_overwrites_with_one_physical_line_per_record(tmp_path: Path, separator: str) -> None:
    """Successful writes replace stale output and escape JSONL line separators."""
    output_path = tmp_path / "records.jsonl"
    output_path.write_text("stale output", encoding="utf-8")
    text = f"Trước{separator}Sau"

    write_jsonl_atomic(output_path, ({"text": text}, {"text": "second"}))

    serialized = output_path.read_text(encoding="utf-8")
    lines = serialized.splitlines()
    assert len(lines) == 2
    assert separator not in serialized
    assert json.loads(lines[0])["text"] == text
    assert json.loads(lines[1])["text"] == "second"


def test_write_jsonl_atomic_preserves_old_output_and_original_error(tmp_path: Path) -> None:
    """A failure while iterating records leaves prior output intact and removes the temp file."""
    output_path = tmp_path / "records.jsonl"
    output_path.write_text("previous output", encoding="utf-8")
    expected_error = RuntimeError("boom")

    def failing_records() -> Iterator[dict[str, object]]:
        yield {"text": "ok"}
        raise expected_error

    with pytest.raises(RuntimeError) as raised:
        write_jsonl_atomic(output_path, failing_records())

    assert raised.value is expected_error
    assert output_path.read_text(encoding="utf-8") == "previous output"
    assert list(tmp_path.iterdir()) == [output_path]


def test_write_jsonl_atomic_creates_missing_parent_directories(tmp_path: Path) -> None:
    """Output directories are created on demand."""
    output_path = tmp_path / "nested" / "records.jsonl"

    write_jsonl_atomic(output_path, ({"a": 1},))

    assert json.loads(output_path.read_text(encoding="utf-8").splitlines()[0]) == {"a": 1}
