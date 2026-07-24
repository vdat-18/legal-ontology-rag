"""Tests for parser JSONL scope isolation and ordering."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from legal_rag.chunker.reader import read_source_scopes


def _record(unit_id: str, sequence_index: int, *, child: bool = False) -> dict[str, object]:
    """Build one minimal parser record for reader tests."""
    record: dict[str, object] = {
        "schema_version": "2.0.0",
        "unit_id": unit_id,
        "source_name": "source.docx",
        "source_sha256": "source-hash",
        "source_ref": f"word/document.xml#/body/{sequence_index}",
        "kind": "paragraph",
        "sequence_index": sequence_index,
        "text": unit_id,
    }
    if child:
        record.update(
            {
                "source_name": "appendix.docx",
                "source_sha256": "member-hash",
                "record_scope": "attachment_child",
                "child_sequence_index": sequence_index,
                "root_source_name": "source.docx",
                "root_source_sha256": "source-hash",
                "parent_attachment_id": "attachment-id",
                "attachment_member_path": "appendix.docx",
                "attachment_member_sha256": "member-hash",
            }
        )
    return record


def test_read_source_scopes_separates_and_orders_root_and_attachment_records(tmp_path: Path) -> None:
    """Root and child records become separate streams ordered by their own indices."""
    input_path = tmp_path / "source.jsonl"
    records = (
        _record("root-1", 1),
        _record("child-1", 1, child=True),
        _record("root-0", 0),
        _record("child-0", 0, child=True),
    )
    input_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    scopes = read_source_scopes(input_path)

    assert len(scopes) == 2
    assert scopes[0].record_scope == "root"
    assert [record.unit_id for record in scopes[0].records] == ["root-0", "root-1"]
    assert scopes[0].parent_attachment_id is None
    assert scopes[1].record_scope == "attachment_child"
    assert [record.unit_id for record in scopes[1].records] == ["child-0", "child-1"]
    assert scopes[1].root_source_name == "source.docx"
    assert scopes[1].parent_attachment_id == "attachment-id"
    assert scopes[1].attachment_member_path == "appendix.docx"
    assert scopes[1].attachment_member_sha256 == "member-hash"
    assert scopes[0].scope_id != scopes[1].scope_id


def test_read_source_scopes_preserves_native_json_error(tmp_path: Path) -> None:
    """Malformed JSON raises its original decoder exception without wrapping."""
    input_path = tmp_path / "invalid.jsonl"
    input_path.write_text("{invalid}\n", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        read_source_scopes(input_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", "3.0.0", "Unsupported parser schema"),
        ("kind", "unknown", "Unsupported parser record kind"),
        ("record_scope", "unknown", "Unsupported parser record scope"),
    ],
)
def test_read_source_scopes_rejects_unsupported_contract_values(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    """Unsupported parser contract values fail at the JSONL boundary."""
    input_path = tmp_path / "unsupported.jsonl"
    record = _record("unit-0", 0)
    record[field] = value
    input_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        read_source_scopes(input_path)
