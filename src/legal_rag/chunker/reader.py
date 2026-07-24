"""Read parser JSONL records into isolated document scopes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from legal_rag.chunker.models import (
    SOURCE_SCHEMA_VERSION,
    SUPPORTED_SOURCE_KINDS,
    SourceRecord,
    SourceScope,
)
from legal_rag.parser.hashing import sha256_json


def read_source_scopes(input_path: Path) -> tuple[SourceScope, ...]:
    """Read one parser JSONL file and return independently ordered source scopes."""
    grouped_records: dict[tuple[str, ...], list[SourceRecord]] = {}
    scope_metadata: dict[tuple[str, ...], dict[str, Any]] = {}

    with input_path.open(encoding="utf-8") as source_file:
        for line in source_file:
            payload = json.loads(line)
            if payload["schema_version"] != SOURCE_SCHEMA_VERSION:
                msg = f"Unsupported parser schema: {payload['schema_version']}"
                raise ValueError(msg)
            if payload["kind"] not in SUPPORTED_SOURCE_KINDS:
                msg = f"Unsupported parser record kind: {payload['kind']}"
                raise ValueError(msg)

            record_scope = payload.get("record_scope", "root")
            if record_scope == "root":
                order_index = payload["sequence_index"]
                root_source_name = payload["source_name"]
                root_source_sha256 = payload["source_sha256"]
                parent_attachment_id = None
                attachment_member_path = None
                attachment_member_sha256 = None
                scope_key = (record_scope, root_source_name, root_source_sha256)
            elif record_scope == "attachment_child":
                order_index = payload["child_sequence_index"]
                root_source_name = payload["root_source_name"]
                root_source_sha256 = payload["root_source_sha256"]
                parent_attachment_id = payload["parent_attachment_id"]
                attachment_member_path = payload["attachment_member_path"]
                attachment_member_sha256 = payload["attachment_member_sha256"]
                scope_key = (
                    record_scope,
                    root_source_name,
                    root_source_sha256,
                    parent_attachment_id,
                    attachment_member_path,
                    attachment_member_sha256,
                )
            else:
                msg = f"Unsupported parser record scope: {record_scope}"
                raise ValueError(msg)

            grouped_records.setdefault(scope_key, []).append(
                SourceRecord(
                    unit_id=payload["unit_id"],
                    source_name=payload["source_name"],
                    source_sha256=payload["source_sha256"],
                    source_ref=payload["source_ref"],
                    kind=payload["kind"],
                    record_scope=record_scope,
                    order_index=order_index,
                    payload=payload,
                )
            )
            scope_metadata.setdefault(
                scope_key,
                {
                    "source_name": payload["source_name"],
                    "source_sha256": payload["source_sha256"],
                    "record_scope": record_scope,
                    "root_source_name": root_source_name,
                    "root_source_sha256": root_source_sha256,
                    "parent_attachment_id": parent_attachment_id,
                    "attachment_member_path": attachment_member_path,
                    "attachment_member_sha256": attachment_member_sha256,
                },
            )

    scopes: list[SourceScope] = []
    for scope_key, records in grouped_records.items():
        metadata = scope_metadata[scope_key]
        scopes.append(
            SourceScope(
                scope_id=sha256_json({"scope": scope_key}),
                source_name=metadata["source_name"],
                source_sha256=metadata["source_sha256"],
                record_scope=metadata["record_scope"],
                root_source_name=metadata["root_source_name"],
                root_source_sha256=metadata["root_source_sha256"],
                parent_attachment_id=metadata["parent_attachment_id"],
                attachment_member_path=metadata["attachment_member_path"],
                attachment_member_sha256=metadata["attachment_member_sha256"],
                records=tuple(sorted(records, key=lambda record: record.order_index)),
            )
        )
    return tuple(scopes)
