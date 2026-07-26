"""Persistence for final parser output."""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from pathlib import Path, PurePosixPath

from legal_rag.io_utils import write_jsonl_atomic
from legal_rag.parser.attachments import DocxAttachment
from legal_rag.parser.models import AttachmentChildDocument, ParsedDocument


def write_parsed_document(output_path: Path, document: ParsedDocument) -> None:
    """Atomically write enriched source units as UTF-8 JSON Lines."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for unit in document.units:
        for attachment in unit.attachments:
            _write_attachment_asset(output_path.parent, attachment)
    for child_document in document.child_documents:
        for attachment in child_document.attachments:
            _write_attachment_asset(output_path.parent, attachment)
    write_jsonl_atomic(output_path, _ordered_records(document))


def _ordered_records(document: ParsedDocument) -> Iterator[dict[str, object]]:
    """Yield parent units and their attachment-child records in stable order."""
    children_by_attachment: dict[str, list[AttachmentChildDocument]] = {}
    for child_document in document.child_documents:
        children_by_attachment.setdefault(child_document.parent_attachment_id, []).append(child_document)
    for unit in document.units:
        yield unit.to_dict()
        for attachment in unit.attachments:
            for child_document in children_by_attachment.get(attachment.attachment_id, ()):
                yield from child_document.records(document.source_name, document.source_sha256)


def _write_attachment_asset(output_directory: Path, attachment: DocxAttachment) -> None:
    """Atomically persist one content-addressed attachment payload."""
    asset_path = output_directory.joinpath(*PurePosixPath(attachment.asset_path).parts)
    asset_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=asset_path.parent, delete=False) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(attachment.payload_bytes)
        temporary_path.replace(asset_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
