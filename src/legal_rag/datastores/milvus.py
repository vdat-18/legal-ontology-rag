"""Connect to the shared Milvus database."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from pymilvus import MilvusClient

from legal_rag.settings import MilvusSettings

_CORPUS_FINGERPRINT_VERSION = "1"


@dataclass(frozen=True, slots=True)
class MilvusCorpusFingerprint:
    """Verified identity of the non-vector rows in one active Milvus collection."""

    fingerprint_version: str
    collection_name: str
    row_count: int
    corpus_sha256: str
    embedding_profile_id: str
    embedding_model_id: str
    embedding_model_revision: str

    def to_dict(self) -> dict[str, str | int]:
        """Return a stable manifest-ready representation."""
        return asdict(self)


def build_milvus_corpus_fingerprint(
    collection_name: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    embedding_profile_id: str,
    embedding_model_id: str,
    embedding_model_revision: str,
) -> MilvusCorpusFingerprint:
    """Hash verified non-vector collection rows in primary-key order."""
    ordered_rows = sorted((dict(row) for row in rows), key=lambda row: str(row["chunk_id"]))
    encoded = json.dumps(
        {"fingerprint_version": _CORPUS_FINGERPRINT_VERSION, "rows": ordered_rows},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return MilvusCorpusFingerprint(
        fingerprint_version=_CORPUS_FINGERPRINT_VERSION,
        collection_name=collection_name,
        row_count=len(ordered_rows),
        corpus_sha256=hashlib.sha256(encoded).hexdigest(),
        embedding_profile_id=embedding_profile_id,
        embedding_model_id=embedding_model_id,
        embedding_model_revision=embedding_model_revision,
    )


def build_milvus_client(settings: MilvusSettings) -> MilvusClient:
    """Connect to Milvus and select the configured shared database, creating it if absent."""
    connection_options = {"uri": settings.uri}
    if settings.token is not None:
        connection_options["token"] = settings.token
    client = MilvusClient(**connection_options)
    try:
        if settings.database not in client.list_databases():
            client.create_database(settings.database)
        client.use_database(settings.database)
    except BaseException as error:
        try:
            client.close()
        except BaseException as cleanup_error:
            error.add_note(f"Failed to close Milvus client after connection setup failed: {cleanup_error!r}")
        raise
    return client
