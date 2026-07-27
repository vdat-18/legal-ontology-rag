"""Tests for building Graphiti episodes from chunked JSONL data and corpus metadata."""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from legal_rag.chunker.models import Chunk
from legal_rag.graphiti_rag.indexer import (
    CorpusDocument,
    CorpusMetadata,
    index_chunks,
    index_chunks_concurrent,
    load_corpus_metadata,
)
from legal_rag.graphiti_rag.ontology import EDGE_TYPES, ENTITY_TYPES


class _FakeGraphiti:
    """Record add_episode calls instead of talking to a real Graphiti/LLM/Neo4j stack."""

    def __init__(self, *, always_fail: bool = False) -> None:
        self.calls: list[dict[str, object]] = []
        self._always_fail = always_fail

    async def add_episode(self, **kwargs: object) -> None:
        """Record the episode arguments passed by the indexer, or raise if configured to fail.

        Yields to the event loop once (``asyncio.sleep(0)``) so concurrent workers sharing one
        queue actually interleave in tests, instead of one worker draining the whole queue
        before another gets a turn (real network calls always yield; these fakes otherwise
        would not, since they contain no genuine suspension point of their own).
        """
        await asyncio.sleep(0)
        self.calls.append(kwargs)
        if self._always_fail:
            msg = "simulated DeepSeek structured-output failure"
            raise ValueError(msg)


class _FakeNeo4jDriver:
    """Report a fixed set of (name, group_id) episode keys as already indexed."""

    def __init__(self, existing_episodes: frozenset[tuple[str, str]] = frozenset()) -> None:
        self._existing_episodes = existing_episodes

    async def execute_query(self, _query: str, **params: object) -> tuple[list[object], None, None]:
        """Return one record when the queried (name, group_id) is in the pre-seeded existing set."""
        key = (params.get("name"), params.get("group_id"))
        records = [object()] if key in self._existing_episodes else []
        return records, None, None


def _chunk(chunk_id: str, source_name: str, chunk_index: int = 0, text: str = "Phạt tiền từ 4 đến 6 triệu.") -> Chunk:
    """Build one final chunk for indexer tests."""
    return Chunk(
        chunk_id=chunk_id,
        content_sha256="content-hash",
        chunker_config_sha256="config-hash",
        scope_id="scope-id",
        chunk_index=chunk_index,
        kind="paragraph_group",
        source_name=source_name,
        source_sha256="source-hash",
        record_scope="root",
        parent_attachment_id=None,
        attachment_member_path=None,
        attachment_member_sha256=None,
        unit_ids=("unit-0",),
        source_refs=("word/document.xml#/body/0",),
        sequence_start=0,
        sequence_end=0,
        text=text,
        context_text="",
        embedding_text=text,
        text_token_count=5,
        embedding_token_count=5,
        previous_chunk_id=None,
        next_chunk_id=None,
    )


def test_load_corpus_metadata_reads_group_id_and_documents(tmp_path: Path) -> None:
    """Corpus metadata parses the shared group id and per-document temporal fields."""
    corpus_config = tmp_path / "corpus.json"
    corpus_config.write_text(
        """
        {
          "group_id": "traffic_penalty_groundtruth_v1",
          "documents": [
            {"file": "168_2024_ND-CP.docx", "so_hieu": "168/2024/NĐ-CP", "effective_date": "2025-01-01", "amends": null},
            {"file": "238_2026_ND-CP.docx", "so_hieu": "238/2026/NĐ-CP", "effective_date": null, "amends": "168/2024/NĐ-CP"}
          ]
        }
        """,
        encoding="utf-8",
    )

    metadata = load_corpus_metadata(corpus_config)

    assert metadata.group_id == "traffic_penalty_groundtruth_v1"
    assert metadata.documents["168_2024_ND-CP.docx"].effective_date == date(2025, 1, 1)
    assert metadata.documents["238_2026_ND-CP.docx"].effective_date is None
    assert metadata.documents["238_2026_ND-CP.docx"].amends == "168/2024/NĐ-CP"


def test_index_chunks_skips_when_source_not_in_corpus_config() -> None:
    """A chunk whose source document isn't in the corpus config is skipped, not guessed."""
    metadata = CorpusMetadata(group_id="g1", documents={})
    chunk = _chunk("chunk-1", "unknown.docx")

    summary = asyncio.run(index_chunks([chunk], metadata, _FakeGraphiti(), _FakeNeo4jDriver()))

    assert summary.indexed == 0
    assert summary.skipped_no_metadata == 1


def test_index_chunks_skips_when_effective_date_is_missing() -> None:
    """A chunk whose document has no known effective_date is skipped, not defaulted to now."""
    metadata = CorpusMetadata(
        group_id="g1",
        documents={"doc.docx": CorpusDocument(so_hieu="1/2024", effective_date=None, amends=None)},
    )
    chunk = _chunk("chunk-1", "doc.docx")

    summary = asyncio.run(index_chunks([chunk], metadata, _FakeGraphiti(), _FakeNeo4jDriver()))

    assert summary.indexed == 0
    assert summary.skipped_no_date == 1


def test_index_chunks_falls_back_to_extracted_effective_date_when_config_date_is_null() -> None:
    """A null config effective_date is resolved from the document's own "Hiệu lực thi hành" text."""
    metadata = CorpusMetadata(
        group_id="g1",
        documents={"doc.docx": CorpusDocument(so_hieu="1/2024", effective_date=None, amends=None)},
    )
    chunk = _chunk(
        "chunk-1",
        "doc.docx",
        text="Điều 20. Hiệu lực thi hành\n\nNghị định này có hiệu lực thi hành từ ngày 15 tháng 8 năm 2026.",
    )
    graphiti = _FakeGraphiti()

    summary = asyncio.run(index_chunks([chunk], metadata, graphiti, _FakeNeo4jDriver()))

    assert summary.indexed == 1
    assert summary.skipped_no_date == 0
    assert graphiti.calls[0]["reference_time"] == datetime(2026, 8, 15, tzinfo=UTC)


def test_index_chunks_adds_one_episode_with_correct_fields() -> None:
    """A valid chunk is added as one Graphiti episode with the expected temporal/group fields."""
    metadata = CorpusMetadata(
        group_id="traffic_penalty_groundtruth_v1",
        documents={"doc.docx": CorpusDocument(so_hieu="168/2024/NĐ-CP", effective_date=date(2025, 1, 1), amends=None)},
    )
    chunk = _chunk("chunk-1", "doc.docx", chunk_index=3, text="Phạt tiền từ 4 đến 6 triệu.")
    graphiti = _FakeGraphiti()

    summary = asyncio.run(index_chunks([chunk], metadata, graphiti, _FakeNeo4jDriver()))

    assert summary.indexed == 1
    assert len(graphiti.calls) == 1
    call = graphiti.calls[0]
    assert call["name"] == "doc.docx#3"
    assert call["episode_body"] == "Phạt tiền từ 4 đến 6 triệu."
    assert call["source_description"] == "doc.docx"
    assert call["reference_time"] == datetime(2025, 1, 1, tzinfo=UTC)
    assert call["group_id"] == "traffic_penalty_groundtruth_v1"
    assert "uuid" not in call
    assert call["entity_types"] is ENTITY_TYPES
    assert call["edge_types"] is EDGE_TYPES


def test_index_chunks_skips_when_episode_already_indexed() -> None:
    """A chunk whose deterministic episode name already exists in the group is skipped."""
    metadata = CorpusMetadata(
        group_id="g1",
        documents={"doc.docx": CorpusDocument(so_hieu="1/2024", effective_date=date(2025, 1, 1), amends=None)},
    )
    chunk = _chunk("chunk-1", "doc.docx", chunk_index=0)
    graphiti = _FakeGraphiti()
    existing = frozenset({("doc.docx#0", "g1")})

    summary = asyncio.run(index_chunks([chunk], metadata, graphiti, _FakeNeo4jDriver(existing)))

    assert summary.indexed == 0
    assert summary.skipped_already_indexed == 1
    assert graphiti.calls == []


def test_index_chunks_retries_then_skips_a_chunk_that_keeps_failing() -> None:
    """A chunk whose add_episode call keeps raising is retried, then skipped, not fatal."""
    metadata = CorpusMetadata(
        group_id="g1",
        documents={"doc.docx": CorpusDocument(so_hieu="1/2024", effective_date=date(2025, 1, 1), amends=None)},
    )
    chunk = _chunk("chunk-1", "doc.docx")
    graphiti = _FakeGraphiti(always_fail=True)

    summary = asyncio.run(index_chunks([chunk], metadata, graphiti, _FakeNeo4jDriver()))

    assert summary.indexed == 0
    assert summary.skipped_error == 1
    assert len(graphiti.calls) == 3


def test_index_chunks_concurrent_distributes_work_across_all_clients() -> None:
    """All chunks are indexed, and more than one client picks up work from the shared queue."""
    metadata = CorpusMetadata(
        group_id="g1",
        documents={"doc.docx": CorpusDocument(so_hieu="1/2024", effective_date=date(2025, 1, 1), amends=None)},
    )
    chunks = [_chunk(f"chunk-{i}", "doc.docx", chunk_index=i) for i in range(6)]
    clients = [_FakeGraphiti(), _FakeGraphiti()]

    summary = asyncio.run(index_chunks_concurrent(chunks, metadata, clients, _FakeNeo4jDriver()))

    assert summary.indexed == 6
    assert summary.skipped_error == 0
    total_calls = len(clients[0].calls) + len(clients[1].calls)
    assert total_calls == 6
    assert len(clients[0].calls) > 0
    assert len(clients[1].calls) > 0


def test_index_chunks_concurrent_skips_no_metadata_and_no_date_chunks() -> None:
    """Ineligible chunks are still filtered out before being handed to any worker."""
    metadata = CorpusMetadata(
        group_id="g1",
        documents={"doc.docx": CorpusDocument(so_hieu="1/2024", effective_date=date(2025, 1, 1), amends=None)},
    )
    chunks = [
        _chunk("chunk-1", "doc.docx", chunk_index=0),
        _chunk("chunk-2", "unknown.docx", chunk_index=0),
    ]
    clients = [_FakeGraphiti(), _FakeGraphiti()]

    summary = asyncio.run(index_chunks_concurrent(chunks, metadata, clients, _FakeNeo4jDriver()))

    assert summary.indexed == 1
    assert summary.skipped_no_metadata == 1


def test_index_chunks_concurrent_retries_then_counts_persistent_failures_as_errors() -> None:
    """A client that always fails still lets the run finish, counting its chunk as an error."""
    metadata = CorpusMetadata(
        group_id="g1",
        documents={"doc.docx": CorpusDocument(so_hieu="1/2024", effective_date=date(2025, 1, 1), amends=None)},
    )
    chunks = [_chunk("chunk-1", "doc.docx", chunk_index=0)]
    clients = [_FakeGraphiti(always_fail=True)]

    summary = asyncio.run(index_chunks_concurrent(chunks, metadata, clients, _FakeNeo4jDriver()))

    assert summary.indexed == 0
    assert summary.skipped_error == 1


if __name__ == "__main__":
    pytest.main([__file__])
