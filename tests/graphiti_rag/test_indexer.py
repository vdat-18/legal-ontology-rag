"""Tests for building Graphiti episodes from chunked JSONL data and corpus metadata."""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime
from pathlib import Path

import neo4j.exceptions
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

_WINDOW_TOKEN_BUDGET = 40_000


class _FakeGraphiti:
    """Record add_episode calls instead of talking to a real Graphiti/LLM/Neo4j stack."""

    def __init__(self, *, always_fail: bool = False, fail_with: Exception | None = None) -> None:
        self.calls: list[dict[str, object]] = []
        self._always_fail = always_fail
        self._fail_with = fail_with

    async def add_episode(self, **kwargs: object) -> None:
        """Record the episode arguments passed by the indexer, or raise if configured to fail.

        Yields to the event loop once (``asyncio.sleep(0)``) so concurrent workers sharing one
        queue actually interleave in tests, instead of one worker draining the whole queue
        before another gets a turn (real network calls always yield; these fakes otherwise
        would not, since they contain no genuine suspension point of their own).
        """
        await asyncio.sleep(0)
        self.calls.append(kwargs)
        if self._fail_with is not None:
            raise self._fail_with
        if self._always_fail:
            msg = "simulated add_episode failure"
            raise ValueError(msg)


class _BlockingGraphiti:
    """Hold one worker in its first episode until the concurrency test releases it."""

    def __init__(self, started: asyncio.Event, release: asyncio.Event) -> None:
        self.calls: list[dict[str, object]] = []
        self._started = started
        self._release = release

    async def add_episode(self, **kwargs: object) -> None:
        """Record one episode and block so a sibling worker can fail."""
        self.calls.append(kwargs)
        self._started.set()
        await self._release.wait()


class _FailAfterSiblingStartsGraphiti:
    """Fail only after the blocking sibling has begun its first episode."""

    def __init__(self, sibling_started: asyncio.Event) -> None:
        self.calls: list[dict[str, object]] = []
        self._sibling_started = sibling_started

    async def add_episode(self, **kwargs: object) -> None:
        """Wait for the sibling worker, then raise a deterministic indexing failure."""
        self.calls.append(kwargs)
        await self._sibling_started.wait()
        msg = "simulated concurrent failure"
        raise ValueError(msg)


class _FakeEffectiveDateClient:
    """Return a fixed extracted date without making any real LLM call."""

    def __init__(self, result: date | None = None) -> None:
        self._result = result

    def extract(self, document_text: str) -> date | None:
        """Return the configured fixed result, ignoring the document text."""
        return self._result


class _FakeNeo4jDriver:
    """Return persisted episode fields for a fixed set of (name, group_id) keys."""

    def __init__(
        self,
        existing_episodes: dict[tuple[str, str], tuple[str, str]] | None = None,
    ) -> None:
        self._existing_episodes = existing_episodes or {}
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def execute_query(self, query: str, **params: object) -> tuple[list[object], None, None]:
        """Compare the expected fields with the pre-seeded persisted episode."""
        self.calls.append((query, params))
        key = (params.get("name"), params.get("group_id"))
        persisted = self._existing_episodes.get(key)
        records: list[object] = []
        if persisted is not None:
            persisted_content, persisted_source_description = persisted
            records.append(
                {
                    "content_matches": persisted_content == params.get("content"),
                    "source_description_matches": persisted_source_description == params.get("source_description"),
                }
            )
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
    assert metadata.documents["168_2024_ND-CP.docx"].document_number == "168/2024/NĐ-CP"
    assert metadata.documents["168_2024_ND-CP.docx"].effective_date == date(2025, 1, 1)
    assert metadata.documents["238_2026_ND-CP.docx"].effective_date is None
    assert metadata.documents["238_2026_ND-CP.docx"].amends == "168/2024/NĐ-CP"


def test_load_corpus_metadata_accepts_selection_only_documents(tmp_path: Path) -> None:
    """The broad corpus config can omit temporal fields and a formal document number."""
    corpus_config = tmp_path / "corpus.json"
    corpus_config.write_text(
        """
        {
          "group_id": "traffic_penalty_groundtruth_v1",
          "documents": [
            {
              "file": "168_2024_ND-CP_619502.docx",
              "selection_reason": "Core road traffic penalty decree."
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    metadata = load_corpus_metadata(corpus_config)

    document = metadata.documents["168_2024_ND-CP_619502.docx"]
    assert document.document_number == "168/2024/ND-CP"
    assert document.effective_date is None
    assert document.amends is None


def test_load_corpus_metadata_prefers_adjacent_validated_document_metadata(tmp_path: Path) -> None:
    """Prepared metadata supplies the authoritative accented number for selection-only configs."""
    corpus_config = tmp_path / "corpus.json"
    corpus_config.write_text(
        """
        {
          "group_id": "traffic_penalty_groundtruth_v1",
          "documents": [{"file": "168_2024_ND-CP_619502.docx"}]
        }
        """,
        encoding="utf-8",
    )
    (tmp_path / "168_2024_ND-CP_619502.document_metadata.jsonl").write_text(
        '{"document_number":"168/2024/NĐ-CP","schema_version":"1.0.0","valid_from":"2025-01-01","valid_to":null}\n',
        encoding="utf-8",
    )

    metadata = load_corpus_metadata(corpus_config)

    document = metadata.documents["168_2024_ND-CP_619502.docx"]
    assert document.document_number == "168/2024/NĐ-CP"
    assert document.effective_date == date(2025, 1, 1)


def test_load_corpus_metadata_rejects_conflicting_metadata_artifacts(tmp_path: Path) -> None:
    """Two authoritative artifacts cannot silently identify one source as different documents."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    corpus_config = config_dir / "corpus.json"
    corpus_config.write_text(
        '{"group_id":"g1","documents":[{"file":"168_2024_ND-CP_619502.docx"}]}',
        encoding="utf-8",
    )
    local_metadata = config_dir / "168_2024_ND-CP_619502.document_metadata.jsonl"
    local_metadata.write_text(
        '{"document_number":"168/2024/NĐ-CP","schema_version":"1.0.0","valid_from":"2025-01-01","valid_to":null}\n',
        encoding="utf-8",
    )
    processed_dir = tmp_path / "processed" / "groundtruth"
    processed_dir.mkdir(parents=True)
    (processed_dir / local_metadata.name).write_text(
        '{"document_number":"999/2024/NĐ-CP","schema_version":"1.0.0","valid_from":"2025-01-01","valid_to":null}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Conflicting document metadata"):
        load_corpus_metadata(corpus_config)


def test_index_chunks_skips_when_source_not_in_corpus_config() -> None:
    """A chunk whose source document isn't in the corpus config is skipped, not guessed."""
    metadata = CorpusMetadata(group_id="g1", documents={})
    chunk = _chunk("chunk-1", "unknown.docx")

    summary = asyncio.run(
        index_chunks(
            [chunk], metadata, _FakeGraphiti(), _FakeNeo4jDriver(), _FakeEffectiveDateClient(), _WINDOW_TOKEN_BUDGET
        )
    )

    assert summary.indexed == 0
    assert summary.skipped_no_metadata == 1


def test_index_chunks_skips_when_effective_date_is_missing() -> None:
    """A chunk whose document has no known effective_date is skipped, not defaulted to now."""
    metadata = CorpusMetadata(
        group_id="g1",
        documents={"doc.docx": CorpusDocument(document_number="1/2024", effective_date=None, amends=None)},
    )
    chunk = _chunk("chunk-1", "doc.docx")

    summary = asyncio.run(
        index_chunks(
            [chunk], metadata, _FakeGraphiti(), _FakeNeo4jDriver(), _FakeEffectiveDateClient(), _WINDOW_TOKEN_BUDGET
        )
    )

    assert summary.indexed == 0
    assert summary.skipped_no_date == 1


def test_index_chunks_falls_back_to_extracted_effective_date_when_config_date_is_null() -> None:
    """A null config effective_date is resolved from the date extraction client's own result."""
    metadata = CorpusMetadata(
        group_id="g1",
        documents={"doc.docx": CorpusDocument(document_number="1/2024", effective_date=None, amends=None)},
    )
    chunk = _chunk("chunk-1", "doc.docx")
    graphiti = _FakeGraphiti()
    date_extraction_client = _FakeEffectiveDateClient(date(2026, 8, 15))

    summary = asyncio.run(
        index_chunks([chunk], metadata, graphiti, _FakeNeo4jDriver(), date_extraction_client, _WINDOW_TOKEN_BUDGET)
    )

    assert summary.indexed == 1
    assert summary.skipped_no_date == 0
    assert graphiti.calls[0]["reference_time"] == datetime(2026, 8, 15, tzinfo=UTC)


def test_index_chunks_adds_one_episode_named_by_its_chunk_id() -> None:
    """A valid chunk is added as one Graphiti episode named by its stable chunk_id."""
    metadata = CorpusMetadata(
        group_id="traffic_penalty_groundtruth_v1",
        documents={
            "doc.docx": CorpusDocument(document_number="168/2024/NĐ-CP", effective_date=date(2025, 1, 1), amends=None)
        },
    )
    chunk = _chunk("chunk-abc123", "doc.docx", chunk_index=3, text="Phạt tiền từ 4 đến 6 triệu.")
    graphiti = _FakeGraphiti()

    summary = asyncio.run(
        index_chunks([chunk], metadata, graphiti, _FakeNeo4jDriver(), _FakeEffectiveDateClient(), _WINDOW_TOKEN_BUDGET)
    )

    assert summary.indexed == 1
    assert len(graphiti.calls) == 1
    call = graphiti.calls[0]
    assert call["name"] == "chunk-abc123"
    assert call["episode_body"] == "Phạt tiền từ 4 đến 6 triệu."
    assert call["source_description"] == "doc.docx"
    assert call["reference_time"] == datetime(2025, 1, 1, tzinfo=UTC)
    assert call["group_id"] == "traffic_penalty_groundtruth_v1"
    assert "uuid" not in call
    assert call["entity_types"] is ENTITY_TYPES
    assert call["edge_types"] is EDGE_TYPES


def test_index_chunks_normalizes_undated_fact_edges_from_reference_time() -> None:
    """Facts without an extracted date inherit their persisted episode reference date."""
    metadata = CorpusMetadata(
        group_id="g1",
        documents={"doc.docx": CorpusDocument(document_number="1/2024", effective_date=date(2025, 1, 1), amends=None)},
    )
    chunk = _chunk("chunk-1", "doc.docx")
    driver = _FakeNeo4jDriver()

    asyncio.run(
        index_chunks([chunk], metadata, _FakeGraphiti(), driver, _FakeEffectiveDateClient(), _WINDOW_TOKEN_BUDGET)
    )

    query, params = driver.calls[-1]
    assert "edge.valid_at IS NULL" in query
    assert "coalesce(edge.reference_time, episode.valid_at)" in query
    assert params == {"name": "chunk-1", "group_id": "g1"}


def test_index_chunks_skips_when_episode_already_indexed() -> None:
    """A matching persisted episode is skipped without another Graphiti write."""
    metadata = CorpusMetadata(
        group_id="g1",
        documents={"doc.docx": CorpusDocument(document_number="1/2024", effective_date=date(2025, 1, 1), amends=None)},
    )
    chunk = _chunk("chunk-1", "doc.docx", chunk_index=0)
    graphiti = _FakeGraphiti()
    neo4j_driver = _FakeNeo4jDriver({("chunk-1", "g1"): (chunk.text, chunk.source_name)})

    summary = asyncio.run(
        index_chunks([chunk], metadata, graphiti, neo4j_driver, _FakeEffectiveDateClient(), _WINDOW_TOKEN_BUDGET)
    )

    assert summary.indexed == 0
    assert summary.skipped_already_indexed == 1
    assert graphiti.calls == []
    query, params = neo4j_driver.calls[0]
    assert "$name" in query
    assert "$group_id" in query
    assert "$content" in query
    assert "$source_description" in query
    assert chunk.text not in query
    assert params == {
        "name": "chunk-1",
        "group_id": "g1",
        "content": chunk.text,
        "source_description": "doc.docx",
    }


def test_index_chunks_rejects_existing_episode_with_stale_content() -> None:
    """A reused chunk id cannot silently retain content from an older corpus build."""
    metadata = CorpusMetadata(
        group_id="g1",
        documents={"doc.docx": CorpusDocument(document_number="1/2024", effective_date=date(2025, 1, 1), amends=None)},
    )
    chunk = _chunk("chunk-1", "doc.docx", text="Nội dung hiện tại.")
    graphiti = _FakeGraphiti()
    neo4j_driver = _FakeNeo4jDriver({("chunk-1", "g1"): ("Nội dung cũ.", "doc.docx")})

    with pytest.raises(ValueError, match=r"chunk-1.*content.*Rebuild the Graphiti index"):
        asyncio.run(
            index_chunks([chunk], metadata, graphiti, neo4j_driver, _FakeEffectiveDateClient(), _WINDOW_TOKEN_BUDGET)
        )

    assert graphiti.calls == []


def test_index_chunks_propagates_a_failing_add_episode() -> None:
    """A failing add_episode raises out of the run rather than being swallowed as a skip."""
    metadata = CorpusMetadata(
        group_id="g1",
        documents={"doc.docx": CorpusDocument(document_number="1/2024", effective_date=date(2025, 1, 1), amends=None)},
    )
    chunk = _chunk("chunk-1", "doc.docx")
    graphiti = _FakeGraphiti(always_fail=True)

    with pytest.raises(ValueError, match="simulated add_episode failure"):
        asyncio.run(
            index_chunks(
                [chunk], metadata, graphiti, _FakeNeo4jDriver(), _FakeEffectiveDateClient(), _WINDOW_TOKEN_BUDGET
            )
        )

    assert len(graphiti.calls) == 1


def test_index_chunks_propagates_a_configuration_error_without_special_casing() -> None:
    """A configuration error (bad Neo4j credentials, ...) surfaces on the first attempt, not retried."""
    metadata = CorpusMetadata(
        group_id="g1",
        documents={"doc.docx": CorpusDocument(document_number="1/2024", effective_date=date(2025, 1, 1), amends=None)},
    )
    chunk = _chunk("chunk-1", "doc.docx")
    graphiti = _FakeGraphiti(fail_with=neo4j.exceptions.AuthError("bad credentials"))

    with pytest.raises(neo4j.exceptions.AuthError):
        asyncio.run(
            index_chunks(
                [chunk], metadata, graphiti, _FakeNeo4jDriver(), _FakeEffectiveDateClient(), _WINDOW_TOKEN_BUDGET
            )
        )

    assert len(graphiti.calls) == 1


def test_index_chunks_concurrent_distributes_work_across_all_clients() -> None:
    """All chunks are indexed, and more than one client picks up work from the shared queue."""
    metadata = CorpusMetadata(
        group_id="g1",
        documents={"doc.docx": CorpusDocument(document_number="1/2024", effective_date=date(2025, 1, 1), amends=None)},
    )
    chunks = [_chunk(f"chunk-{i}", "doc.docx", chunk_index=i) for i in range(6)]
    clients = [_FakeGraphiti(), _FakeGraphiti()]

    summary = asyncio.run(
        index_chunks_concurrent(
            chunks, metadata, clients, _FakeNeo4jDriver(), _FakeEffectiveDateClient(), _WINDOW_TOKEN_BUDGET
        )
    )

    assert summary.indexed == 6
    total_calls = len(clients[0].calls) + len(clients[1].calls)
    assert total_calls == 6
    assert len(clients[0].calls) > 0
    assert len(clients[1].calls) > 0


def test_index_chunks_concurrent_skips_no_metadata_and_no_date_chunks() -> None:
    """Ineligible chunks are still filtered out before being handed to any worker."""
    metadata = CorpusMetadata(
        group_id="g1",
        documents={"doc.docx": CorpusDocument(document_number="1/2024", effective_date=date(2025, 1, 1), amends=None)},
    )
    chunks = [
        _chunk("chunk-1", "doc.docx", chunk_index=0),
        _chunk("chunk-2", "unknown.docx", chunk_index=0),
    ]
    clients = [_FakeGraphiti(), _FakeGraphiti()]

    summary = asyncio.run(
        index_chunks_concurrent(
            chunks, metadata, clients, _FakeNeo4jDriver(), _FakeEffectiveDateClient(), _WINDOW_TOKEN_BUDGET
        )
    )

    assert summary.indexed == 1
    assert summary.skipped_no_metadata == 1


def test_index_chunks_concurrent_rejects_existing_episode_with_stale_source_description() -> None:
    """Concurrent workers apply the same provenance freshness check as the serial indexer."""
    metadata = CorpusMetadata(
        group_id="g1",
        documents={"doc.docx": CorpusDocument(document_number="1/2024", effective_date=date(2025, 1, 1), amends=None)},
    )
    chunk = _chunk("chunk-1", "doc.docx")
    client = _FakeGraphiti()
    neo4j_driver = _FakeNeo4jDriver({("chunk-1", "g1"): (chunk.text, "old-name.docx")})

    with pytest.raises(ValueError, match=r"chunk-1.*source_description.*Rebuild the Graphiti index"):
        asyncio.run(
            index_chunks_concurrent(
                [chunk], metadata, [client], neo4j_driver, _FakeEffectiveDateClient(), _WINDOW_TOKEN_BUDGET
            )
        )

    assert client.calls == []


def test_index_chunks_concurrent_propagates_a_failing_add_episode() -> None:
    """A worker whose add_episode fails propagates the error instead of counting it as a skip."""
    metadata = CorpusMetadata(
        group_id="g1",
        documents={"doc.docx": CorpusDocument(document_number="1/2024", effective_date=date(2025, 1, 1), amends=None)},
    )
    chunks = [_chunk("chunk-1", "doc.docx", chunk_index=0)]
    clients = [_FakeGraphiti(always_fail=True)]

    with pytest.raises(ValueError, match="simulated add_episode failure"):
        asyncio.run(
            index_chunks_concurrent(
                chunks, metadata, clients, _FakeNeo4jDriver(), _FakeEffectiveDateClient(), _WINDOW_TOKEN_BUDGET
            )
        )


def test_index_chunks_concurrent_cancels_sibling_workers_before_raising() -> None:
    """A failed worker cannot keep indexing after the concurrent call has returned an error."""

    async def run_scenario() -> None:
        metadata = CorpusMetadata(
            group_id="g1",
            documents={
                "doc.docx": CorpusDocument(document_number="1/2024", effective_date=date(2025, 1, 1), amends=None)
            },
        )
        chunks = [_chunk(f"chunk-{index}", "doc.docx", chunk_index=index) for index in range(4)]
        sibling_started = asyncio.Event()
        release = asyncio.Event()
        blocking_client = _BlockingGraphiti(sibling_started, release)
        failing_client = _FailAfterSiblingStartsGraphiti(sibling_started)

        with pytest.raises(ValueError, match="simulated concurrent failure"):
            await index_chunks_concurrent(
                chunks,
                metadata,
                [blocking_client, failing_client],
                _FakeNeo4jDriver(),
                _FakeEffectiveDateClient(),
                _WINDOW_TOKEN_BUDGET,
            )

        release.set()
        await asyncio.sleep(0)
        assert len(blocking_client.calls) == 1
        assert len(failing_client.calls) == 1

    asyncio.run(run_scenario())


def test_index_chunks_concurrent_cancels_workers_when_caller_is_cancelled() -> None:
    """Cancelling the parent waits for worker cancellation before returning control."""

    async def run_scenario() -> None:
        metadata = CorpusMetadata(
            group_id="g1",
            documents={
                "doc.docx": CorpusDocument(document_number="1/2024", effective_date=date(2025, 1, 1), amends=None)
            },
        )
        chunks = [_chunk(f"chunk-{index}", "doc.docx", chunk_index=index) for index in range(4)]
        started = asyncio.Event()
        release = asyncio.Event()
        clients = [_BlockingGraphiti(started, release), _BlockingGraphiti(started, release)]
        operation = asyncio.create_task(
            index_chunks_concurrent(
                chunks,
                metadata,
                clients,
                _FakeNeo4jDriver(),
                _FakeEffectiveDateClient(),
                _WINDOW_TOKEN_BUDGET,
            )
        )

        await started.wait()
        calls_before_cancel = sum(len(client.calls) for client in clients)
        operation.cancel()

        with pytest.raises(asyncio.CancelledError):
            await operation

        live_tasks = [task for task in asyncio.all_tasks() if task is not asyncio.current_task() and not task.done()]
        assert live_tasks == []

        release.set()
        await asyncio.sleep(0)
        assert sum(len(client.calls) for client in clients) == calls_before_cancel

    asyncio.run(run_scenario())


if __name__ == "__main__":
    pytest.main([__file__])
