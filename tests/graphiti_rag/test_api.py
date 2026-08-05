"""Tests for the Graphiti knowledge-graph RAG HTTP API."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from fastapi.testclient import TestClient

from legal_rag.baml_client.types import ConversationTurn
from legal_rag.graphiti_rag.api import server
from legal_rag.graphiti_rag.pipeline import GraphitiRAGResult
from legal_rag.graphiti_rag.retriever import FactSource, RetrievedFact

_ENVIRONMENT = {
    "LLM_BASE_URL": "http://provider.test/v1",
    "LLM_API_KEY": "test-key",
    "LLM_MODEL": "provider/test-model",
    "LLM_TIMEOUT_SECONDS": "60",
    "EMBEDDING_PROFILE": "qwen3_0_6b",
    "NEO4J_URI": "bolt://localhost:7687",
    "NEO4J_USER": "neo4j",
    "NEO4J_PASSWORD": "password",
}


class _FakePipeline:
    """Answer without touching Neo4j, an embedder, or a language model."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int | None]] = []
        self.histories: list[list[ConversationTurn]] = []
        self.closed = False

    async def run(
        self,
        query: str,
        max_facts: int | None = None,
        history: Sequence[ConversationTurn] = (),
    ) -> GraphitiRAGResult:
        """Record the call and the context it was given, returning one fixed grounded answer."""
        self.calls.append((query, max_facts))
        self.histories.append(list(history))
        return GraphitiRAGResult(
            query=query,
            rewritten_query="không chấp hành hiệu lệnh của đèn tín hiệu giao thông",
            answer="Phạt tiền từ 18 đến 20 triệu đồng.",
            sources=(
                RetrievedFact(
                    fact="Không chấp hành hiệu lệnh của đèn tín hiệu bị phạt 18-20 triệu đồng.",
                    sources=(FactSource(chunk_id="chunk-abc", source_name="168_2024_ND-CP.docx"),),
                ),
            ),
            unindexed_citations=("100/2019/NĐ-CP",),
        )

    async def close(self) -> None:
        """Record that the server released the pipeline."""
        self.closed = True


@pytest.fixture(autouse=True)
def _environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide the settings the API reads, without depending on the developer's real .env."""
    for name, value in _ENVIRONMENT.items():
        monkeypatch.setenv(name, value)


def test_health_does_not_load_the_pipeline() -> None:
    """Health must answer while Neo4j and the embedder are still unavailable."""

    def _fail_if_built() -> _FakePipeline:
        message = "health must not build the pipeline"
        raise AssertionError(message)

    with TestClient(server.create_app(pipeline_factory=_fail_if_built, group_id="g1")) as client:
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["pipeline_loaded"] is False
    assert body["group_id"] == "g1"
    assert body["neo4j_uri"] == "bolt://localhost:7687"


def test_query_returns_the_answer_with_provenance_and_the_rewritten_query() -> None:
    """The response carries the answer, every supporting chunk, and the query actually searched."""
    pipeline = _FakePipeline()

    with TestClient(server.create_app(pipeline_factory=lambda: pipeline, group_id="g1")) as client:
        response = client.post("/graphiti/query", json={"query": "vượt đèn đỏ phạt bao nhiêu"})

    assert response.status_code == 200
    assert response.json() == {
        "query": "vượt đèn đỏ phạt bao nhiêu",
        "rewritten_query": "không chấp hành hiệu lệnh của đèn tín hiệu giao thông",
        "answer": "Phạt tiền từ 18 đến 20 triệu đồng.",
        "sources": [
            {
                "fact": "Không chấp hành hiệu lệnh của đèn tín hiệu bị phạt 18-20 triệu đồng.",
                "sources": [{"chunk_id": "chunk-abc", "source_name": "168_2024_ND-CP.docx"}],
            }
        ],
        "unindexed_citations": ["100/2019/NĐ-CP"],
    }
    assert pipeline.calls == [("vượt đèn đỏ phạt bao nhiêu", None)]


def test_query_forwards_max_facts_to_the_pipeline() -> None:
    """An explicit max_facts override reaches the pipeline instead of being dropped."""
    pipeline = _FakePipeline()

    with TestClient(server.create_app(pipeline_factory=lambda: pipeline, group_id="g1")) as client:
        client.post("/graphiti/query", json={"query": "q", "max_facts": 3})

    assert pipeline.calls == [("q", 3)]


@pytest.mark.parametrize("payload", [{"query": ""}, {"query": "q", "max_facts": 0}])
def test_query_rejects_invalid_input(payload: dict[str, object]) -> None:
    """An empty query or a non-positive max_facts is refused before the pipeline is built."""

    def _fail_if_built() -> _FakePipeline:
        message = "invalid input must not build the pipeline"
        raise AssertionError(message)

    with TestClient(server.create_app(pipeline_factory=_fail_if_built, group_id="g1")) as client:
        response = client.post("/graphiti/query", json=payload)

    assert response.status_code == 422


def test_shutdown_closes_the_pipeline() -> None:
    """The Neo4j connection is released when the server stops, not leaked."""
    pipeline = _FakePipeline()

    with TestClient(server.create_app(pipeline_factory=lambda: pipeline, group_id="g1")) as client:
        client.post("/graphiti/query", json={"query": "q"})

    assert pipeline.closed is True


def test_pipeline_is_built_once_and_reused() -> None:
    """The embedder and Neo4j connection are created once, not per request."""
    built: list[_FakePipeline] = []

    def _factory() -> _FakePipeline:
        pipeline = _FakePipeline()
        built.append(pipeline)
        return pipeline

    with TestClient(server.create_app(pipeline_factory=_factory, group_id="g1")) as client:
        client.post("/graphiti/query", json={"query": "first"})
        client.post("/graphiti/query", json={"query": "second"})

    assert len(built) == 1
    assert built[0].calls == [("first", None), ("second", None)]


def test_allows_the_frontend_origin_so_the_browser_can_call_the_api() -> None:
    """The chat frontend is a cross-origin caller and is blocked without an explicit allow list."""
    pipeline = _FakePipeline()
    app = server.create_app(pipeline_factory=lambda: pipeline, group_id="g1", cors_origins=("http://localhost:3000",))

    with TestClient(app) as client:
        response = client.post("/graphiti/query", json={"query": "q"}, headers={"Origin": "http://localhost:3000"})

    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_does_not_allow_an_unknown_origin() -> None:
    """A wildcard would let any site call the API from a visitor's browser, so origins are pinned."""
    pipeline = _FakePipeline()
    app = server.create_app(pipeline_factory=lambda: pipeline, group_id="g1", cors_origins=("http://localhost:3000",))

    with TestClient(app) as client:
        response = client.post("/graphiti/query", json={"query": "q"}, headers={"Origin": "http://evil.test"})

    assert "access-control-allow-origin" not in response.headers


class _FakeHistory:
    """In-memory stand-in for the Redis-backed conversation store."""

    def __init__(self) -> None:
        self.turns: dict[str, list[ConversationTurn]] = {}

    def load(self, session_id: str) -> list[ConversationTurn]:
        """Return the recorded turns for a session."""
        return list(self.turns.get(session_id, []))

    def append(self, session_id: str, question: str, answer: str) -> None:
        """Record one completed turn."""
        self.turns.setdefault(session_id, []).append(ConversationTurn(question=question, answer=answer))


def test_earlier_turns_are_passed_to_the_pipeline_for_a_session() -> None:
    """A follow-up must reach the pipeline with its context, or it cannot be resolved."""
    pipeline = _FakePipeline()
    history = _FakeHistory()
    app = server.create_app(pipeline_factory=lambda: pipeline, group_id="g1", history=history)

    with TestClient(app) as client:
        client.post("/graphiti/query", json={"query": "ô tô vượt đèn đỏ phạt bao nhiêu", "session_id": "s1"})
        client.post("/graphiti/query", json={"query": "còn xe máy thì sao", "session_id": "s1"})

    assert pipeline.histories[0] == []
    assert [turn.question for turn in pipeline.histories[1]] == ["ô tô vượt đèn đỏ phạt bao nhiêu"]


def test_sessions_do_not_leak_into_one_another() -> None:
    """One visitor's conversation must never appear in another's context."""
    pipeline = _FakePipeline()
    history = _FakeHistory()
    app = server.create_app(pipeline_factory=lambda: pipeline, group_id="g1", history=history)

    with TestClient(app) as client:
        client.post("/graphiti/query", json={"query": "câu hỏi của phiên A", "session_id": "a"})
        client.post("/graphiti/query", json={"query": "câu hỏi của phiên B", "session_id": "b"})

    assert pipeline.histories[1] == []


def test_a_request_without_a_session_is_answered_statelessly() -> None:
    """Omitting session_id must not record anything or borrow another session's context."""
    pipeline = _FakePipeline()
    history = _FakeHistory()
    app = server.create_app(pipeline_factory=lambda: pipeline, group_id="g1", history=history)

    with TestClient(app) as client:
        client.post("/graphiti/query", json={"query": "câu hỏi lẻ"})

    assert pipeline.histories == [[]]
    assert history.turns == {}


def test_no_history_store_still_answers() -> None:
    """The API stays usable when no conversation store is configured."""
    pipeline = _FakePipeline()
    app = server.create_app(pipeline_factory=lambda: pipeline, group_id="g1")

    with TestClient(app) as client:
        response = client.post("/graphiti/query", json={"query": "q", "session_id": "s1"})

    assert response.status_code == 200
    assert pipeline.histories == [[]]
