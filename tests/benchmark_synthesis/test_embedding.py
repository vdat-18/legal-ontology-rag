"""Tests for synthesis-only OpenAI-compatible embedding inference."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from loguru import logger

from legal_rag.benchmark_synthesis import embedding
from legal_rag.benchmark_synthesis.embedding import IncompleteEmbeddingBatchesError, SynthesisEmbeddingEncoder
from legal_rag.embedding.profiles import EmbeddingProfile
from legal_rag.settings import SynthesisEmbeddingSettings


class _Response:
    """Bounded in-memory HTTP response for embedding tests."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._body = json.dumps(payload).encode()

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, amount: int) -> bytes:
        """Return at most the requested number of response bytes."""
        return self._body[:amount]


def _profile() -> EmbeddingProfile:
    """Build a two-dimensional profile for transport tests."""
    return EmbeddingProfile(
        profile_id="test-bge-m3",
        model_id="BAAI/bge-m3",
        model_revision="revision",
        tokenizer_id="BAAI/bge-m3",
        tokenizer_revision="revision",
        dimensions=2,
        max_input_tokens=8192,
        query_instruction=None,
    )


def _settings() -> SynthesisEmbeddingSettings:
    """Build dedicated embedding provider settings."""
    return SynthesisEmbeddingSettings(
        base_url="https://embedding.test/v1",
        api_key="secret-key",
        model="provider/bge-m3",
        timeout_seconds=120,
    )


def test_synthesis_embedding_encoder_batches_orders_and_normalizes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Requests are bounded by batch size and provider indices determine output order."""
    requests: list[object] = []
    payloads = [
        {"data": [{"index": 1, "embedding": [0.0, 4.0]}, {"index": 0, "embedding": [3.0, 0.0]}]},
        {"data": [{"index": 0, "embedding": [3.0, 4.0]}]},
    ]

    def _urlopen(request: object, *, timeout: float) -> _Response:
        requests.append(request)
        assert timeout == 120
        return _Response(payloads[len(requests) - 1])

    monkeypatch.setattr(embedding, "urlopen", _urlopen)
    encoder = SynthesisEmbeddingEncoder(_settings(), _profile(), batch_size=2, max_response_bytes=4096)

    vectors = encoder.encode(["a", "b", "c"])

    assert len(requests) == 2
    assert np.allclose(vectors, np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]]))


def test_synthesis_embedding_encoder_logs_provider_token_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Embedding responses expose standard provider token counts in logs."""
    monkeypatch.setattr(
        embedding,
        "urlopen",
        lambda request, *, timeout: _Response(
            {
                "data": [{"index": 0, "embedding": [1.0, 0.0]}],
                "usage": {"prompt_tokens": 12, "total_tokens": 12},
            }
        ),
    )
    messages: list[str] = []
    sink_id = logger.add(messages.append, format="{message}")

    try:
        SynthesisEmbeddingEncoder(
            _settings(),
            _profile(),
            batch_size=1,
            max_response_bytes=4096,
        ).encode(["a"])
    finally:
        logger.remove(sink_id)

    assert any(
        "Embedding token usage: model=provider/bge-m3 input_tokens=12 total_tokens=12" in message
        for message in messages
    )


def test_synthesis_embedding_encoder_rejects_wrong_dimensions(monkeypatch: pytest.MonkeyPatch) -> None:
    """A provider response cannot silently change the configured embedding dimensions."""
    monkeypatch.setattr(
        embedding,
        "urlopen",
        lambda request, *, timeout: _Response({"data": [{"index": 0, "embedding": [1.0]}]}),
    )
    encoder = SynthesisEmbeddingEncoder(_settings(), _profile(), batch_size=2, max_response_bytes=4096)

    with pytest.raises(IncompleteEmbeddingBatchesError, match="Embedding response shape"):
        encoder.encode(["a"])


def test_synthesis_embedding_encoder_runs_batches_concurrently(monkeypatch: pytest.MonkeyPatch) -> None:
    """Independent requests overlap while final vectors retain input order."""
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    active = 0
    max_active = 0

    def _urlopen(request: Any, *, timeout: float) -> _Response:
        nonlocal active, max_active
        text = json.loads(request.data)["input"][0]
        with lock:
            active += 1
            max_active = max(max_active, active)
        barrier.wait(timeout=2)
        time.sleep(0.01)
        with lock:
            active -= 1
        vector = [1.0, 0.0] if text == "a" else [0.0, 1.0]
        return _Response({"data": [{"index": 0, "embedding": vector}]})

    monkeypatch.setattr(embedding, "urlopen", _urlopen)
    encoder = SynthesisEmbeddingEncoder(
        _settings(),
        _profile(),
        batch_size=1,
        max_concurrency=2,
        max_response_bytes=4096,
    )

    vectors = encoder.encode(["a", "b"])

    assert max_active == 2
    assert np.allclose(vectors, np.array([[1.0, 0.0], [0.0, 1.0]]))


def test_synthesis_embedding_encoder_retries_one_failed_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    """A retryable provider failure does not restart successful attempts from other batches."""
    attempts = 0

    def _urlopen(request: object, *, timeout: float) -> _Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError
        return _Response({"data": [{"index": 0, "embedding": [1.0, 0.0]}]})

    monkeypatch.setattr(embedding, "urlopen", _urlopen)
    monkeypatch.setattr(embedding.time, "sleep", lambda _seconds: None)
    encoder = SynthesisEmbeddingEncoder(
        _settings(),
        _profile(),
        batch_size=1,
        max_retries=1,
        retry_delay_seconds=120,
        max_response_bytes=4096,
    )

    vectors = encoder.encode(["a"])

    assert attempts == 2
    assert np.allclose(vectors, np.array([[1.0, 0.0]]))


def test_synthesis_embedding_encoder_resumes_completed_batch_checkpoints(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A resumed run calls the provider only for a missing batch."""
    requested_inputs: list[list[str]] = []

    def _urlopen(request: Any, *, timeout: float) -> _Response:
        texts = json.loads(request.data)["input"]
        requested_inputs.append(texts)
        return _Response({"data": [{"index": index, "embedding": [1.0, 0.0]} for index, _ in enumerate(texts)]})

    monkeypatch.setattr(embedding, "urlopen", _urlopen)
    checkpoint_directory = tmp_path / "embeddings"
    encoder = SynthesisEmbeddingEncoder(
        _settings(),
        _profile(),
        batch_size=2,
        max_concurrency=2,
        max_response_bytes=4096,
        checkpoint_directory=checkpoint_directory,
        run_id="candidate-run",
    )
    encoder.encode(["a", "b", "c"])
    (checkpoint_directory / "batch-000001.npy").unlink()
    requested_inputs.clear()

    resumed_vectors = encoder.encode(["a", "b", "c"])

    assert requested_inputs == [["c"]]
    assert np.allclose(resumed_vectors, np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]))


def test_synthesis_embedding_encoder_checkpoints_successes_before_reporting_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A failed run preserves successful concurrent batches for the next resume."""
    fail_second_batch = True
    requested_inputs: list[str] = []

    def _urlopen(request: Any, *, timeout: float) -> _Response:
        text = json.loads(request.data)["input"][0]
        requested_inputs.append(text)
        if text == "b" and fail_second_batch:
            raise OSError
        return _Response({"data": [{"index": 0, "embedding": [1.0, 0.0]}]})

    monkeypatch.setattr(embedding, "urlopen", _urlopen)
    checkpoint_directory = tmp_path / "embeddings"
    encoder = SynthesisEmbeddingEncoder(
        _settings(),
        _profile(),
        batch_size=1,
        max_concurrency=2,
        max_response_bytes=4096,
        checkpoint_directory=checkpoint_directory,
        run_id="candidate-run",
    )

    with pytest.raises(IncompleteEmbeddingBatchesError, match="embedding-batch:1"):
        encoder.encode(["a", "b"])
    assert (checkpoint_directory / "batch-000000.npy").exists()
    assert not (checkpoint_directory / "batch-000001.npy").exists()

    fail_second_batch = False
    requested_inputs.clear()
    vectors = encoder.encode(["a", "b"])

    assert requested_inputs == ["b"]
    assert np.allclose(vectors, np.array([[1.0, 0.0], [1.0, 0.0]]))
