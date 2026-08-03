"""Tests for synthesis-only OpenAI-compatible embedding inference."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from legal_rag.benchmark_synthesis import embedding
from legal_rag.benchmark_synthesis.embedding import SynthesisEmbeddingEncoder
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


def test_synthesis_embedding_encoder_rejects_wrong_dimensions(monkeypatch: pytest.MonkeyPatch) -> None:
    """A provider response cannot silently change the configured embedding dimensions."""
    monkeypatch.setattr(
        embedding,
        "urlopen",
        lambda request, *, timeout: _Response({"data": [{"index": 0, "embedding": [1.0]}]}),
    )
    encoder = SynthesisEmbeddingEncoder(_settings(), _profile(), batch_size=2, max_response_bytes=4096)

    with pytest.raises(ValueError, match="Embedding response shape"):
        encoder.encode(["a"])
