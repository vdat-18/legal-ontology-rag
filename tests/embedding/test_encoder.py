"""Tests for the versioned embedding encoder adapter."""

from __future__ import annotations

import numpy as np

from legal_rag.embedding.encoder import EmbeddingEncoder
from legal_rag.embedding.profiles import EmbeddingProfile


class _FakeEncoderBackend:
    """Record encode calls and return one deterministic vector per input."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def encode(
        self,
        sentences: list[str],
        *,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> np.ndarray:
        """Return one fixed-size vector per input sentence."""
        self.calls.append(
            {
                "sentences": sentences,
                "convert_to_numpy": convert_to_numpy,
                "normalize_embeddings": normalize_embeddings,
            }
        )
        return np.array([[float(len(text)), 0.0] for text in sentences])


def _profile(query_instruction: str | None) -> EmbeddingProfile:
    """Build one embedding profile for encoder tests."""
    return EmbeddingProfile(
        profile_id="test-profile",
        model_id="test/model",
        model_revision="abc123",
        tokenizer_id="test/model",
        tokenizer_revision="abc123",
        dimensions=2,
        max_input_tokens=512,
        query_instruction=query_instruction,
    )


def test_encode_requests_normalized_numpy_vectors() -> None:
    """Passage-side encoding always requests normalized numpy output."""
    backend = _FakeEncoderBackend()
    encoder = EmbeddingEncoder(_profile(None), backend)

    vectors = encoder.encode(["ab", "abcd"])

    assert vectors.tolist() == [[2.0, 0.0], [4.0, 0.0]]
    assert backend.calls == [{"sentences": ["ab", "abcd"], "convert_to_numpy": True, "normalize_embeddings": True}]


def test_encode_query_without_instruction_matches_encode() -> None:
    """Query-side encoding is identical to passage-side when no instruction is configured."""
    backend = _FakeEncoderBackend()
    encoder = EmbeddingEncoder(_profile(None), backend)

    vectors = encoder.encode_query(["ab"])

    assert vectors.tolist() == [[2.0, 0.0]]
    assert backend.calls[0]["sentences"] == ["ab"]


def test_encode_query_prepends_configured_instruction() -> None:
    """Query-side encoding prefixes each text with the profile's query instruction."""
    backend = _FakeEncoderBackend()
    encoder = EmbeddingEncoder(_profile("Instruct: "), backend)

    encoder.encode_query(["ab"])

    assert backend.calls[0]["sentences"] == ["Instruct: ab"]
