"""Tests for the Graphiti embedder adapter backed by the shared embedding encoder."""

from __future__ import annotations

import asyncio

import numpy as np

from legal_rag.embedding.encoder import EmbeddingEncoder
from legal_rag.embedding.profiles import EmbeddingProfile
from legal_rag.graphiti_rag.embedder import QwenGraphitiEmbedder


class _FakeEncoderBackend:
    """Return one deterministic 2-dim vector per input text."""

    def encode(self, sentences: list[str], *, convert_to_numpy: bool, normalize_embeddings: bool) -> np.ndarray:
        """Return a fixed-size vector per sentence."""
        return np.array([[float(len(text)), 0.0] for text in sentences])


def _profile() -> EmbeddingProfile:
    """Build one 2-dim embedding profile for embedder tests."""
    return EmbeddingProfile(
        profile_id="test-profile",
        model_id="test/model",
        model_revision="abc123",
        tokenizer_id="test/model",
        tokenizer_revision="abc123",
        dimensions=2,
        max_input_tokens=512,
        query_instruction=None,
    )


def test_create_returns_one_vector_as_a_plain_list() -> None:
    """A single-text embed request returns a plain list of floats."""
    embedder = QwenGraphitiEmbedder(EmbeddingEncoder(_profile(), _FakeEncoderBackend()))

    vector = asyncio.run(embedder.create("ab"))

    assert vector == [2.0, 0.0]


def test_create_batch_returns_one_vector_per_input() -> None:
    """A batch embed request returns one plain-list vector per input text."""
    embedder = QwenGraphitiEmbedder(EmbeddingEncoder(_profile(), _FakeEncoderBackend()))

    vectors = asyncio.run(embedder.create_batch(["ab", "abcd"]))

    assert vectors == [[2.0, 0.0], [4.0, 0.0]]
