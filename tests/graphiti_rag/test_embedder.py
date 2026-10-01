"""Tests for the Graphiti embedder adapter backed by the shared embedding encoder."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from legal_rag.embedding.encoder import EmbeddingEncoder
from legal_rag.embedding.profiles import EmbeddingProfile
from legal_rag.graphiti_rag import embedder as embedder_module
from legal_rag.graphiti_rag.embedder import QwenGraphitiEmbedder


class _FakeEncoderBackend:
    """Return one deterministic 2-dim vector per input text."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, sentences: list[str], *, convert_to_numpy: bool, normalize_embeddings: bool) -> np.ndarray:
        """Return a fixed-size vector per sentence."""
        self.calls.append(list(sentences))
        return np.array([[float(len(text)), 0.0] for text in sentences])


def _profile(*, query_instruction: str | None = None) -> EmbeddingProfile:
    """Build one 2-dim embedding profile for embedder tests."""
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


def test_create_returns_one_vector_as_a_plain_list() -> None:
    """A single-text embed request returns a plain list of floats."""
    backend = _FakeEncoderBackend()
    embedder = QwenGraphitiEmbedder(EmbeddingEncoder(_profile(), backend))

    vector = asyncio.run(embedder.create("ab"))

    assert vector == [2.0, 0.0]
    assert backend.calls == [["ab"]]


def test_create_batch_returns_one_vector_per_input() -> None:
    """A batch embed request returns one plain-list vector per input text."""
    backend = _FakeEncoderBackend()
    embedder = QwenGraphitiEmbedder(EmbeddingEncoder(_profile(), backend))

    vectors = asyncio.run(embedder.create_batch(["ab", "abcd"]))

    assert vectors == [[2.0, 0.0], [4.0, 0.0]]
    assert backend.calls == [["ab", "abcd"]]


def test_create_unwraps_graphiti_one_item_text_lists() -> None:
    """Graphiti's one-item list input embeds the text rather than its Python representation."""
    backend = _FakeEncoderBackend()
    embedder = QwenGraphitiEmbedder(EmbeddingEncoder(_profile(), backend))

    vector = asyncio.run(embedder.create(["ab"]))

    assert vector == [2.0, 0.0]
    assert backend.calls == [["ab"]]


@pytest.mark.parametrize("input_data", [[], ["ab", "cd"], [1], [[1, 2]]])
def test_create_rejects_unsupported_non_single_text_inputs(input_data: object) -> None:
    """Ambiguous token and multi-text inputs fail instead of producing corrupted embeddings."""
    embedder = QwenGraphitiEmbedder(EmbeddingEncoder(_profile(), _FakeEncoderBackend()))

    with pytest.raises(TypeError, match="expects one string"):
        asyncio.run(embedder.create(input_data))  # type: ignore[arg-type]


def test_create_query_applies_the_profile_query_instruction() -> None:
    """Graph search uses query-side encoding while stored graph facts remain passage embeddings."""
    backend = _FakeEncoderBackend()
    embedder = QwenGraphitiEmbedder(EmbeddingEncoder(_profile(query_instruction="Instruct: "), backend))

    vector = asyncio.run(embedder.create_query("ab"))

    assert vector == [12.0, 0.0]
    assert backend.calls == [["Instruct: ab"]]


def test_create_query_without_an_instruction_matches_passage_text() -> None:
    """Profiles without query instructions preserve the raw query text."""
    backend = _FakeEncoderBackend()
    embedder = QwenGraphitiEmbedder(EmbeddingEncoder(_profile(), backend))

    vector = asyncio.run(embedder.create_query("ab"))

    assert vector == [2.0, 0.0]
    assert backend.calls == [["ab"]]


def test_index_metadata_validation_is_cached_per_group(monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeated queries avoid adding a metadata round trip to every benchmark sample."""
    calls: list[tuple[object, str, EmbeddingProfile]] = []

    async def fake_validate(driver: object, group_id: str, profile: EmbeddingProfile) -> None:
        calls.append((driver, group_id, profile))

    monkeypatch.setattr(embedder_module, "validate_graphiti_index_metadata", fake_validate)
    profile = _profile()
    embedder = QwenGraphitiEmbedder(EmbeddingEncoder(profile, _FakeEncoderBackend()))
    driver = object()

    async def validate_twice() -> None:
        await embedder.validate_index(driver, "group-1")
        await embedder.validate_index(driver, "group-1")

    asyncio.run(validate_twice())

    assert calls == [(driver, "group-1", profile)]
