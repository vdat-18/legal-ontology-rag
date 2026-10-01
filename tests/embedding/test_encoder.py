"""Tests for the versioned embedding encoder adapter."""

from __future__ import annotations

import numpy as np
import pytest

from legal_rag.embedding.encoder import EmbeddingEncoder, embedding_local_files_only
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


@pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
def test_embedding_local_files_only_accepts_documented_true_values(value: str) -> None:
    """The shared embedding loader recognizes the same true values for every pipeline."""
    assert embedding_local_files_only({"EMBEDDING_LOCAL_FILES_ONLY": value}) is True


def test_embedding_local_files_only_defaults_to_false() -> None:
    """Missing embedding cache configuration permits the pinned model download."""
    assert embedding_local_files_only({}) is False


@pytest.mark.parametrize("value", ["", "0", "false", "NO", "off"])
def test_embedding_local_files_only_rejects_false_values(value: str) -> None:
    """False-like values do not accidentally force an offline model load."""
    assert embedding_local_files_only({"EMBEDDING_LOCAL_FILES_ONLY": value}) is False


def test_from_profile_passes_local_files_only_to_sentence_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    """The shared loader applies the offline flag used by Naive, Graphiti, and Ontology RAG."""
    captured: dict[str, object] = {}
    backend = _FakeEncoderBackend()

    def fake_sentence_transformer(model_id: str, **kwargs: object) -> _FakeEncoderBackend:
        captured["model_id"] = model_id
        captured.update(kwargs)
        return backend

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", fake_sentence_transformer)

    EmbeddingEncoder.from_profile(
        _profile(None),
        environment={"EMBEDDING_LOCAL_FILES_ONLY": "true"},
    )

    assert captured == {
        "model_id": "test/model",
        "revision": "abc123",
        "local_files_only": True,
    }
