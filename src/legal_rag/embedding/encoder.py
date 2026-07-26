"""Sentence-embedding inference pinned to a versioned embedding profile."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from legal_rag.embedding.profiles import EmbeddingProfile


class _EncoderBackend(Protocol):
    """Embedding model behavior required by the encoder adapter."""

    def encode(
        self,
        sentences: list[str],
        *,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> np.ndarray:
        """Encode sentences into embedding vectors."""


class EmbeddingEncoder:
    """Encode text into L2-normalized vectors using a pinned embedding profile."""

    def __init__(self, profile: EmbeddingProfile, model: _EncoderBackend) -> None:
        """Bind a loaded embedding model to its versioned profile."""
        self.profile = profile
        self._model = model

    @classmethod
    def from_profile(cls, profile: EmbeddingProfile) -> EmbeddingEncoder:
        """Load the sentence-transformers model pinned by an embedding profile."""
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(profile.model_id, revision=profile.model_revision)
        return cls(profile, model)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return L2-normalized embedding vectors for passage-side text."""
        return self._model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True)

    def encode_query(self, texts: Sequence[str]) -> np.ndarray:
        """Return L2-normalized embedding vectors for query-side text."""
        if self.profile.query_instruction is None:
            return self.encode(texts)
        instructed = [self.profile.query_instruction + text for text in texts]
        return self.encode(instructed)
