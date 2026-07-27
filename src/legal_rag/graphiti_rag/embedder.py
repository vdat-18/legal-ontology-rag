"""Graphiti embedder adapter backed by the shared versioned embedding encoder."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable

from graphiti_core.embedder.client import EmbedderClient

from legal_rag.embedding.encoder import EmbeddingEncoder
from legal_rag.embedding.profiles import EmbeddingProfile


class QwenGraphitiEmbedder(EmbedderClient):
    """Adapt the project's sentence-transformers encoder to Graphiti's embedder interface."""

    def __init__(self, encoder: EmbeddingEncoder) -> None:
        """Bind a loaded embedding encoder for use by Graphiti's async embedder interface."""
        self._encoder = encoder

    @classmethod
    def from_profile(cls, profile: EmbeddingProfile) -> QwenGraphitiEmbedder:
        """Load the sentence-transformers model pinned by an embedding profile."""
        return cls(EmbeddingEncoder.from_profile(profile))

    async def create(
        self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
    ) -> list[float]:
        """Embed one text and return its vector as a plain list of floats."""
        text = input_data if isinstance(input_data, str) else str(input_data)
        vectors = await asyncio.to_thread(self._encoder.encode, [text])
        return vectors[0].tolist()

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        """Embed a batch of texts and return their vectors as plain lists of floats."""
        vectors = await asyncio.to_thread(self._encoder.encode, input_data_list)
        return vectors.tolist()
