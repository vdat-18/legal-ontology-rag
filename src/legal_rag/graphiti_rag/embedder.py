"""Graphiti embedder adapter backed by the shared versioned embedding encoder."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from typing import TYPE_CHECKING

from graphiti_core.embedder.client import EmbedderClient

from legal_rag.embedding.encoder import EmbeddingEncoder
from legal_rag.embedding.profiles import EmbeddingProfile
from legal_rag.graphiti_rag.index_metadata import validate_graphiti_index_metadata

if TYPE_CHECKING:
    from graphiti_core.driver.driver import GraphDriver


class QwenGraphitiEmbedder(EmbedderClient):
    """Adapt the project's sentence-transformers encoder to Graphiti's embedder interface."""

    def __init__(self, encoder: EmbeddingEncoder) -> None:
        """Bind a loaded embedding encoder for use by Graphiti's async embedder interface."""
        self._encoder = encoder
        self._validated_group_ids: set[str] = set()
        self._index_validation_lock = asyncio.Lock()

    @classmethod
    def from_profile(cls, profile: EmbeddingProfile) -> QwenGraphitiEmbedder:
        """Load the sentence-transformers model pinned by an embedding profile."""
        return cls(EmbeddingEncoder.from_profile(profile))

    async def create(self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]) -> list[float]:
        """Embed one passage and return its vector as a plain list of floats."""
        text = _single_text(input_data)
        vectors = await asyncio.to_thread(self._encoder.encode, [text])
        return vectors[0].tolist()

    async def create_query(self, query: str) -> list[float]:
        """Embed one retrieval query with the profile's query-side instruction."""
        vectors = await asyncio.to_thread(self._encoder.encode_query, [query])
        return vectors[0].tolist()

    async def validate_index(self, driver: GraphDriver, group_id: str) -> None:
        """Validate one group's contract once; rebuilt groups require a fresh client."""
        if group_id in self._validated_group_ids:
            return
        async with self._index_validation_lock:
            if group_id in self._validated_group_ids:
                return
            await validate_graphiti_index_metadata(driver, group_id, self._encoder.profile)
            self._validated_group_ids.add(group_id)

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        """Embed a batch of texts and return their vectors as plain lists of floats."""
        vectors = await asyncio.to_thread(self._encoder.encode, input_data_list)
        return vectors.tolist()


def _single_text(input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]) -> str:
    """Normalize Graphiti's one-text input without stringifying its container."""
    if isinstance(input_data, str):
        return input_data
    if isinstance(input_data, list) and len(input_data) == 1 and isinstance(input_data[0], str):
        return input_data[0]
    msg = "QwenGraphitiEmbedder.create expects one string or a one-item list of strings"
    raise TypeError(msg)
