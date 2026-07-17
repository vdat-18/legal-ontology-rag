"""A Vietnamese-capable embedder for Graphiti backed by BGE-M3.

Graphiti ships OpenAI/Gemini/Voyage embedders; none is tuned for Vietnamese.
This adapter wraps a local ``sentence-transformers`` BGE-M3 model (1024-dim,
matching Graphiti's default ``EMBEDDING_DIM``) behind the ``EmbedderClient``
interface. The blocking ``encode`` call is offloaded to a thread so it does not
stall the event loop.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable

from graphiti_core.embedder.client import EmbedderClient

_DEFAULT_MODEL = "BAAI/bge-m3"


class BGEM3Embedder(EmbedderClient):
    """EmbedderClient implementation using a local BGE-M3 model."""

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        """Load the sentence-transformers model (imported lazily)."""
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def _encode(self, texts: list[str]) -> list[list[float]]:
        """Encode texts to normalised dense vectors (runs in a worker thread)."""
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]

    async def create(
        self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
    ) -> list[float]:
        """Return the embedding for a single input (first item if a list)."""
        texts = [input_data] if isinstance(input_data, str) else [str(x) for x in input_data]
        vectors = await asyncio.to_thread(self._encode, texts)
        return vectors[0]

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        """Return embeddings for a batch of strings."""
        return await asyncio.to_thread(self._encode, input_data_list)
