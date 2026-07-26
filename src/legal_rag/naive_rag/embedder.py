"""BGE-M3 embedding wrapper using SentenceTransformer."""

from __future__ import annotations

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from legal_rag.naive_rag.config import EmbedderConfig


class BGEM3Embedder:
    """Local BGE-M3 embedder producing 1024-dim normalized vectors."""

    def __init__(self, config: EmbedderConfig | None = None) -> None:
        """Initialize the embedder with the given configuration."""
        self.config = config or EmbedderConfig()
        logger.info("Loading embedding model: {}", self.config.model_name)
        self.model = SentenceTransformer(self.config.model_name, device=self.config.device)
        self.dim = self.config.dimension

    def encode(self, texts: str | list[str]) -> np.ndarray:
        """Encode text(s) into L2-normalized dense vectors.

        Args:
            texts: Single string or list of strings to encode.

        Returns:
            Array of shape ``(n, 1024)`` with unit-norm rows.
        """
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=True,
        )
