"""Embedding utility for generating dense vector representations of text.

Uses sentence-transformers to run inference with the configured local embedding model.
"""

from __future__ import annotations

import gc
import os
from collections.abc import Sequence

import torch
from loguru import logger
from sentence_transformers import SentenceTransformer

from legal_rag.embedding.profiles import load_configured_embedding_profile


class EmbeddingModel:
    """Helper class to load a sentence-transformer model and generate embeddings."""

    def __init__(self, profile_id: str) -> None:
        """Initialize the embedding model using a project profile.

        Args:
            profile_id: Identifier of the embedding profile to load.
        """
        logger.info(f"Loading embedding profile: {profile_id}")
        profile = load_configured_embedding_profile(profile_id)
        self.model_id = profile.model_id
        self.dimensions = profile.dimensions
        self.max_input_tokens = profile.max_input_tokens
        self.query_instruction = profile.query_instruction

        # Determine the execution device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing SentenceTransformer with '{self.model_id}' on device: {self.device}")

        local_files_only = _read_env_bool("EMBEDDING_LOCAL_FILES_ONLY")
        self.model = SentenceTransformer(
            self.model_id,
            revision=profile.model_revision,
            device=self.device,
            local_files_only=local_files_only,
        )
        logger.info("Embedding model loaded successfully.")

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate embeddings for a list of document strings.

        Args:
            texts: List of strings to embed.

        Returns:
            A list of dense vector representations (lists of floats).
        """
        if not texts:
            return []

        logger.info(f"Generating embeddings for {len(texts)} documents...")
        embeddings = self.model.encode(list(texts), batch_size=32, show_progress_bar=False, convert_to_numpy=True)
        # Convert numpy array to list of floats
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Generate an embedding for a single search query.

        Args:
            text: Query string to embed.

        Returns:
            A dense vector representation of the query (list of floats).
        """
        # Apply query instruction if configured (e.g. for Qwen)
        formatted_query = text
        if self.query_instruction:
            formatted_query = f"{self.query_instruction}{text}"

        logger.debug(f"Embedding query: '{formatted_query[:60]}...'")
        embedding = self.model.encode(formatted_query, convert_to_numpy=True)
        return embedding.tolist()

    def close(self) -> None:
        """Release resources and clear cache."""
        logger.info("Cleaning up embedding model resources...")
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def _read_env_bool(name: str) -> bool:
    """Read a boolean flag from process environment variables."""
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}
