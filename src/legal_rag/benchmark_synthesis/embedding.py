"""OpenAI-compatible embedding inference for benchmark synthesis."""

from __future__ import annotations

import json
from collections.abc import Sequence
from urllib.request import Request, urlopen

import numpy as np
from loguru import logger

from legal_rag.embedding.profiles import EmbeddingProfile
from legal_rag.settings import SynthesisEmbeddingSettings


class SynthesisEmbeddingEncoder:
    """Encode synthesis facts through a dedicated OpenAI-compatible endpoint."""

    def __init__(
        self,
        settings: SynthesisEmbeddingSettings,
        profile: EmbeddingProfile,
        *,
        batch_size: int,
        max_response_bytes: int,
    ) -> None:
        """Bind provider settings and reproducible request bounds."""
        self._settings = settings
        self._profile = profile
        self._batch_size = batch_size
        self._max_response_bytes = max_response_bytes

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return L2-normalized embeddings in the same order as the input texts."""
        batches = [
            self._encode_batch(texts[start : start + self._batch_size])
            for start in range(0, len(texts), self._batch_size)
        ]
        if not batches:
            return np.empty((0, self._profile.dimensions), dtype=np.float32)
        return np.concatenate(batches)

    def _encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        """Send one bounded embedding request and normalize its vectors."""
        body = json.dumps(
            {"model": self._settings.model, "input": list(texts)},
            ensure_ascii=False,
        ).encode("utf-8")
        request = Request(  # noqa: S310 - settings reject non-HTTP provider URLs.
            f"{self._settings.base_url}/embeddings",
            data=body,
            headers={
                "Authorization": f"Bearer {self._settings.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urlopen(request, timeout=self._settings.timeout_seconds) as response:  # noqa: S310
            response_body = response.read(self._max_response_bytes + 1)
        if len(response_body) > self._max_response_bytes:
            msg = f"Embedding response exceeded {self._max_response_bytes} bytes"
            raise ValueError(msg)

        payload = json.loads(response_body)
        records = sorted(payload["data"], key=lambda record: record["index"])
        if [record["index"] for record in records] != list(range(len(texts))):
            msg = "Embedding response indices do not match the request"
            raise ValueError(msg)
        vectors = np.asarray([record["embedding"] for record in records], dtype=np.float32)
        if vectors.shape != (len(texts), self._profile.dimensions):
            msg = f"Embedding response shape {vectors.shape} does not match ({len(texts)}, {self._profile.dimensions})"
            raise ValueError(msg)
        if not np.isfinite(vectors).all():
            msg = "Embedding response contains non-finite values"
            raise ValueError(msg)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms == 0):
            msg = "Embedding response contains a zero vector"
            raise ValueError(msg)
        logger.debug("Encoded {} synthesis facts with {}", len(texts), self._settings.model)
        return vectors / norms
