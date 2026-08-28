"""Concurrent, resumable OpenAI-compatible embedding inference for benchmark synthesis."""

from __future__ import annotations

import http.client
import json
import tempfile
import time
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
from loguru import logger

from legal_rag.embedding.profiles import EmbeddingProfile
from legal_rag.io_utils import write_json_atomic
from legal_rag.parser.hashing import sha256_json
from legal_rag.settings import SynthesisEmbeddingSettings

EMBEDDING_CHECKPOINT_SCHEMA_VERSION = "1.0.0"

_RETRYABLE_EMBEDDING_ERRORS = (
    http.client.HTTPException,
    OSError,
    ValueError,
    TypeError,
    KeyError,
)


@dataclass(frozen=True, slots=True)
class _EmbeddingBatch:
    """One ordered embedding request."""

    index: int
    texts: tuple[str, ...]

    @property
    def task_key(self) -> str:
        """Return a stable log and checkpoint key."""
        return f"embedding-batch:{self.index}"


@dataclass(frozen=True, slots=True)
class _EmbeddingBatchFailure:
    """One batch that exhausted retries."""

    batch: _EmbeddingBatch
    error: str


class _EmbeddingBatchExhaustedError(RuntimeError):
    """Raised when one embedding batch exhausts its attempts."""

    def __init__(self, error: Exception) -> None:
        """Preserve the final provider or validation failure in the message."""
        super().__init__(f"exhausted retries: {error}")


class IncompleteEmbeddingBatchesError(RuntimeError):
    """Raised after all runnable batches finish but some exhaust retries."""

    def __init__(self, failures: Sequence[_EmbeddingBatchFailure]) -> None:
        """Describe the batches that remain incomplete for the next resume."""
        details = "; ".join(f"{failure.batch.task_key}: {failure.error}" for failure in failures)
        super().__init__(f"Embedding left {len(failures)} incomplete batch(es): {details}")


class SynthesisEmbeddingEncoder:
    """Encode synthesis facts concurrently through a dedicated OpenAI-compatible endpoint."""

    def __init__(
        self,
        settings: SynthesisEmbeddingSettings,
        profile: EmbeddingProfile,
        *,
        batch_size: int,
        max_concurrency: int = 1,
        max_retries: int = 0,
        retry_delay_seconds: float = 0,
        max_response_bytes: int,
        checkpoint_directory: Path | None = None,
        run_id: str | None = None,
    ) -> None:
        """Bind provider settings, execution bounds, and optional checkpoint identity."""
        if batch_size < 1 or max_concurrency < 1 or max_response_bytes < 1:
            msg = "Embedding batch size, concurrency, and response bound must be positive"
            raise ValueError(msg)
        if max_retries < 0 or retry_delay_seconds < 0:
            msg = "Embedding retry count and delay must not be negative"
            raise ValueError(msg)
        if (checkpoint_directory is None) != (run_id is None):
            msg = "Embedding checkpoint_directory and run_id must be configured together"
            raise ValueError(msg)
        self._settings = settings
        self._profile = profile
        self._batch_size = batch_size
        self._max_concurrency = max_concurrency
        self._max_retries = max_retries
        self._retry_delay_seconds = retry_delay_seconds
        self._max_response_bytes = max_response_bytes
        self._checkpoint_directory = checkpoint_directory
        self._run_id = run_id

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return L2-normalized embeddings in input order, resuming completed batches."""
        batches = tuple(
            _EmbeddingBatch(index, tuple(texts[start : start + self._batch_size]))
            for index, start in enumerate(range(0, len(texts), self._batch_size))
        )
        if not batches:
            return np.empty((0, self._profile.dimensions), dtype=np.float32)

        vectors_by_batch = self._load_checkpoints(texts, batches)
        remaining_batches = tuple(batch for batch in batches if batch.index not in vectors_by_batch)
        logger.info(
            "Embedding started: {} texts, {} batches, batch_size={}, max_concurrency={}, {} cached",
            len(texts),
            len(batches),
            self._batch_size,
            self._max_concurrency,
            len(vectors_by_batch),
        )
        failures: list[_EmbeddingBatchFailure] = []
        if remaining_batches:
            with ThreadPoolExecutor(max_workers=self._max_concurrency) as executor:
                futures: dict[Future[np.ndarray], _EmbeddingBatch] = {
                    executor.submit(self._encode_batch_with_retries, batch): batch for batch in remaining_batches
                }
                for future in as_completed(futures):
                    batch = futures[future]
                    try:
                        vectors = future.result()
                    except RuntimeError as error:
                        failures.append(_EmbeddingBatchFailure(batch=batch, error=str(error)))
                        logger.error("{} remained incomplete: {}", batch.task_key, error)
                        continue
                    vectors_by_batch[batch.index] = vectors
                    self._write_checkpoint(batch, vectors)
                    logger.info(
                        "Embedding batch {}/{} completed: texts={}",
                        batch.index + 1,
                        len(batches),
                        len(batch.texts),
                    )
        if failures:
            raise IncompleteEmbeddingBatchesError(sorted(failures, key=lambda failure: failure.batch.index))
        return np.concatenate([vectors_by_batch[batch.index] for batch in batches])

    def _encode_batch_with_retries(self, batch: _EmbeddingBatch) -> np.ndarray:
        """Encode one batch with bounded retries."""
        logger.info("Embedding batch {} started: texts={}", batch.index + 1, len(batch.texts))
        for attempt in range(self._max_retries + 1):
            try:
                return self._encode_batch(batch.texts)
            except _RETRYABLE_EMBEDDING_ERRORS as error:
                logger.warning(
                    "{} failed on attempt {}/{}: {}",
                    batch.task_key,
                    attempt + 1,
                    self._max_retries + 1,
                    error,
                )
                if attempt == self._max_retries:
                    raise _EmbeddingBatchExhaustedError(error) from error
                logger.info("Retrying {} in {} seconds", batch.task_key, self._retry_delay_seconds)
                time.sleep(self._retry_delay_seconds)
        raise AssertionError("unreachable")

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
        self._validate_vectors(vectors, len(texts))
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        logger.debug("Encoded {} synthesis facts with {}", len(texts), self._settings.model)
        return vectors / norms

    def _load_checkpoints(
        self,
        texts: Sequence[str],
        batches: Sequence[_EmbeddingBatch],
    ) -> dict[int, np.ndarray]:
        """Initialize checkpoint identity and load every completed batch."""
        if self._checkpoint_directory is None:
            return {}
        expected_manifest = self._checkpoint_manifest(texts)
        manifest_path = self._checkpoint_directory / "manifest.json"
        self._checkpoint_directory.mkdir(parents=True, exist_ok=True)
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest != expected_manifest:
                msg = "Existing embedding checkpoints belong to a different candidate-selection run"
                raise ValueError(msg)
        else:
            if any(self._checkpoint_directory.iterdir()):
                msg = "Embedding checkpoint directory contains files without a manifest"
                raise ValueError(msg)
            write_json_atomic(manifest_path, expected_manifest)

        loaded: dict[int, np.ndarray] = {}
        for batch in batches:
            checkpoint_path = self._batch_checkpoint_path(batch)
            if not checkpoint_path.exists():
                continue
            vectors = np.load(checkpoint_path, allow_pickle=False)
            self._validate_vectors(vectors, len(batch.texts), require_normalized=True)
            loaded[batch.index] = vectors
        return loaded

    def _checkpoint_manifest(self, texts: Sequence[str]) -> dict[str, object]:
        """Return the exact identity required to reuse persisted vectors."""
        return {
            "schema_version": EMBEDDING_CHECKPOINT_SCHEMA_VERSION,
            "run_id": self._run_id,
            "input_sha256": sha256_json(list(texts)),
            "text_count": len(texts),
            "batch_size": self._batch_size,
            "dimensions": self._profile.dimensions,
            "embedding_model": self._settings.model,
            "embedding_profile": self._profile.profile_id,
        }

    def _write_checkpoint(self, batch: _EmbeddingBatch, vectors: np.ndarray) -> None:
        """Atomically persist one completed batch without rewriting prior batches."""
        if self._checkpoint_directory is None:
            return
        checkpoint_path = self._batch_checkpoint_path(batch)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=self._checkpoint_directory, delete=False) as temporary_file:
                temporary_path = Path(temporary_file.name)
                np.save(temporary_file, vectors, allow_pickle=False)
            temporary_path.replace(checkpoint_path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def _batch_checkpoint_path(self, batch: _EmbeddingBatch) -> Path:
        """Return one deterministic batch checkpoint path."""
        if self._checkpoint_directory is None:
            raise AssertionError
        return self._checkpoint_directory / f"batch-{batch.index:06d}.npy"

    def _validate_vectors(
        self,
        vectors: np.ndarray,
        expected_rows: int,
        *,
        require_normalized: bool = False,
    ) -> None:
        """Reject malformed provider responses and corrupt checkpoints."""
        expected_shape = (expected_rows, self._profile.dimensions)
        if vectors.shape != expected_shape:
            msg = f"Embedding response shape {vectors.shape} does not match {expected_shape}"
            raise ValueError(msg)
        if not np.isfinite(vectors).all():
            msg = "Embedding response contains non-finite values"
            raise ValueError(msg)
        norms = np.linalg.norm(vectors, axis=1)
        if np.any(norms == 0):
            msg = "Embedding response contains a zero vector"
            raise ValueError(msg)
        if require_normalized and not np.allclose(norms, 1.0, atol=1e-5):
            msg = "Embedding checkpoint contains non-normalized vectors"
            raise ValueError(msg)
