"""Filter low-signal chunks before spending fact-extraction LLM calls on them."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from baml_py.logging import set_log_level

from legal_rag.baml_client.sync_client import b
from legal_rag.baml_transport import build_client_registry, execute_baml_request
from legal_rag.chunker.models import Chunk
from legal_rag.settings import EvaluationSettings, Settings


@dataclass(frozen=True, slots=True)
class ChunkQualityScore:
    """Four-criteria quality score for one chunk."""

    clarity: float
    depth: float
    structure: float
    relevance: float

    @property
    def average(self) -> float:
        """Return the unweighted mean of the four criteria."""
        return (self.clarity + self.depth + self.structure + self.relevance) / 4


@dataclass(frozen=True, slots=True)
class ChunkQualityExclusion:
    """One chunk excluded from fact extraction, with an auditable reason."""

    chunk_id: str
    score: ChunkQualityScore
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable exclusion record."""
        return {
            "chunk_id": self.chunk_id,
            "clarity": self.score.clarity,
            "depth": self.score.depth,
            "structure": self.score.structure,
            "relevance": self.score.relevance,
            "reason": self.reason,
        }


class ChunkQualityClient(Protocol):
    """Boundary implemented by a chunk-quality scoring language-model client."""

    def evaluate(self, chunk_text: str) -> ChunkQualityScore:
        """Return one typed quality score."""


class BamlChunkQualityClient:
    """Typed BAML client backed by a runtime OpenAI-compatible provider."""

    def __init__(self, settings: Settings, evaluation_settings: EvaluationSettings) -> None:
        """Create a runtime BAML client from typed, secret-safe configuration."""
        set_log_level("OFF")
        self._registry = build_client_registry(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            max_output_tokens=evaluation_settings.chunk_quality_max_output_tokens,
            temperature=evaluation_settings.chunk_quality_temperature,
            timeout_seconds=settings.llm_timeout_seconds,
        )
        self._timeout_seconds = settings.llm_timeout_seconds
        self._max_response_bytes = evaluation_settings.max_response_bytes

    def evaluate(self, chunk_text: str) -> ChunkQualityScore:
        """Run one bounded BAML call and return its generated typed result."""
        request = b.request.EvaluateChunkQuality(chunk_text, baml_options={"client_registry": self._registry})
        raw_output = execute_baml_request(request, self._timeout_seconds, self._max_response_bytes)
        result = b.parse.EvaluateChunkQuality(raw_output, baml_options={"client_registry": self._registry})
        return ChunkQualityScore(
            clarity=result.clarity,
            depth=result.depth,
            structure=result.structure,
            relevance=result.relevance,
        )


def filter_chunks_by_quality(
    chunks: Sequence[Chunk],
    client: ChunkQualityClient,
    *,
    threshold: float,
) -> tuple[tuple[Chunk, ...], tuple[ChunkQualityExclusion, ...]]:
    """Split chunks into those worth reading for fact extraction and those excluded."""
    accepted: list[Chunk] = []
    excluded: list[ChunkQualityExclusion] = []
    for chunk in chunks:
        score = client.evaluate(chunk.text)
        if score.average >= threshold:
            accepted.append(chunk)
        else:
            reason = f"average score {score.average:.2f} below threshold {threshold:.2f}"
            excluded.append(ChunkQualityExclusion(chunk_id=chunk.chunk_id, score=score, reason=reason))
    return tuple(accepted), tuple(excluded)
