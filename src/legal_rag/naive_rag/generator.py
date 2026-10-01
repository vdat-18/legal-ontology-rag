"""Generate answers from retrieved context using the shared BAML LLM client."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from legal_rag.baml_client.sync_client import b
from legal_rag.baml_client.types import RetrievedChunkContext as BamlRetrievedChunkContext
from legal_rag.baml_transport import BamlCaller
from legal_rag.naive_rag.models import RetrievedChunk
from legal_rag.settings import NaiveRagSettings, Settings


class GenerationClient(Protocol):
    """Boundary implemented by an answer-generation language-model client."""

    def generate(self, query_vi: str, chunks: Sequence[BamlRetrievedChunkContext]) -> str:
        """Return one generated answer, in Vietnamese, grounded in *chunks*."""


class BamlGenerationClient:
    """Typed BAML client backed by the shared runtime OpenAI-compatible provider."""

    def __init__(self, settings: Settings, naive_rag_settings: NaiveRagSettings) -> None:
        """Create a runtime BAML client from typed, secret-safe configuration."""
        self._caller = BamlCaller.from_settings(
            settings,
            max_output_tokens=naive_rag_settings.answer_max_output_tokens,
            temperature=naive_rag_settings.answer_temperature,
            max_response_bytes=naive_rag_settings.max_response_bytes,
        )

    def generate(self, query_vi: str, chunks: Sequence[BamlRetrievedChunkContext]) -> str:
        """Run one bounded BAML call and return the generated answer text."""
        request = b.request.GenerateAnswer(query_vi, list(chunks), baml_options=self._caller.options)
        raw_output = self._caller.execute(request)
        return b.parse.GenerateAnswer(raw_output, baml_options=self._caller.options).answer_vi


def generate_answer(query: str, hits: Sequence[RetrievedChunk], client: GenerationClient) -> str:
    """Generate one answer grounded in the retrieved chunks."""
    chunk_contexts = [
        BamlRetrievedChunkContext(chunk_id=hit.chunk_id, text=hit.text, source_name=hit.source_name) for hit in hits
    ]
    return client.generate(query, chunk_contexts)
