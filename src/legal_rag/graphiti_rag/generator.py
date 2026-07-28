"""Generate answers from retrieved graph facts using the shared BAML LLM client."""

from __future__ import annotations

from collections.abc import Sequence

from legal_rag.baml_client.types import RetrievedChunkContext as BamlRetrievedChunkContext
from legal_rag.graphiti_rag.retriever import RetrievedFact
from legal_rag.naive_rag.generator import GenerationClient


def generate_answer(query: str, facts: Sequence[RetrievedFact], client: GenerationClient) -> str:
    """Generate one answer grounded in the selected graph facts.

    A fact supported by several episodes contributes one context per supporting source, so every
    chunk that reinforces the fact is cited rather than only the first.
    """
    contexts = [
        BamlRetrievedChunkContext(chunk_id=source.chunk_id, text=fact.fact, source_name=source.source_name)
        for fact in facts
        for source in fact.sources
    ]
    return client.generate(query, contexts)
