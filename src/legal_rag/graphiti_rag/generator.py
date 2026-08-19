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
        BamlRetrievedChunkContext(
            chunk_id=source.chunk_id,
            text=_context_text(fact.fact, source.text),
            source_name=source.source_name,
        )
        for fact in facts
        for source in fact.sources
    ]
    return client.generate(query, contexts)


def _context_text(fact: str, passage: str) -> str:
    """Combine the source passage with the fact extracted from it.

    The passage comes first because quoting a provision needs its actual wording — the article
    heading naming the vehicle, the clause numbering, the exceptions — all of which the extracted
    fact has already dropped. The fact follows as the model's shortcut to what the passage
    established, for passages long enough that the relevant sentence is easy to miss.
    """
    if not passage:
        return fact
    return f"{passage}\n\n[Trích xuất] {fact}"
