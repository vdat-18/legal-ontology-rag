"""Generate answers from retrieved graph facts using the shared BAML LLM client."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

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
            text=format_fact_context(fact.fact, source.text, fact.as_of_dates),
            source_name=source.source_name,
        )
        for fact in facts
        for source in fact.sources
    ]
    return client.generate(query, contexts)


def format_fact_context(fact: str, passage: str, as_of_dates: Sequence[date] = ()) -> str:
    """Combine the source passage with the fact extracted from it.

    The passage comes first because quoting a provision needs its actual wording — the article
    heading naming the vehicle, the clause numbering, the exceptions — all of which the extracted
    fact has already dropped. The fact follows as the model's shortcut to what the passage
    established, for passages long enough that the relevant sentence is easy to miss.
    """
    content = fact if not passage else f"{passage}\n\n[Trích xuất] {fact}"
    if not as_of_dates:
        return content
    dates = ", ".join(item.isoformat() for item in as_of_dates)
    return f"[Có hiệu lực tại mốc truy hồi: {dates}]\n{content}"
