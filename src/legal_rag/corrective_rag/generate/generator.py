"""LLM-based response generator for Corrective RAG.

Formulates answers from refined internal and external knowledge items.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from typing import Any

import httpx
from loguru import logger

from legal_rag.corrective_rag.generate.config import RAGConfig


class ResponseGenerator:
    """Sends generation prompts to the configured OpenAI-compatible LLM."""

    def __init__(self, config: RAGConfig) -> None:
        """Initialize the response generator."""
        self.config = config
        self.client = httpx.Client(
            base_url=self.config.llm_base_url,
            headers={"Authorization": f"Bearer {self.config.llm_api_key}", "Content-Type": "application/json"},
            timeout=self.config.llm_timeout_seconds,
        )

    def generate_answer(self, query: str, relevant_chunks: Sequence[Mapping[str, Any] | Any]) -> str:
        """Generate a Vietnamese legal answer from refined knowledge."""
        if not relevant_chunks:
            return (
                "Rat tiec, toi khong tim thay thong tin hoac quy dinh phap luat phu hop "
                "de tra loi cau hoi trong du lieu hien co."
            )

        context = _format_crag_knowledge_context(relevant_chunks)
        system_prompt = (
            "You are a careful Vietnamese road-traffic-law information assistant. "
            "Answer in Vietnamese. This task is allowed public legal information, not legal advice or wrongdoing. "
            "Do not refuse to answer traffic-penalty questions when the provided knowledge contains matching legal "
            "grounds, sanction ranges, license-point deductions, or remedial measures. "
            "Use only the provided knowledge. "
            "When Knowledge1 and Knowledge2 are both present, treat Knowledge1 as refined internal corpus knowledge "
            "and Knowledge2 as external corrective knowledge. "
            "Prefer official legal sources. Cite the source name, and cite the URL for web sources. "
            "If a knowledge item directly answers the question, state the applicable conduct and penalty clearly. "
            "Do not answer about unrelated conduct just because it appears in the context. "
            "If the provided knowledge is insufficient, say that the available documents do not contain enough information."
        )
        user_prompt = f"REFERENCE KNOWLEDGE:\n{context}\n\nQUESTION:\n{query}"

        payload = {
            "model": self.config.llm_model,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            "temperature": 0.0,
        }

        logger.info("Requesting answer generation from LLM...")
        response = self.client.post("/chat/completions", json=payload)
        response.raise_for_status()
        response_json = response.json()
        answer = _message_content(response_json, "generation")
        logger.info("Answer generated successfully.")
        return answer

    def reformulate_query(self, query: str) -> str:
        """Rewrite a user question into a compact legal search query."""
        system_prompt = (
            "You are a Vietnamese road-traffic-law search expert. "
            "Rewrite the user's question into concise legal search keywords. "
            "Include likely statute, decree, violation behavior, sanction, and license-point terms when useful. "
            "Return only the rewritten search query."
        )
        user_prompt = f"Rewrite this question as legal search keywords: {query}"

        payload = {
            "model": self.config.llm_model,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            "temperature": 0.3,
        }

        logger.info("Triggering query reformulation.")
        response = self.client.post("/chat/completions", json=payload)
        response.raise_for_status()
        response_json = response.json()
        new_query = _message_content(response_json, "query reformulation").strip("\"'")
        if not new_query:
            msg = "LLM query reformulation response was empty after trimming quotes."
            raise ValueError(msg)
        logger.info("Query reformulation completed.")
        return new_query

    def close(self) -> None:
        """Close the HTTP client connection."""
        self.client.close()


def _as_mapping(item: Mapping[str, Any] | Any) -> Mapping[str, Any]:
    """Return a mapping view over a knowledge item."""
    if isinstance(item, Mapping):
        return item
    if is_dataclass(item):
        return asdict(item)
    return {}


def _message_content(response_json: dict[str, object], context: str) -> str:
    """Return the first chat completion message content."""
    choices = response_json.get("choices")
    if not isinstance(choices, list) or not choices:
        msg = f"LLM {context} response did not include choices."
        raise ValueError(msg)
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        msg = f"LLM {context} response did not include message content."
        raise ValueError(msg)
    return content.strip()


def _format_crag_knowledge_context(relevant_chunks: Sequence[Mapping[str, Any] | Any]) -> str:
    """Format action-selected CRAG knowledge for answer generation."""
    internal_items: list[Mapping[str, Any]] = []
    external_items: list[Mapping[str, Any]] = []
    for item in relevant_chunks:
        mapped = _as_mapping(item)
        if mapped.get("kind") == "external":
            external_items.append(mapped)
        else:
            internal_items.append(mapped)

    if internal_items and external_items:
        return (
            "Knowledge1:\n"
            f"{_format_knowledge_items(internal_items)}\n\n[sep]\n\n"
            "Knowledge2:\n"
            f"{_format_knowledge_items(external_items)}"
        )
    return _format_knowledge_items(internal_items or external_items)


def _format_knowledge_items(items: Sequence[Mapping[str, Any]]) -> str:
    """Format knowledge items with source metadata."""
    context_parts = []
    for idx, item in enumerate(items, 1):
        source = item.get("source_name", "unknown source")
        text = str(item.get("text", "")).strip()
        kind = item.get("kind", "internal")
        url = item.get("url")
        source_line = f"[{idx}] source_type={kind}; source={source}"
        if url:
            source_line = f"{source_line}; url={url}"
        context_parts.append(f"{source_line}\ncontent:\n{text}")
    return "\n\n".join(context_parts)
