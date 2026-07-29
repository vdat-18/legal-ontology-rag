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
            "You are a careful Vietnamese road-traffic-law assistant. "
            "Answer in Vietnamese. Use only the provided knowledge. "
            "When Knowledge1 and Knowledge2 are both present, treat Knowledge1 as refined internal corpus knowledge "
            "and Knowledge2 as external corrective knowledge. "
            "Prefer official legal sources. Cite the source name, and cite the URL for web sources. "
            "If the provided knowledge is insufficient, say that the available documents do not contain enough information."
        )
        user_prompt = f"REFERENCE KNOWLEDGE:\n{context}\n\nQUESTION:\n{query}"

        payload = {
            "model": self.config.llm_model,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            "temperature": 0.2,
        }

        try:
            logger.info("Requesting answer generation from LLM...")
            response = self.client.post("/chat/completions", json=payload)
            response.raise_for_status()
            response_json = response.json()
        except Exception as e:
            logger.error(f"Error calling LLM for generation: {e}")
            return f"Da xay ra loi khi ket noi voi mo hinh ngon ngu de tao cau tra loi: {e}"
        else:
            choices = response_json.get("choices", [])
            if not choices:
                return "Loi: Khong nhan duoc phan hoi tu mo hinh ngon ngu."

            answer = choices[0].get("message", {}).get("content", "").strip()
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

        try:
            logger.info(f"Triggering query reformulation for: '{query}'")
            response = self.client.post("/chat/completions", json=payload)
            response.raise_for_status()
            response_json = response.json()
        except Exception as e:
            logger.error(f"Error during query reformulation: {e}")
            return query

        choices = response_json.get("choices", [])
        if not choices:
            return query

        new_query = choices[0].get("message", {}).get("content", "").strip()
        new_query = new_query.strip("\"'")
        logger.info(f"Query reformulated to: '{new_query}'")
        return new_query or query

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
