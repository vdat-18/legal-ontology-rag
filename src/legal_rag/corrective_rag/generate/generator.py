"""LLM-based response generator for Corrective RAG.

Formulates answers from refined internal and external knowledge items.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from typing import Any

import httpx
from loguru import logger

from legal_rag.baml_transport import record_openai_usage
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

    def generate_answer(
        self,
        query: str,
        relevant_chunks: Sequence[Mapping[str, Any] | Any],
        *,
        original_query: str | None = None,
    ) -> str:
        """Generate a Vietnamese legal answer from refined knowledge.

        ``query`` is the standalone question used throughout retrieval. For a
        context-dependent follow-up, ``original_query`` preserves the user's
        wording so generation answers that intent without treating chat history
        as legal evidence.
        """
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
        question_prompt = f"QUESTION:\n{query}"
        if original_query and original_query != query:
            question_prompt = (
                f"ORIGINAL USER QUESTION:\n{original_query}\n\nSTANDALONE QUESTION RESOLVED FROM CONVERSATION:\n{query}"
            )
        user_prompt = f"REFERENCE KNOWLEDGE:\n{context}\n\n{question_prompt}"

        payload = self._completion_payload(
            model=self.config.llm_model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=self.config.answer_temperature,
            max_tokens=self.config.answer_max_output_tokens,
        )

        logger.info("Requesting answer generation from LLM...")
        response_json = self._post_completion(payload, "generation")
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

        payload = self._completion_payload(
            model=self.config.llm_model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=self.config.query_reformulation_temperature,
            max_tokens=self.config.query_reformulation_max_output_tokens,
        )

        logger.info("Triggering query reformulation.")
        response_json = self._post_completion(payload, "query reformulation")
        new_query = _message_content(response_json, "query reformulation").strip("\"'")
        if not new_query:
            msg = "LLM query reformulation response was empty after trimming quotes."
            raise ValueError(msg)
        logger.info("Query reformulation completed.")
        return new_query

    def _completion_payload(self, **standard_fields: object) -> dict[str, object]:
        """Build one bounded request while preserving configured provider fields."""
        payload = dict(self.config.llm_extra_request_body)
        payload.update(standard_fields)
        payload["stream"] = False
        return payload

    def _post_completion(self, payload: Mapping[str, object], context: str) -> dict[str, object]:
        """Post one completion and read at most the configured response size."""
        response_body = bytearray()
        with self.client.stream("POST", "/chat/completions", json=payload) as response:
            response.raise_for_status()
            for chunk in response.iter_bytes():
                if len(chunk) > self.config.max_response_bytes - len(response_body):
                    msg = f"LLM {context} response exceeded {self.config.max_response_bytes} bytes."
                    raise ValueError(msg)
                response_body.extend(chunk)
        try:
            response_json = json.loads(response_body)
        except json.JSONDecodeError as error:
            msg = f"LLM {context} response was not valid JSON."
            raise ValueError(msg) from error
        if not isinstance(response_json, dict):
            msg = f"LLM {context} response must be a JSON object."
            raise TypeError(msg)
        record_openai_usage(response_json)
        return response_json

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
    choice = choices[0]
    if isinstance(choice, dict) and choice.get("finish_reason") == "length":
        msg = f"LLM {context} response stopped at the configured output-token limit."
        raise ValueError(msg)
    message = choice.get("message") if isinstance(choice, dict) else None
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
