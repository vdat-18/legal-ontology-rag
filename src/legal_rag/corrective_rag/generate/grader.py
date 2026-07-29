"""LLM-based retrieval evaluator for Corrective RAG.

Scores whether a retrieved chunk can answer a user query.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import httpx
from loguru import logger

from legal_rag.corrective_rag.generate.config import RAGConfig


@dataclass(frozen=True, slots=True)
class RelevanceEvaluation:
    """Evaluator output for one query-document pair."""

    score: float
    rationale: str = ""


class CorrectiveGrader:
    """Evaluates retrieved documents with the configured OpenAI-compatible LLM."""

    def __init__(self, config: RAGConfig) -> None:
        """Initialize the grader client."""
        self.config = config
        self.client = httpx.Client(
            base_url=self.config.llm_base_url,
            headers={"Authorization": f"Bearer {self.config.llm_api_key}", "Content-Type": "application/json"},
            timeout=self.config.llm_timeout_seconds,
        )

    def score_document(self, query: str, document_text: str) -> RelevanceEvaluation:
        """Score how relevant a document is to the query."""
        system_prompt = (
            "You are a strict retrieval evaluator for Vietnamese road-traffic-law QA. "
            "Score whether the document can answer the question accurately. "
            'Return only valid JSON: {"score": <number from 0 to 1>, "rationale": "<short Vietnamese reason>"}. '
            "Use 1.0 for direct and complete evidence, 0.5 for partial or ambiguous evidence, "
            "and 0.0 for irrelevant or misleading evidence."
        )
        user_prompt = f"Question:\n{query}\n\nRetrieved document:\n{document_text}"

        payload = {
            "model": self.config.llm_model,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            "temperature": 0.0,
        }

        try:
            logger.debug(f"Grading chunk relevance for query: '{query[:30]}...'")
            response = self.client.post("/chat/completions", json=payload)
            response.raise_for_status()
            response_json = response.json()
        except Exception as e:
            logger.error(f"Error calling LLM for grading: {e}")
            return RelevanceEvaluation(score=0.0, rationale=str(e))
        else:
            choices = response_json.get("choices", [])
            if not choices:
                logger.warning("Empty response choices received from LLM during grading.")
                return RelevanceEvaluation(score=0.0, rationale="empty evaluator response")

            content = choices[0].get("message", {}).get("content", "").strip().lower()
            logger.debug(f"Grader raw response: '{content}'")
            evaluation = _parse_evaluation(content)
            logger.info(f"Chunk relevance score: {evaluation.score:.2f}")
            return evaluation

    def grade_document(self, query: str, document_text: str) -> bool:
        """Evaluate whether a document is strongly relevant to the query."""
        evaluation = self.score_document(query, document_text)
        return evaluation.score >= self.config.evaluator_upper_threshold

    def close(self) -> None:
        """Close the HTTP client connection."""
        self.client.close()


def _parse_evaluation(content: str) -> RelevanceEvaluation:
    """Parse an evaluator JSON response."""
    json_text = _extract_json_object(content)
    if json_text is not None:
        data = json.loads(json_text)
        return RelevanceEvaluation(
            score=_clamp_score(float(data.get("score", 0.0))),
            rationale=str(data.get("rationale", "")),
        )

    if "yes" in content:
        return RelevanceEvaluation(score=1.0, rationale="legacy yes response")
    if "no" in content:
        return RelevanceEvaluation(score=0.0, rationale="legacy no response")

    match = re.search(r"0(?:\.\d+)?|1(?:\.0+)?", content)
    if match:
        return RelevanceEvaluation(score=_clamp_score(float(match.group(0))), rationale="")

    return RelevanceEvaluation(score=0.0, rationale="unparseable evaluator response")


def _extract_json_object(content: str) -> str | None:
    """Extract the first JSON object from text."""
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return content[start : end + 1]


def _clamp_score(value: float) -> float:
    """Clamp a score to the evaluator range."""
    return max(0.0, min(1.0, value))
