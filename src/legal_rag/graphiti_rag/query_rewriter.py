"""Rewrite colloquial legal questions into formal terminology before graph search.

Retrieval quality on the Graphiti graph is sensitive to vocabulary: the graph stores facts
using the formal terms legal documents actually use (e.g. "không chấp hành hiệu lệnh của đèn
tín hiệu giao thông"), while users naturally ask in colloquial terms (e.g. "vượt đèn đỏ").
A colloquial query can fail to retrieve the correct fact even though it exists in the graph.
Rewriting the query into formal terminology before searching closes most of this gap.
"""

from __future__ import annotations

from typing import Protocol

from openai import AsyncOpenAI

from legal_rag.settings import Settings

_SYSTEM_PROMPT = (
    "Bạn là chuyên gia thuật ngữ pháp luật giao thông đường bộ Việt Nam. Diễn đạt lại câu hỏi "
    "sau đây bằng đúng thuật ngữ pháp lý chính thức thường dùng trong nghị định, thông tư "
    "(ví dụ: 'vượt đèn đỏ' -> 'không chấp hành hiệu lệnh của đèn tín hiệu giao thông'), giữ "
    "nguyên ý nghĩa và đối tượng của câu hỏi gốc. Chỉ trả về câu hỏi đã diễn đạt lại, không "
    "thêm giải thích hay tiền tố nào khác."
)


class QueryRewriteClient(Protocol):
    """Boundary implemented by a query-rewriting language-model client."""

    async def rewrite(self, query: str) -> str:
        """Return *query* rewritten into formal legal terminology."""


class OpenAIQueryRewriteClient:
    """Rewrites colloquial queries into formal legal terminology via the configured LLM."""

    def __init__(self, settings: Settings) -> None:
        """Create a client from the project's OpenAI-compatible LLM settings."""
        self._client = AsyncOpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)
        self._model = settings.llm_model

    async def rewrite(self, query: str) -> str:
        """Return *query* rewritten into formal legal terminology, or the original on failure.

        A rewrite failure (empty response, API error) degrades to the original query rather
        than blocking the pipeline — retrieval with the original query is still better than no
        retrieval at all.
        """
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                temperature=0.0,
                max_tokens=200,
                extra_body={"thinking": {"type": "disabled"}},
            )
        except Exception:
            return query
        rewritten = (response.choices[0].message.content or "").strip()
        return rewritten or query
