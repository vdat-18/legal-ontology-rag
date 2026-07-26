"""Generate answers from retrieved context using Gemini."""

from __future__ import annotations

from google import genai
from google.genai import types
from loguru import logger

from legal_rag.naive_rag.config import GeneratorConfig
from legal_rag.naive_rag.retriever import RetrievedChunk

_SYSTEM_PROMPT = (
    "Bạn là trợ lý pháp luật giao thông Việt Nam. "
    "Trả lời câu hỏi dựa HOÀN TOÀN vào các đoạn trích pháp luật được cung cấp.\n\n"
    "Quy tắc:\n"
    "- Chỉ trả lời dựa trên thông tin trong các đoạn trích. Nếu không tìm thấy, nói rõ.\n"
    "- Trích dẫn nguồn (tên văn bản, điều khoản) khi trả lời.\n"
    "- Nêu rõ mức phạt tiền, hình phạt bổ sung nếu có.\n"
    "- Trả lời bằng tiếng Việt, ngắn gọn, chính xác."
)


def _build_prompt(query: str, hits: list[RetrievedChunk]) -> str:
    """Combine retrieved chunks and the user query into a single prompt."""
    context_parts = [f"[Đoạn {i} — {hit.source_name}]\n{hit.text}" for i, hit in enumerate(hits, 1)]
    context = "\n\n".join(context_parts)
    return f"Đoạn trích pháp luật:\n\n{context}\n\n---\nCâu hỏi: {query}"


async def generate(query: str, hits: list[RetrievedChunk], config: GeneratorConfig | None = None) -> str:
    """Send retrieved context and query to Gemini and return the answer.

    Args:
        query: User question.
        hits: Retrieved chunks providing legal context.
        config: Generator settings. Uses defaults when *None*.

    Returns:
        Model-generated answer string.
    """
    cfg = config or GeneratorConfig()
    if not cfg.api_key:
        msg = "GEMINI_API_KEY is not set"
        raise ValueError(msg)

    client = genai.Client(api_key=cfg.api_key)
    user_prompt = _build_prompt(query, hits)

    gen_config = types.GenerateContentConfig(
        temperature=cfg.temperature,
        max_output_tokens=cfg.max_output_tokens,
        system_instruction=_SYSTEM_PROMPT,
    )
    response = await client.aio.models.generate_content(
        model=cfg.model,
        contents=[user_prompt],
        config=gen_config,
    )
    logger.debug("Generated answer ({} chars) for: {:.60s}", len(response.text or ""), query)
    return response.text or ""
