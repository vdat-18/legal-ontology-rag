"""Tests for CRAG answer-generation prompts."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace

import httpx

from legal_rag.baml_transport import capture_llm_usage
from legal_rag.corrective_rag.generate.generator import ResponseGenerator


def test_generation_prompt_does_not_refuse_public_traffic_penalty_questions() -> None:
    """Traffic-penalty answers are public legal information when supported by knowledge."""
    generator = ResponseGenerator(_llm_config())
    client = _CapturingLlmClient()
    generator.client = client

    answer = generator.generate_answer(
        "Xe mô tô không chấp hành hiệu lệnh của đèn tín hiệu giao thông bị phạt bao nhiêu tiền?",
        [
            {
                "kind": "internal",
                "source_name": "168_2024_ND-CP_619502.docx",
                "text": (
                    "7. Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng đối với người điều khiển xe "
                    "thực hiện hành vi không chấp hành hiệu lệnh của đèn tín hiệu giao thông."
                ),
            }
        ],
    )

    assert answer == "answer"
    payload = client.payload
    assert payload["temperature"] == 0.0
    assert payload["max_tokens"] == 2048
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["stream"] is False
    system_prompt = payload["messages"][0]["content"]
    assert "allowed public legal information" in system_prompt
    assert "Do not refuse" in system_prompt
    assert "Do not answer about unrelated conduct" in system_prompt


def test_multi_turn_generation_preserves_original_and_standalone_questions() -> None:
    """The answer prompt retains user wording after the follow-up is resolved."""
    generator = ResponseGenerator(_llm_config())
    client = _CapturingLlmClient()
    generator.client = client

    generator.generate_answer(
        "Xe mô tô không chấp hành đèn tín hiệu giao thông bị phạt bao nhiêu?",
        [{"kind": "internal", "source_name": "source", "text": "Phạt tiền."}],
        original_query="Còn xe máy thì sao?",
    )

    user_prompt = client.payload["messages"][1]["content"]
    assert "ORIGINAL USER QUESTION:\nCòn xe máy thì sao?" in user_prompt
    assert "STANDALONE QUESTION RESOLVED FROM CONVERSATION" in user_prompt
    assert "Xe mô tô không chấp hành đèn tín hiệu giao thông bị phạt bao nhiêu?" in user_prompt


def test_query_reformulation_uses_its_own_output_bound() -> None:
    """Web-search reformulation keeps its smaller configured completion budget."""
    generator = ResponseGenerator(_llm_config())
    client = _CapturingLlmClient()
    generator.client = client

    rewritten = generator.reformulate_query("Xe máy vượt đèn đỏ bị phạt gì?")

    assert rewritten == "answer"
    assert client.payload["max_tokens"] == 512
    assert client.payload["temperature"] == 0.3
    assert client.payload["thinking"] == {"type": "disabled"}


def test_generation_reports_provider_token_usage_to_the_active_evaluation() -> None:
    """The direct CRAG transport participates in shared per-sample usage accounting."""
    generator = ResponseGenerator(_llm_config())
    generator.client = _CapturingLlmClient()

    with capture_llm_usage() as usage:
        generator.generate_answer("question", [{"kind": "internal", "text": "context"}])

    assert usage.to_dict()["total_tokens"] == 15


def _llm_config() -> SimpleNamespace:
    """Return only the settings needed by the generator."""
    return SimpleNamespace(
        llm_base_url="http://provider.test/v1",
        llm_api_key="test-key",
        llm_model="test-model",
        llm_timeout_seconds=1,
        llm_extra_request_body={"thinking": {"type": "disabled"}},
        answer_max_output_tokens=2048,
        answer_temperature=0.0,
        query_reformulation_max_output_tokens=512,
        query_reformulation_temperature=0.3,
        max_response_bytes=4096,
    )


class _CapturingLlmClient:
    """HTTP client fake that captures the generation payload."""

    def __init__(self) -> None:
        self.payload: dict[str, object] = {}

    @contextmanager
    def stream(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, object],
    ) -> Iterator[httpx.Response]:
        """Capture the payload and return one answer."""
        self.payload = json
        request = httpx.Request(method, f"http://provider.test{path}")
        yield httpx.Response(
            200,
            json={
                "model": "test-model",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                "choices": [{"message": {"content": "answer"}}],
            },
            request=request,
        )
