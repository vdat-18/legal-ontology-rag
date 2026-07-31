"""Tests for CRAG answer-generation prompts."""

from __future__ import annotations

from types import SimpleNamespace

import httpx

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
    system_prompt = payload["messages"][0]["content"]
    assert "allowed public legal information" in system_prompt
    assert "Do not refuse" in system_prompt
    assert "Do not answer about unrelated conduct" in system_prompt


def _llm_config() -> SimpleNamespace:
    """Return only the settings needed by the generator."""
    return SimpleNamespace(
        llm_base_url="http://provider.test/v1",
        llm_api_key="test-key",
        llm_model="test-model",
        llm_timeout_seconds=1,
    )


class _CapturingLlmClient:
    """HTTP client fake that captures the generation payload."""

    def __init__(self) -> None:
        self.payload: dict[str, object] = {}

    def post(self, path: str, *, json: dict[str, object]) -> httpx.Response:
        """Capture the payload and return one answer."""
        self.payload = json
        request = httpx.Request("POST", f"http://provider.test{path}")
        return httpx.Response(200, json={"choices": [{"message": {"content": "answer"}}]}, request=request)
