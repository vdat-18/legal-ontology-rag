"""Semantic and size contracts for ontology LLM prompts."""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROMPT_CASES = (
    (
        "baml_src/ontology_rule_assembly.baml",
        4000,
        (
            "Mục đích bước 3/6",
            "không chứng minh cùng rule",
            "các proposal không trùng member",
            "Không suy diễn role, LegalChange, LegalException",
        ),
    ),
    (
        "baml_src/ontology_change_resolution.baml",
        3200,
        (
            "Mục đích bước 4/6",
            "trực tiếp xác định target, loại thay đổi và ngày hiệu lực",
            "Không suy ra từ similarity",
            "Thiếu evidence thì không đề xuất",
        ),
    ),
    (
        "baml_src/ontology_exception_resolution.baml",
        3000,
        (
            "Mục đích bước 5/6",
            "rule chuyên biệt làm hẹp general rule",
            "citation, provision và target rule riêng",
            "không redirect theo similarity",
        ),
    ),
    (
        "baml_src/ontology_concept_resolution.baml",
        2800,
        (
            "Mục đích bước 6/6",
            "SameConcept chỉ khi hai occurrence có cùng nghĩa pháp lý",
            "Broader/Narrower chỉ khi một nghĩa pháp lý bao hàm nghĩa kia",
            "proposal=null",
        ),
    ),
)


def test_conduct_prompt_separates_locator_context_from_target_semantics() -> None:
    """Step 1 may resolve provision hierarchy without extracting referenced or neighboring conducts."""
    prompt = (PROJECT_ROOT / "baml_src/ontology_population.baml").read_text(encoding="utf-8")
    compact_prompt = " ".join(prompt.split())

    for phrase in (
        "từng nhánh tự là hành vi độc lập",
        "source xem toàn cụm là một hành vi",
        "tham chiếu số ít hoặc predicate/qualifier chung",
        "điều khoản chỉ ánh xạ sang provision khác",
        "trả rỗng dù nó nhắc/mô tả lại hành vi",
        "Locator context chỉ khôi phục Khoản/Điểm",
        "không cấp conduct/evidence",
    ):
        assert phrase in compact_prompt


def test_population_binding_prompt_distinguishes_warning_and_referenced_penalty() -> None:
    """Step 2 preserves two legal effects that are not additional sanctions."""
    prompt = (PROJECT_ROOT / "baml_src/ontology_population.baml").read_text(encoding="utf-8")
    compact_prompt = " ".join(prompt.split())

    for phrase in (
        "Warning cho hình thức cảnh cáo",
        "ReferencedPenalty khi hậu quả được xác định bằng cách dẫn chiếu",
        "AdditionalSanction chỉ khi source xác định là hình thức xử phạt bổ sung",
        "target_label_vi là hành vi hoặc quy tắc được dẫn chiếu",
    ):
        assert phrase in compact_prompt


@pytest.mark.parametrize(("relative_path", "max_length", "required_phrases"), PROMPT_CASES)
def test_downstream_ontology_prompt_is_concise_and_preserves_boundaries(
    relative_path: str,
    max_length: int,
    required_phrases: tuple[str, ...],
) -> None:
    """Each step keeps its semantic gate without repeating schema instructions."""
    prompt = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
    compact_prompt = " ".join(prompt.split())

    assert "Bạn là chuyên gia" not in prompt
    assert "Bối cảnh: xây dựng ontology" in compact_prompt
    assert "chỉ cho quy định xử phạt vi phạm hành chính về giao thông đường bộ Việt Nam" in compact_prompt
    assert '{{ _.role("system") }}' in prompt
    assert '{{ _.role("user") }}' in prompt
    assert "{{ ctx.output_format }}" in prompt
    assert len(prompt) < max_length
    for phrase in required_phrases:
        assert phrase in compact_prompt
