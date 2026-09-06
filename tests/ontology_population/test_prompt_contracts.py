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
            "Giữ riêng các động từ/hành động pháp lý độc lập",
            "Phần bổ ngữ bị lược để tránh lặp",
            "không chứng minh cùng rule",
            "các proposal không trùng member",
            "Không suy diễn role, LegalChange, LegalException",
        ),
    ),
    (
        "baml_src/ontology_rule_assembly_resolution.baml",
        3000,
        (
            "Mục đích bước 3",
            "tạo kết quả nhất quán toàn cục",
            "Giữ riêng các động từ/hành động pháp lý độc lập",
            "Phần bổ ngữ bị lược để tránh lặp",
            "các proposal trong một component không trùng member",
            "trả [] nếu chưa đủ căn cứ",
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
        "Cổng bắt buộc",
        "gắn conduct với cảnh cáo, tiền phạt",
        "sửa/thay Khoản/Điểm xác định",
        "nội dung trực tiếp liệt kê conduct vi phạm giao thông",
        "không cần lặp hậu quả của điều khoản đích",
        "Loại kỷ luật, hình sự/dân sự, cưỡng chế/nghĩa vụ thi hành",
        "thủ tục chung nhiều lĩnh vực",
        "Không tự đảo nghĩa vụ tích cực thành vi phạm",
        "Audit phân rã bắt buộc",
        "xác định mọi động từ đầu",
        "mỗi hành động/không hành động tự đủ bị phạt là một conduct lá",
        "Động từ đầu độc lập vẫn là lá riêng",
        "P X hoặc Q Y",
        "P bị A, Q, R",
        "khi Q, R là động từ đầu",
        "Cùng động từ đầu, chỉ khác công cụ/thuộc tính/phương thức/ngưỡng thì một lá",
        "P bằng A/B/C",
        "Chỉ gộp động từ đầu khác nhau nếu chúng phải cùng xảy ra",
        "Nhánh rút gọn đã có động từ đầu thì bắt đầu nhãn tại đó",
        "chỉ khôi phục bổ ngữ dùng chung",
        "không mang động từ nhánh khác",
        "Role/loại xe để bước 2 bind",
        "cùng độ chi tiết cho cấu trúc tương đương",
        "danh sách locator thuần túy",
        "dẫn chiếu phải xác định conduct và mức/điều khoản xử phạt áp dụng",
        "locator hoặc cụm đối tượng không đủ",
        "provision_path là điều khoản được sửa/thay",
        "Điều sửa đổi chỉ là provenance",
        "không đoán/lặp loại locator",
        "Locator context chỉ khôi phục Khoản/Điểm",
        "không cấp evidence",
    ):
        assert phrase in compact_prompt


def test_population_binding_prompt_distinguishes_warning_and_referenced_penalty() -> None:
    """Step 2 preserves two legal effects that are not additional sanctions."""
    prompt = (PROJECT_ROOT / "baml_src/ontology_population.baml").read_text(encoding="utf-8")
    compact_prompt = " ".join(prompt.split())

    for phrase in (
        "Warning cho cảnh cáo",
        "ReferencedPenalty cho dẫn chiếu mức phạt",
        "AdditionalSanction chỉ khi source nêu hình thức xử phạt bổ sung",
        "target_label_vi là đích dẫn chiếu",
    ):
        assert phrase in compact_prompt


def test_population_binding_prompt_audits_subject_variants_and_cumulative_consequences() -> None:
    """Step 2 must not collapse actor-specific ranges or omit referenced consequences."""
    prompt = (PROJECT_ROOT / "baml_src/ontology_population.baml").read_text(encoding="utf-8")
    compact_prompt = " ".join(prompt.split())

    for phrase in (
        "Audit toàn window cho từng conduct",
        "đối chiếu toàn bộ danh sách locator",
        "Không dừng sau hậu quả đầu",
        "vai trò chỉ một con người thực hiện được là NaturalPerson",
        'dù không có từ "cá nhân"',
        "vai trò chỉ một pháp nhân/tổ chức thực hiện được là Organization",
        "tạo đủ hai variant",
        "không dùng target_label_vi thay subject",
        "RemedialMeasure phải giữ đối tượng",
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
