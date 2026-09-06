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
        "Thứ tự ưu tiên bắt buộc",
        "bao phủ đủ Target từ đầu đến cuối",
        "Không được bỏ một nhánh nguồn để làm label ngắn hoặc dễ tách",
        "Target trực tiếp",
        "gắn conduct với cảnh cáo, tiền phạt",
        "sửa/thay điều khoản xác định",
        "danh sách conduct vi phạm giao thông",
        "không cần lặp hậu quả",
        "Loại kỷ luật, hình sự/dân sự, cưỡng chế/thi hành",
        "thủ tục/thẩm quyền",
        "Không tự đảo nghĩa vụ tích cực thành vi phạm",
        "Quét Target từ ký tự đầu đến cuối",
        "Audit riêng đoạn mở đầu trước heading mới",
        "tiếp nối Điểm/Khoản/danh sách",
        "Không bỏ đoạn mở đầu vì phía sau có heading mới",
        "conduct/evidence vẫn chỉ từ Target",
        "Audit từng Điểm/nhánh",
        "đánh dấu mọi hành động, không hành động, trạng thái vi phạm",
        "mỗi đoạn mở đầu, Điểm và nhánh trực tiếp bị xử phạt",
        "Kiểm tra lại coverage sau khi phân rã",
        "không bỏ sót khi Target chuyển Khoản/Điều",
        "xác định chủ thể chịu xử phạt từ câu dẫn mức phạt/cấu trúc chi phối",
        "Mọi label phải giữ nguyên chủ thể đó và đủ loại phương tiện/phạm vi",
        "không lấy công cụ, phần mềm, giấy tờ, phương tiện",
        "Mỗi phương án tự đủ bị phạt là một conduct lá",
        "Khôi phục phần dùng chung",
        "mọi vị ngữ độc lập",
        "động từ dùng như bổ nghĩa bị động",
        "S V1, V2 O",
        "O V1, V2 trái quy định",
        "các bổ ngữ lựa chọn độc lập cùng một vị ngữ",
        "S V O1 hoặc O2",
        "thay lần lượt O1/O2 đều tạo câu vi phạm tự đủ",
        "P không đúng A, không đúng B",
        "danh sách dấu phẩy sau một lần xuất hiện của vị ngữ",
        "không lặp vị ngữ và không dùng liên từ lựa chọn",
        "V tại A, B",
        "làm thay đổi A, B",
        "P bằng A/B/C",
        "không tách chỉ vì từng phần có thể ghép thành câu ngắn",
        "Hai đối tượng pháp lý độc lập nối bằng `hoặc`",
        "đọc lại từng label theo chủ thể-vị ngữ-bổ ngữ",
        "Chỉ tách khi có nhiều vị ngữ hoặc các nhánh lựa chọn độc lập trong source",
        "không dùng khả năng viết lại thành câu ngắn làm căn cứ duy nhất",
        "O V1, V2 trái quy định",
        "quy tắc này ưu tiên hơn quy tắc giữ danh sách dấu phẩy",
        "phương tiện chế tạo, cải tạo trái quy định",
        "hai conduct cho `chế tạo` và `cải tạo`",
        "Không được trả label vẫn chứa hai động từ cùng đứng trước `trái quy định`",
        "S cho phép/chở/để T V1, V2 hoặc V3",
        "giữ S là chủ thể chịu xử phạt",
        "tách V1/V2/V3",
        "pháp lý buộc chúng cùng cấu thành một yêu cầu",
        "động từ lồng sau mục đích/khả năng",
        "S P O để/thực hiện việc V1, V2, V3",
        "lặp toàn bộ khung `S P O để V1`",
        "Động từ lồng vẫn là vị ngữ cần audit",
        "Chủ thể label là chủ thể chịu xử phạt trong mệnh đề chi phối",
        "không phải người thực hiện hành động phụ",
        "S tự V hoặc chở/cho phép/để T V",
        "không đổi nhánh thứ hai thành `T V`",
        "mọi label đều bắt đầu từ S",
        "Nhánh rút gọn phải tự đủ nghĩa",
        "khôi phục cả chủ thể và vị ngữ chi phối bị lược",
        "S V O1; O2",
        "phải lặp lại V trong label",
        "không đổi `S V O` thành `S O` hay hành động khác của S",
        "Bảo toàn toàn bộ điều kiện xác định đối tượng",
        "phần trong ngoặc hoặc mệnh đề cuối áp dụng cho nhiều nhánh",
        "ưu tiên đủ nghĩa hơn label ngắn",
        "cụm danh từ pháp lý được lặp lại hoặc rút gọn",
        "mỗi lá giữ đủ giấy tờ, loại xe, bên giữ/cấp và phạm vi",
        "Không thêm động từ/hành động để nối chủ thể với hành vi",
        "source trực tiếp hỗ trợ",
        "Không rút ngắn phần trong ngoặc",
        "thay đổi đối tượng hoặc điều kiện áp dụng của leaf",
        "Phân loại chủ thể để bước 2 bind",
        "locator thuần túy",
        "Dẫn chiếu phải trực tiếp xác định conduct",
        "chỉ thi hành, khắc phục hoặc tính hậu quả/số lợi",
        "không tạo conduct mới",
        "provision_path là điều khoản được sửa/thay",
        "Điều sửa đổi chỉ là provenance",
        "provision_path chỉ gồm mã locator",
        "không kèm tên/tiêu đề Chương, Mục, Điều",
        "không đoán/lặp locator",
        "Context không cấp evidence hay conduct",
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
        "Xác định chủ thể từ câu phạt, tiêu đề Điều và structural context",
        "Vai trò chỉ một con người thực hiện được là NaturalPerson",
        'dù không có từ "cá nhân"',
        "vai trò chỉ một pháp nhân/tổ chức thực hiện được là Organization",
        "tiêu đề xác định người điều khiển và loại xe",
        "không để subject_kind null",
        "mọi conduct thuộc câu đó phải có đủ hai variant",
        "NaturalPerson chỉ nhận mức cá nhân",
        "Organization chỉ nhận mức tổ chức",
        "không chỉ tách một số conduct trong cùng danh sách",
        "kiểm tra đủ hai variant cho từng conduct_id",
        "consequence_label_vi phải thống nhất với subject_kind và các trường số tiền",
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
