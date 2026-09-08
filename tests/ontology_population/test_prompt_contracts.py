"""Semantic and size contracts for ontology LLM prompts."""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROMPT_CASES = (
    (
        "baml_src/ontology_rule_assembly.baml",
        (
            "Mục đích bước 3/6",
            "Giữ riêng các động từ/hành động pháp lý độc lập",
            "Phần bổ ngữ bị lược để tránh lặp",
            "không chứng minh cùng rule",
            "các proposal không trùng member",
            "Cây conditions của một rule chi phối mọi consequence",
            "không hợp nhất record vô điều kiện với record có điều kiện",
            "audit từng consequence của assembled_rule",
            "Không suy diễn role, LegalChange, LegalException",
        ),
    ),
    (
        "baml_src/ontology_rule_assembly_resolution.baml",
        (
            "Mục đích bước 3",
            "tạo kết quả nhất quán toàn cục",
            "Giữ riêng các động từ/hành động pháp lý độc lập",
            "Phần bổ ngữ bị lược để tránh lặp",
            "Cây conditions của một rule chi phối mọi consequence",
            "Không hợp nhất record vô điều kiện với record có điều kiện",
            "Audit từng consequence của assembled_rule",
            "các proposal trong một component không trùng member",
            "trả [] nếu chưa đủ căn cứ",
            "So sánh conduct leaf của từng source record",
            "không mở rộng assembled conduct thành hợp của các leaf thành viên",
            "không dùng source rộng hơn để đưa sibling conduct trở lại",
            "so sánh từng conduct_label_vi như một hành vi độc lập",
            "không mở rộng label từ evidence",
            "bất kỳ động từ hoặc phương án nào không có trong conduct label của từng member",
            "hai label bổ sung các hành động khác nhau cho nhau không phải các mảnh của một rule",
        ),
    ),
    (
        "baml_src/ontology_change_resolution.baml",
        (
            "Mục đích bước 4/6",
            "trực tiếp xác định target, loại thay đổi và ngày hiệu lực",
            "audit mọi earlier rule tại đúng locator đó",
            "không có later rule kế thừa chính earlier variant",
            "change_provision_path là locator pháp lý đích",
            "Không đưa Điều chứa lệnh sửa đổi",
            "Không suy ra từ similarity",
            "Thiếu evidence thì không đề xuất",
            "Cổng ghép successor bắt buộc",
            "chỉ cùng chủ đề, phương tiện, hợp đồng, thiết bị hoặc hậu quả là chưa đủ",
            "không nối chúng",
            "Không chuyển leaf giữa hai Điểm chỉ vì nội dung giống nhau",
            "Câu chỉ sửa một Điểm không cho phép lấy earlier record từ Điểm khác",
            "không chọn later sibling gần nghĩa để tránh later null",
        ),
    ),
    (
        "baml_src/ontology_exception_resolution.baml",
        (
            "Mục đích bước 5/6",
            "rule chuyên biệt làm hẹp general rule",
            "citation, provision và target rule riêng",
            "không redirect theo similarity",
            "dẫn chiếu chỉ xác định rule đang nhận một hậu quả",
            "trả citations=[]",
            "không redirect ngoại lệ về general rule",
            "giữ nguyên cả cụm",
            "vốn sẽ thuộc general rule nếu không có câu ngoại lệ",
            "Không biến điều kiện áp dụng thông thường",
            "cách tính số lượng/ngưỡng",
            "quy tắc chọn/gộp hậu quả xử phạt thành ngoại lệ",
            "Không lấy ngoại lệ từ source_contexts, record anh em",
            "không trực tiếp loại hoặc redirect trường hợp thì không đề xuất",
            "Ngoại lệ ở một Điểm/nhánh liền kề không được kế thừa",
            "trường hợp loại trừ không xuất hiện trong evidence đó thì xóa discovery",
        ),
    ),
    (
        "baml_src/ontology_concept_resolution.baml",
        (
            "Mục đích bước 6/6",
            "SameConcept chỉ khi hai occurrence có cùng nghĩa pháp lý",
            "Broader/Narrower chỉ khi một nghĩa pháp lý bao hàm nghĩa kia",
            "proposals=[]",
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
        "Cổng domain áp dụng trước coverage",
        "chỉ lấy conduct xử phạt thuộc giao thông đường bộ",
        "đường sắt, hàng hải, hàng không",
        "Đánh giá từng conduct theo nội dung trực tiếp",
        "Xác định domain sau khi khôi phục toàn bộ phần dùng chung của câu",
        "không được xóa bổ ngữ ngoài domain để biến clause thành đường bộ",
        "phương tiện đường sắt thành đường bộ",
        "Target chỉ có nội dung ngoài domain thì trả conducts rỗng",
        "Target trực tiếp",
        "gắn conduct với cảnh cáo, tiền phạt",
        "sửa/thay điều khoản xác định",
        "danh sách conduct vi phạm giao thông",
        "không cần lặp hậu quả",
        "Loại kỷ luật, hình sự/dân sự, cưỡng chế/thi hành",
        "thủ tục/thẩm quyền",
        "Không tự đảo nghĩa vụ tích cực thành vi phạm",
        "Xét riêng từng nhánh có hiệu lực pháp lý",
        "không ra quyết định xử phạt",
        "không biến hành vi được nhắc lại trong nhánh đó thành conduct",
        "Đây không phải dẫn chiếu mức phạt",
        "Quét Target từ ký tự đầu đến cuối",
        "Audit riêng đoạn mở đầu trước heading mới",
        "tiếp nối Điểm/Khoản/danh sách",
        "Không bỏ đoạn mở đầu vì phía sau có heading mới",
        "Heading mới chỉ chi phối nội dung đứng sau nó",
        "không được áp dụng ngược cho đoạn mở đầu",
        "kế thừa locator đang còn hiệu lực ở cuối Locator context",
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
        "Label chỉ mô tả conduct",
        "không nối hình thức xử phạt, biện pháp khắc phục",
        "`bị phạt`, `bị tước`, `bị buộc`",
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
        "Với `S V1 hoặc V2 C`",
        "tạo `S V1 C` và `S V2 C`",
        "không trả nhánh cụt",
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
        "chỉ là provenance và phải bị loại khỏi path",
        "Lấy locator đích từ câu `sửa đổi/bổ sung ... Điều ...`",
        "dùng Locator context để hoàn thành chính locator đích đó",
        "Không nối path của văn bản sửa đổi với path của điều khoản được sửa",
        "Chỉ áp dụng việc loại ancestor khi source/context thể hiện rõ đây là thao tác sửa đổi",
        "Với quy định trực tiếp, phải giữ đủ mọi ancestor",
        "đã có trong Cấu trúc rồi nối `Khoản`, `Điểm` của Target",
        "không tự rút path xuống chỉ còn `Điều`",
        "provision_path chỉ gồm mã locator",
        "không kèm tên/tiêu đề Chương, Mục, Điều",
        "không đoán/lặp locator",
        "Context không cấp evidence hay conduct",
        "evidence_id được liệt kê trong đúng mục Evidence",
        "Không tự tạo, ghi nhớ hoặc tái sử dụng",
        "chỉ trả ID nguyên văn đã được cấp",
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
        "Cổng hoàn thành",
        "phải quét hết window",
        "Output chỉ có tiền phạt trực tiếp",
        "là output chưa hoàn thành",
        "phải audit độc lập từng câu sau dấu chấm",
        "từng nhóm locator sau dấu chấm phẩy",
        "câu đầu không đại diện cho các câu sau",
        "kế thừa loại hậu quả từ phần dẫn",
        "Audit toàn window cho từng conduct",
        "đối chiếu toàn bộ danh sách locator",
        "Không dừng sau hậu quả đầu",
        "không bỏ hậu quả vì nó nằm xa conduct hoặc có thêm điều kiện",
        "vẫn gắn hậu quả và biểu diễn điều kiện đó trong cây conditions",
        "lập checklist nội bộ cho từng provision_path",
        "lập ma trận từng conduct_id x từng subject_kind x từng clause hậu quả",
        "mặc định clause áp dụng cho mọi conduct và subject variant",
        "chỉ loại một ô khi chính source giới hạn nhánh đó",
        "Không được gắn hậu quả cho một conduct/subject rồi bỏ các conduct đồng cấp",
        "khác biệt câu chữ giữa các conduct lá",
        "không tự tạo ngoại lệ",
        "Clause có nhiều hậu quả cộng dồn phải giữ đủ từng hậu quả",
        "audit từng ô ma trận riêng",
        "tạo chữ ký gồm loại hậu quả, đích, khoảng giá trị và điều kiện",
        "phải có cùng tập chữ ký hậu quả phi tiền tệ",
        "khác số tiền và trừ điểm chỉ cho người trực tiếp điều khiển",
        "Một ô đã đủ không chứng minh ô khác đã đủ",
        "phải bổ sung rule cho chính ô đó",
        "audit lại toàn bộ ma trận trước khi trả output",
        "Một rule chỉ được chứa các hậu quả có cùng tập điều kiện áp dụng",
        "condition_root_id=null",
        "hậu quả có các điều kiện khác nhau phải tách thành các rule",
        "Không được lấy điều kiện của một clause để chi phối",
        "Evidence của mỗi condition và consequence phải chứa chính câu phát biểu",
        "Không dùng Điểm/câu liền kề",
        "làm evidence thay cho clause thực tế",
        "mỗi hậu quả dẫn evidence của clause riêng",
        "Audit provenance cuối cùng cho từng evidence ref",
        "text đó có tự phát biểu assertion đang dẫn hay không",
        "phải bỏ hoặc thay bằng ref của clause thực tế",
        "Một ref đúng khác trong cùng danh sách không hợp thức hóa ref sai",
        "Evidence của condition chỉ gồm clause phát biểu condition",
        "có thể gồm câu dẫn đó và chính clause consequence",
        "không thêm các Điểm/câu sibling nằm giữa chúng",
        "`nếu có`, `nếu đã được cấp`",
        "không cho phép bỏ hậu quả",
        "`trừ trường hợp X` không xóa hậu quả chung",
        "tạo điều kiện Not(X)",
        "Không được bỏ âm bản hoặc dương bản",
        "Tiền phạt có thể khác theo NaturalPerson/Organization",
        "hậu quả bổ sung, khắc phục và điều kiện không tự giới hạn theo subject_kind",
        "phải nhân từng hậu quả đó sang cả hai variant",
        "hoàn thành NaturalPerson không thay thế Organization",
        "Xác định chủ thể từ câu phạt, tiêu đề Điều và structural context",
        "Vai trò chỉ một con người thực hiện được là NaturalPerson",
        'dù không có từ "cá nhân"',
        "vai trò chỉ một pháp nhân/tổ chức thực hiện được là Organization",
        "tiêu đề xác định người điều khiển và loại xe",
        "không để subject_kind null",
        "Xét tính khả thi riêng cho từng conduct lá",
        "không sao chép máy móc mọi loại chủ thể từ câu dẫn chung",
        "trực tiếp điều khiển",
        "Organization không được tạo variant đó",
        "giao phương tiện hoặc để người khác điều khiển",
        "mọi conduct thuộc câu đó phải có đủ hai variant",
        "NaturalPerson chỉ nhận mức cá nhân",
        "Organization chỉ nhận mức tổ chức",
        "không chỉ tách một số conduct trong cùng danh sách",
        "kiểm tra đủ hai variant cho từng conduct_id",
        "consequence_label_vi phải thống nhất với subject_kind và các trường số tiền",
        "không dùng target_label_vi thay subject",
        "RemedialMeasure phải giữ đối tượng",
        "không phân loại chỉ vì có từ",
        "thu hồi hồ sơ, giấy tờ sau vi phạm",
        "văn bản sửa đổi định nghĩa/thay conduct",
        "ReferencedPenalty tới chính Khoản/Điều được sửa",
        "hậu quả bổ sung hoặc khắc phục không thay thế dẫn chiếu mức phạt nền",
        "clause khắc phục/tịch thu dẫn chiếu locator",
        "đối chiếu nội dung từng conduct",
        "giữ clause cùng điều kiện cho mọi conduct có thể thỏa mô tả đó",
        "Ngay trước khi trả output, so sánh các sibling cùng provision_path",
        "chỉ có tiền phạt nhưng sibling có hậu quả phi tiền tệ",
        "output chưa hoàn thành",
        "sao chép đúng chữ ký clause sang variant thiếu",
        "evidence của clause thực tế",
        "không còn sibling thiếu hậu quả",
        "quét ngược từng clause hậu quả trong toàn window",
        "mỗi clause phải được đánh dấu đã có rule",
        "Clause chưa xuất hiện ở bất kỳ sibling nào vẫn không được bỏ",
        "hậu quả kèm `nếu có` phải có rule điều kiện",
        "không chỉ được giữ như phần mô tả",
        "tách hậu quả có điều kiện sang rule riêng",
        "chạy lại cả audit clause và sibling",
    ):
        assert phrase in compact_prompt


@pytest.mark.parametrize(("relative_path", "required_phrases"), PROMPT_CASES)
def test_downstream_ontology_prompt_is_concise_and_preserves_boundaries(
    relative_path: str,
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
    for phrase in required_phrases:
        assert phrase in compact_prompt


def test_rule_assembly_prompt_keeps_empty_candidate_result_wrapper() -> None:
    """Step 3 must complete candidates that have no valid assembly proposal."""
    prompt = (PROJECT_ROOT / "baml_src/ontology_rule_assembly.baml").read_text(encoding="utf-8")
    compact_prompt = " ".join(prompt.split())

    for phrase in (
        "đúng một RuleAssemblyWindowResult cho từng candidate window",
        "sao chép candidate_id nguyên văn",
        "với `proposals=[]`",
        "Không được trả danh sách kết quả tổng thể rỗng",
        "kể cả một input cũng phải có một result",
    ):
        assert phrase in compact_prompt
