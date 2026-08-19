enum Answerability {
  Answerable
  PartiallyAnswerable
  Underspecified
  Unanswerable
}

class VerificationResult {
  is_standalone bool @description(#"Câu hỏi có tự đủ nghĩa, không cần ngữ cảnh bên ngoài."#)
  is_specific bool @description(#"Câu hỏi có đủ cụ thể để có một đáp án xác định, không quá chung chung."#)
  is_question_grounded bool @description(#"Evidence của các fact có đủ để trả lời câu hỏi không."#)
  is_answer_grounded bool @description(#"Mọi nội dung trong đáp án có được evidence hỗ trợ, không bỏ sót chế tài."#)
  is_domain_relevant bool @description(#"Câu hỏi có thuộc phạm vi xử phạt vi phạm giao thông đường bộ tại Việt Nam."#)
  is_temporal_correct bool @description(#"as_of_date, nếu có, nằm trong khoảng hiệu lực của mọi fact liên quan."#)
  uses_required_bridge bool @description(#"Câu hỏi và đáp án dùng bridge được cung cấp; true nếu bridge rỗng."#)
  necessary_reasoning_hops int @description(#"
    Số fact thực sự cần thiết để trả lời, sau khi thử bỏ từng fact xem đáp án
    còn đủ căn cứ không. Có thể khác số fact đã khai trong scenario.
  "#)
  observed_query_type string @description(#"Một giá trị trong danh sách được cung cấp, chọn theo nội dung câu hỏi thực tế, không theo nhãn đã gán trước."#)
  observed_answerability Answerability @description(#"Nhãn quan sát thực tế, có thể khác nhãn đã khai trong scenario."#)
  reasoning string @description(#"Giải thích ngắn gọn cho các phán đoán trên."#)
}

function VerifyCandidate(
  query_vi: string,
  reference_answer_vi: string,
  reasoning_vi: string,
  facts: FactContext[],
  claims: ClaimReference[],
  as_of_date: string?,
  declared_reasoning_hops: int,
  bridge: string[],
  query_type_catalog: string[]
) -> VerificationResult {
  client OpenAICompatibleProvider
  prompt #"
    {{ _.role("user") }}

    Kiểm chứng độc lập câu hỏi, đáp án và lập luận bằng các facts bên dưới;
    không tin nhãn khai báo trước.

    Tiêu chí:
    1. standalone: câu hỏi tự đủ nghĩa.
    2. specific: đủ chi tiết để có đáp án xác định.
    3. question_grounded: facts đủ căn cứ trả lời.
    4. answer_grounded: mọi ý trong đáp án có căn cứ và không bỏ sót chế tài
       liên quan đã có trong facts.
    5. domain_relevant: thuộc xử phạt giao thông đường bộ Việt Nam.
    6. temporal_correct: as_of_date nằm trong hiệu lực của mọi fact cần dùng;
       không có as_of_date thì true.
    7. necessary_reasoning_hops: số fact tối thiểu vẫn đủ trả lời. Thử bỏ từng
       fact; fact bỏ được là không cần thiết. Không mặc định bằng
       declared_reasoning_hops={{ declared_reasoning_hops }}.
    8. uses_required_bridge: query và answer thật sự dùng bridge; bridge rỗng
       thì true.
    9. observed_query_type: chọn đúng một giá trị từ {{ query_type_catalog }}
       theo query thực tế.
    10. observed_answerability: phân loại theo evidence thực tế, không theo
        nhãn đã khai.

    reasoning phải ngắn gọn nhưng nêu rõ tiêu chí thất bại và fact nào thực
    sự cần thiết.

    {% if as_of_date %}Thời điểm áp dụng: {{ as_of_date }}{% endif %}
    Bridge bắt buộc: {{ bridge }}

    Câu hỏi:
    {{ query_vi }}

    Đáp án:
    {{ reference_answer_vi }}

    Lập luận:
    {{ reasoning_vi }}

    Claims:
    {% for claim in claims %}
    - [{{ claim.fact_id }}] {{ claim.claim }}
    {% endfor %}

    Fact căn cứ:
    {% for fact in facts %}
    [{{ fact.fact_id }}] {{ fact.conduct }} — {{ fact.subject }}
    Điều khoản: {{ fact.provision_path }} ({{ fact.document_number }})
    Phạt tiền: {{ fact.monetary_penalty }}
    Trừ điểm: {{ fact.license_points_deducted }}
    Hình phạt bổ sung: {{ fact.additional_penalties }}
    Biện pháp khắc phục: {{ fact.remedial_measures }}
    Hiệu lực: {{ fact.valid_from }} đến {{ fact.valid_to }}
    ---
    {% endfor %}

    {{ ctx.output_format }}
  "#
}
