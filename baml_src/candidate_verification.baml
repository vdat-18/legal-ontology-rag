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
  query_type_catalog: string[]
) -> VerificationResult {
  client OpenAICompatibleProvider
  prompt #"
    {{ _.role("user") }}

    Bạn là người kiểm chứng độc lập, không phải người đã sinh ra câu hỏi này.
    Đọc câu hỏi, đáp án, lập luận và các fact căn cứ dưới đây, sau đó đánh giá
    độc lập theo từng tiêu chí.

    1. is_standalone: câu hỏi có tự đủ nghĩa, không cần ngữ cảnh bên ngoài.
    2. is_specific: câu hỏi có đủ cụ thể để có một đáp án xác định, không quá
       chung chung.
    3. is_question_grounded: evidence của các fact dưới đây có đủ để trả lời
       câu hỏi không.
    4. is_answer_grounded: mọi nội dung trong đáp án có được evidence hỗ trợ
       không, và đáp án có bỏ sót chế tài nào mà fact đã nêu không.
    5. is_domain_relevant: câu hỏi có thuộc phạm vi xử phạt vi phạm giao
       thông đường bộ tại Việt Nam không.
    6. is_temporal_correct: nếu có as_of_date, nó có nằm trong khoảng hiệu
       lực (valid_from, valid_to) của mọi fact liên quan không.
    7. necessary_reasoning_hops: thử bỏ lần lượt từng fact khỏi danh sách
       dưới đây — nếu đáp án vẫn đủ căn cứ khi thiếu một fact, fact đó không
       thực sự cần thiết. Trả về số fact thực sự cần thiết, có thể khác
       declared_reasoning_hops={{ declared_reasoning_hops }}.
    8. observed_query_type: chọn đúng một giá trị trong danh sách sau, dựa
       theo nội dung câu hỏi thực tế, không dựa theo nhãn đã gán trước:
       {{ query_type_catalog }}
    9. observed_answerability: đánh giá độc lập câu hỏi này answerable,
       partially answerable, underspecified, hay unanswerable dựa trên
       evidence thực có, không dựa theo nhãn đã gán trước.

    {% if as_of_date %}Thời điểm áp dụng: {{ as_of_date }}{% endif %}

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
