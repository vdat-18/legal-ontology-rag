enum RelationBundleKind {
  MultiHop
  Temporal
}

class LegalFactSummary {
  fact_id string
  statement string
  valid_from string?
  valid_to string?
  document_number string
}

class ConfirmedRelation {
  is_related bool @description(#"
    True nếu hai fact thực sự có quan hệ pháp lý hoặc nghiệp vụ trực tiếp.
    False nếu chỉ giống nhau bề mặt, cùng chủ đề chung chung, hoặc không có
    căn cứ rõ ràng để nối chúng lại.
  "#)
  bundle_kind RelationBundleKind? @description(#"
    MultiHop nếu hai fact bổ sung nhau cho cùng một câu hỏi. Temporal nếu hai
    fact thể hiện thay đổi quy định hoặc hai chế độ hiệu lực cần phân biệt
    theo thời gian. Null khi is_related=false.
  "#)
  bridge string? @description(#"
    Một cụm ngắn mô tả chính xác nội dung thực sự nối hai fact. Null khi
    is_related=false.
  "#)
}

class EvidenceRelationBatchItem {
  pair_id string
  fact_a LegalFactSummary
  fact_b LegalFactSummary
}

class EvidenceRelationBatchResult {
  pair_id string @description(#"Phải trùng với một pair_id trong danh sách đầu vào."#)
  relation ConfirmedRelation
}

function ConfirmEvidenceRelationBatch(items: EvidenceRelationBatchItem[]) -> EvidenceRelationBatchResult[] {
  client OpenAICompatibleProvider
  prompt #"
    {{ _.role("user") }}

    Nhiệm vụ: xác nhận quan hệ trực tiếp giữa các legal fact hoàn chỉnh trong
    domain xử lý vi phạm giao thông đường bộ. Xử lý độc lập từng pair và giữ
    nguyên pair_id.

    Chỉ đặt is_related=true khi kết hợp hai statement tạo ra một liên hệ pháp
    lý hữu ích và thực sự cần cả hai fact để tạo một câu hỏi đúng grounded.

    Khi related, chọn bundle_kind:
    - MultiHop: hai fact bổ sung trực tiếp cho cùng tình huống, như quy tắc với
      điều kiện, ngoại lệ, thẩm quyền, bước thủ tục, chế tài hoặc hệ quả khác.
    - Temporal: hai fact thể hiện quy định thay đổi, thay thế hoặc các chế độ
      hiệu lực khác nhau cần phân biệt theo thời gian.

    Không xác nhận nếu hai fact chỉ trùng từ, cùng loại phương tiện, cùng văn
    bản/chủ đề, diễn đạt lặp lại, hoặc một fact đã đủ cho câu hỏi. Metadata
    thời gian chỉ hỗ trợ, không tự chứng minh quan hệ temporal.

    Khi related: bridge là một cụm ngắn nêu đúng hành vi, tình huống, thủ tục
    hoặc thay đổi nối hai fact, không chỉ ghi số Điều. Khi không related:
    bundle_kind=null và bridge=null.

    Các cặp cần xác nhận:

    {% for item in items %}
    === pair_id={{ item.pair_id }} ===
    Fact A [{{ item.fact_a.fact_id }}]
    Statement: {{ item.fact_a.statement }}
    Văn bản: {{ item.fact_a.document_number }}
    Hiệu lực: {{ item.fact_a.valid_from }} đến {{ item.fact_a.valid_to }}

    Fact B [{{ item.fact_b.fact_id }}]
    Statement: {{ item.fact_b.statement }}
    Văn bản: {{ item.fact_b.document_number }}
    Hiệu lực: {{ item.fact_b.valid_from }} đến {{ item.fact_b.valid_to }}
    === end pair_id={{ item.pair_id }} ===
    {% endfor %}

    {{ ctx.output_format }}
  "#
}
