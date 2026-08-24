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
    True chỉ khi một câu hỏi grounded hữu ích thực sự cần cả hai fact.
  "#)
  bundle_kind RelationBundleKind? @description(#"
    Hình dạng evidence: MultiHop hoặc Temporal; null khi is_related=false.
  "#)
  bridge string? @description(#"
    Cụm ngắn mô tả nội dung nối hai fact; null khi is_related=false.
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
    {{ _.role("system") }}

    Bối cảnh: chuẩn bị evidence bundle cho benchmark RAG hỏi đáp về
    xử lý vi phạm giao thông đường bộ tại Việt Nam. Retrieval đã
    chỉ đề xuất các pair có khả năng liên quan; bạn đưa ra quyết định
    ngữ nghĩa cuối cùng cho từng pair, độc lập.

    Input: statement là nguồn sự thật pháp lý. document_number, valid_from
    và valid_to chỉ là metadata hỗ trợ. Không dùng kiến thức bên ngoài.

    Đặt is_related=true chỉ khi có một câu hỏi pháp lý tự nhiên, hữu
    ích và grounded mà câu trả lời thực sự cần nội dung của cả hai
    fact. Không xác nhận khi một fact đã đủ; hai fact lặp lại, chỉ
    giống bề mặt/cùng chủ đề, hoặc chỉ có thể ghép thành hai câu hỏi
    độc lập.

    Khi related, bundle_kind chỉ mô tả hình dạng evidence, không phải
    taxonomy quan hệ pháp lý:
    - MultiHop: hai fact bổ sung nhau cho cùng một vấn đề không chủ yếu
      là phân biệt theo thời gian.
    - Temporal: hai fact cùng một vấn đề pháp lý nhưng cần phân biệt
      do thay đổi quy định hoặc phạm vi hiệu lực theo thời gian. Chỉ khác
      văn bản hay mốc thời gian chưa đủ để xác nhận.

    bridge là một cụm ngắn, cụ thể và có căn cứ trong input, nêu
    nội dung thực sự nối hai fact; không chỉ ghi ID, số Điều hay chủ đề
    chung. Khi is_related=false, trả bundle_kind=null và bridge=null.

    Trả đúng một kết quả cho mỗi input và giữ nguyên pair_id.

    {{ _.role("user") }}

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
