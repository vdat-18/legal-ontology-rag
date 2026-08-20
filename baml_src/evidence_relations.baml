enum RelationType {
  ComplementaryRule
  RuleToException
  RuleToAuthority
  AmendedBy
  ProcedureSequence
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
  relation_type RelationType? @description(#"
    Bắt buộc khi is_related=true, null khi is_related=false.
  "#)
  from_fact_id string? @description(#"
    Bắt buộc khi is_related=true và phải là fact_id của một trong hai fact
    đầu vào. Đây là đầu nguồn theo hướng ngữ nghĩa của relation.
  "#)
  to_fact_id string? @description(#"
    Bắt buộc khi is_related=true và phải là fact_id còn lại. Đây là đầu đích
    theo hướng ngữ nghĩa của relation.
  "#)
  bridge string[] @description(#"
    Cụm ngắn mô tả tình huống, quy tắc hoặc đối tượng thực sự nối hai fact.
    Rỗng khi is_related=false.
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
    lý hữu ích để hỏi đáp và thuộc đúng một loại:
    - ComplementaryRule: quy tắc nền -> quy tắc bổ sung cho cùng tình huống,
      như điều kiện áp dụng, chế tài khác, biện pháp hoặc cách thực hiện.
    - RuleToException: quy tắc chung -> trường hợp loại trừ hoặc ngoại lệ.
    - RuleToAuthority: quy tắc được xử lý -> chủ thể có thẩm quyền xử lý.
    - AmendedBy: quy định cũ -> quy định sửa đổi, thay thế hoặc làm thay đổi
      cách áp dụng quy định cũ.
    - ProcedureSequence: bước trước -> bước sau trong cùng quy trình xử lý.

    Không xác nhận nếu hai fact chỉ trùng từ, cùng loại phương tiện, cùng văn
    bản/chủ đề, diễn đạt lặp lại, hoặc không đủ căn cứ xác định hướng. Metadata
    thời gian chỉ hỗ trợ, không tự chứng minh quan hệ sửa đổi.

    Khi related: chọn một relation_type; đặt from_fact_id/to_fact_id theo hướng
    mô tả trên; bridge gồm 1-3 cụm ngắn nêu đúng hành vi, tình huống, thủ tục
    hoặc quy định chung nối hai fact, không chỉ ghi số Điều. Khi không related:
    relation_type, from_fact_id, to_fact_id=null và bridge=[].

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
