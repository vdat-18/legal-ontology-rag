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
    Thực thể hoặc thuộc tính chung khiến hai fact liên quan, ví dụ số điều
    khoản hoặc tên hành vi dùng chung. Rỗng khi is_related=false.
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

    Xác nhận quan hệ pháp lý trực tiếp cho từng cặp fact về xử phạt giao thông
    đường bộ. Xử lý từng pair độc lập và giữ nguyên pair_id.

    Mỗi statement là một quy tắc pháp lý hoàn chỉnh. Chỉ đặt is_related=true
    khi hai statement và metadata chứng minh một quan hệ trực tiếp dưới đây;
    giống từ, cùng Điều hoặc cùng chủ đề chung chung là chưa đủ:
    - ComplementaryRule: quy tắc nền -> quy tắc bổ sung một khía cạnh, điều
      kiện, hệ quả hoặc cách áp dụng cần thiết cho cùng tình huống pháp lý.
    - RuleToException: quy định chung -> ngoại lệ.
    - RuleToAuthority: quy định hành vi -> thẩm quyền xử phạt.
    - AmendedBy: quy định cũ -> quy định sửa đổi/thay thế.
    - ProcedureSequence: bước trước -> bước sau trong cùng thủ tục.

    Khi related: chọn đúng một relation_type, đặt from_fact_id/to_fact_id theo
    hướng trên và bridge chỉ gồm thực thể/thuộc tính thật sự nối hai fact.
    Khi không related: relation_type, from_fact_id, to_fact_id=null và bridge=[].

    Ví dụ: hai fact cùng hành vi, một fact nêu phạt tiền và fact kia nêu trừ
    điểm -> ComplementaryRule. Hai hành vi khác nhau chỉ cùng Điều và cùng
    loại xe -> không related.

    Các cặp cần xác nhận:

    {% for item in items %}
    === pair_id={{ item.pair_id }} ===
    Fact A:
    {{ item.fact_a }}

    Fact B:
    {{ item.fact_b }}
    === end pair_id={{ item.pair_id }} ===
    {% endfor %}

    {{ ctx.output_format }}
  "#
}
