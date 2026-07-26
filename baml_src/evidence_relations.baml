enum RelationType {
  ConductToPenalty
  ConductToSubject
  ConductToPointsDeduction
  ConductToRemedialMeasure
  RuleToException
  RuleToAuthority
  AmendedBy
  ProcedureSequence
}

class LegalFactSummary {
  fact_id string
  conduct string
  subject string
  monetary_penalty string?
  license_points_deducted string?
  additional_penalties string[]
  remedial_measures string[]
  valid_from string?
  valid_to string?
  document_number string
  provision_path string
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
  bridge string[] @description(#"
    Thực thể hoặc thuộc tính chung khiến hai fact liên quan, ví dụ số điều
    khoản hoặc tên hành vi dùng chung. Rỗng khi is_related=false.
  "#)
}

function ConfirmEvidenceRelation(fact_a: LegalFactSummary, fact_b: LegalFactSummary) -> ConfirmedRelation {
  client OpenAICompatibleProvider
  prompt #"
    {{ _.role("user") }}

    Xét hai fact pháp lý dưới đây, mỗi fact đã được trích xuất và duyệt độc
    lập. Xác định chúng có quan hệ pháp lý hoặc nghiệp vụ trực tiếp không, và
    nếu có thì thuộc loại nào:

    - ConductToPenalty: cùng hành vi, một fact nêu mức phạt tiền
    - ConductToSubject: cùng hành vi, khác đối tượng hoặc loại phương tiện
    - ConductToPointsDeduction: cùng hành vi, một fact nêu trừ điểm hoặc tước
      giấy phép
    - ConductToRemedialMeasure: cùng hành vi, một fact nêu biện pháp khắc
      phục hậu quả
    - RuleToException: một fact là quy định chung, fact kia là ngoại lệ
    - RuleToAuthority: một fact là quy định về hành vi, fact kia là thẩm
      quyền xử phạt
    - AmendedBy: một fact là quy định cũ, fact kia sửa đổi hoặc thay thế nó
    - ProcedureSequence: hai fact là các bước liên tiếp trong cùng thủ tục

    Không kết luận có quan hệ chỉ vì hai fact nghe giống nhau hoặc dùng từ
    ngữ tương tự. Chỉ kết luận có quan hệ khi nội dung của cả hai fact thực
    sự thể hiện quan hệ đó.

    Fact A:
    {{ fact_a }}

    Fact B:
    {{ fact_b }}

    {{ ctx.output_format }}
  "#
}
