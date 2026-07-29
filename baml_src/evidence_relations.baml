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

    Bạn là chuyên gia phân tích quan hệ giữa các quy định xử phạt vi phạm
    giao thông đường bộ tại Việt Nam.

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

    Ví dụ:

    Fact A: conduct="Không chấp hành hiệu lệnh của đèn tín hiệu giao thông",
    subject="Người điều khiển ô tô", monetary_penalty="4.000.000 đồng đến
    6.000.000 đồng", provision_path="Điều 6 khoản 9 điểm b"
    Fact B: conduct="Không chấp hành hiệu lệnh của đèn tín hiệu giao thông",
    subject="Người điều khiển ô tô", license_points_deducted="6 điểm",
    provision_path="Điều 6 khoản 10"
    Kết quả: is_related=true, relation_type=ConductToPointsDeduction,
    bridge=["Điều 6", "không chấp hành hiệu lệnh của đèn tín hiệu giao
    thông"]. (Cùng một hành vi, một fact nêu phạt tiền, fact kia nêu trừ
    điểm — quan hệ trực tiếp, có căn cứ rõ ràng.)

    Fact A: conduct="Không chấp hành hiệu lệnh của đèn tín hiệu giao thông",
    subject="Người điều khiển ô tô", provision_path="Điều 6 khoản 9 điểm b"
    Fact B: conduct="Chở quá số người quy định", subject="Người điều khiển ô
    tô", provision_path="Điều 6 khoản 5"
    Kết quả: is_related=false, relation_type=null, bridge=[]. (Hai hành vi
    khác nhau, chỉ cùng nằm trong Điều 6 và cùng đối tượng ô tô — không có
    quan hệ pháp lý trực tiếp giữa hai hành vi này.)

    Bây giờ xét hai fact sau:

    Fact A:
    {{ fact_a }}

    Fact B:
    {{ fact_b }}

    {{ ctx.output_format }}
  "#
}
