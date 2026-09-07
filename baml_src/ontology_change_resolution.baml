enum OntologyLegalChangeType {
  Amendment
  Replacement
  RepealOrSupersession
}

class CrossDocumentChangeWindowInput {
  candidate_id string
  earlier_document_number string
  later_document_number string
  later_document_valid_from string?
  earlier_records LegalChangeRuleSummary[]
  later_records LegalChangeRuleSummary[]
  change_evidence_sources OntologyEvidenceSource[]
}

class LegalChangeConditionSummary {
  condition_id string
  kind OntologyConditionKind
  text_vi string
  operand_ids string[]
}

class LegalChangeConsequenceSummary {
  consequence_id string
  kind OntologyConsequenceKind
  minimum_amount string?
  maximum_amount string?
  currency OntologyCurrency?
  point_value int?
  consequence_label_vi string?
  minimum_duration_months int?
  maximum_duration_months int?
  target_label_vi string?
  responsible_role_label_vi string?
}

class LegalChangeRuleSummary {
  population_id string
  conduct_label_vi string
  subject_kind OntologySubjectKind?
  role_labels_vi string[]
  vehicle_category_labels_vi string[]
  condition_root_id string?
  conditions LegalChangeConditionSummary[]
  consequences LegalChangeConsequenceSummary[]
  valid_from string?
  valid_to string?
  document_number string
  provision_path string
}

class LegalChangeProposalEntry {
  earlier_population_id string @description(#"Một population ID trong earlier_records."#)
  later_population_id string? @description(#"
    Một ID trong later_records khi có later rule version; chỉ null khi source chứng
    minh bãi bỏ/thay thế mà window không có later rule.
  "#)
  change_type OntologyLegalChangeType
  effective_on string @description(#"Ngày hiệu lực YYYY-MM-DD từ source riêng hoặc mặc định văn bản."#)
  change_provision_path string @description(#"Đường dẫn đầy đủ nhất của quy định thay đổi."#)
  evidence OntologyEvidenceReference[] @description(#"Evidence trực tiếp cho target, loại thay đổi và ngày hiệu lực."#)
  rationale_vi string @description(#"Lý do ngắn gọn dựa trên source."#)
}

class CrossDocumentChangeWindowResult {
  candidate_id string
  proposals LegalChangeProposalEntry[]
}

function ProposeCrossDocumentLegalChanges(windows: CrossDocumentChangeWindowInput[]) -> CrossDocumentChangeWindowResult[] {
  client OpenAICompatibleProvider
  prompt #"
    {{ _.role("system") }}

    Bối cảnh: xây dựng ontology có bằng chứng chỉ cho quy định xử phạt vi phạm
    hành chính về giao thông đường bộ Việt Nam; không dùng cho vụ việc hoặc domain
    khác.

    Mục đích bước 4/6: nối các rule versions bằng LegalChange để ontology phản ánh
    diễn biến theo thời gian mà không đồng nhất rule cũ và rule mới.

    Nhiệm vụ: xác định LegalChange giữa rule versions của hai văn bản trong từng
    candidate window.

    - Chỉ đề xuất khi quy định sửa đổi, thay thế hoặc bãi bỏ trực tiếp xác định
      target, loại thay đổi và ngày hiệu lực của rule xử phạt. Văn bản sau có hành
      vi hay mức phạt tương tự không tự tạo LegalChange.
    - Đọc toàn bộ earlier_records và later_records trong window. Chọn đúng rule
      được change_evidence_sources chỉ tới; record xuất hiện trong window chỉ là
      ứng viên, không phải bằng chứng rằng có thay đổi.
    - Đối chiếu locator trước nội dung; bỏ qua Chương/Mục và container của văn bản
      sửa đổi. Với bãi bỏ, earlier record phải khớp chính xác Điều/Khoản/Điểm bị
      bãi bỏ. Với sửa đổi/thay thế, mặc định earlier và later records phải cùng
      Điều/Khoản/Điểm với target.
    - Chỉ liên kết hai Điểm khác nhau trong cùng Điều và Khoản khi source trực tiếp
      viết lại cả hai Điểm, leaf bị loại khỏi Điểm cũ và được giữ hoặc hấp thụ đầy
      đủ tại Điểm mới. Không bao giờ liên kết records khác Điều hoặc Khoản; quy định
      cùng hành vi tại Điều/Khoản khác là rule đồng thời, không phải phiên bản thay
      thế. Không ghép chỉ vì conduct tương tự.
    - Trong cùng locator, đối chiếu đúng leaf theo chủ thể, hành động, đối tượng và
      điều kiện. Không ghép một earlier leaf với later sibling khác hành động. Một
      rule có thể tách thành nhiều leaf hoặc nhiều leaf có thể nhập lại chỉ khi
      chính source và semantics của từng leaf hỗ trợ mọi liên kết đó.
    - later_records chỉ gồm rule có provenance từ chính nhóm source chunks đang
      đọc. Danh sách rỗng là hợp lệ đối với bãi bỏ hoặc source không tạo rule mới.
      Không yêu cầu ghép một later rule khi source chỉ chứng minh bãi bỏ.
    - Không suy ra từ similarity, chênh lệch mức phạt, thứ tự văn bản, metadata
      hoặc valid_to null.
    - effective_on dùng ngày riêng được change_evidence_sources trực tiếp nêu cho
      target; nếu không có ngày riêng thì dùng later_document_valid_from. Không
      yêu cầu ngày mặc định phải được lặp lại trong source batch hiện tại.
    - earlier ID thuộc earlier_records; later ID nếu có thuộc later_records. Không
      lặp lại đúng cùng một cặp earlier/later và change type.
    - Amendment/Replacement cần later rule. RepealOrSupersession chỉ để later null
      khi source trực tiếp chứng minh không có later rule trong window.
    - Chỉ dùng evidence_id, effective_on và provision path được source hỗ trợ;
      không tạo identity, exception hoặc concept relation.
    - Thiếu evidence thì không đề xuất. Trả đúng một result cho mỗi candidate_id.

    {{ _.role("user") }}

    Các candidate window cần đánh giá:
    {% for window in windows %}
    === candidate_id={{ window.candidate_id }} ===
    earlier_document={{ window.earlier_document_number }}
    later_document={{ window.later_document_number }}
    later_document_valid_from={{ window.later_document_valid_from }}
    earlier_records={{ window.earlier_records }}
    later_records={{ window.later_records }}
    change_evidence_sources={{ window.change_evidence_sources }}
    === end candidate_id={{ window.candidate_id }} ===
    {% endfor %}

    {{ ctx.output_format }}
  "#
}
