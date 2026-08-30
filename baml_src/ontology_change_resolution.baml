enum OntologyLegalChangeType {
  Amendment
  Replacement
  RepealOrSupersession
}

class CrossDocumentChangeWindowInput {
  candidate_id string
  earlier_document_number string
  later_document_number string
  earlier_records OntologyPopulationSummary[]
  later_records OntologyPopulationSummary[]
  change_evidence_sources OntologyEvidenceSource[]
}

class LegalChangeProposalEntry {
  earlier_population_id string @description(#"Một population ID trong earlier_records."#)
  later_population_id string? @description(#"
    Một ID trong later_records khi có later rule version; chỉ null khi source chứng
    minh bãi bỏ/thay thế mà window không có later rule.
  "#)
  change_type OntologyLegalChangeType
  effective_on string @description(#"Ngày hiệu lực YYYY-MM-DD được source hỗ trợ."#)
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
    - Không suy ra từ similarity, chênh lệch mức phạt, thứ tự văn bản, metadata
      hoặc valid_to null.
    - earlier ID thuộc earlier_records; later ID nếu có thuộc later_records; mỗi ID
      chỉ dùng một lần trong result.
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
    earlier_records={{ window.earlier_records }}
    later_records={{ window.later_records }}
    change_evidence_sources={{ window.change_evidence_sources }}
    === end candidate_id={{ window.candidate_id }} ===
    {% endfor %}

    {{ ctx.output_format }}
  "#
}
