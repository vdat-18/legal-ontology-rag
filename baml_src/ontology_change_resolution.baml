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
  earlier_population_id string @description(#"Exactly one population ID from earlier_records."#)
  later_population_id string? @description(#"
    One population ID from later_records when the change produces a later rule version. Null only when authoritative
    evidence supports repeal or supersession without a later rule in this window.
  "#)
  change_type OntologyLegalChangeType
  effective_on string @description(#"Authoritatively supported effective date in YYYY-MM-DD format."#)
  change_provision_path string @description(#"Fullest structural path supported for the authoritative change provision."#)
  evidence OntologyEvidenceReference[] @description(#"
    References to supplied source evidence that explicitly supports the change, its target, type, and effective date.
  "#)
  rationale_vi string @description(#"Giải thích ngắn gọn vì sao source trực tiếp chứng minh quan hệ thay đổi này."#)
}

class CrossDocumentChangeWindowResult {
  candidate_id string
  proposals LegalChangeProposalEntry[]
}

function ProposeCrossDocumentLegalChanges(windows: CrossDocumentChangeWindowInput[]) -> CrossDocumentChangeWindowResult[] {
  client OpenAICompatibleProvider
  prompt #"
    {{ _.role("user") }}

    Bạn là chuyên gia ontology pháp luật. Xác định LegalChange giữa các rule
    version của hai văn bản trong từng candidate window.

    Nguyên tắc bắt buộc:
    - Chỉ đề xuất khi nguồn trực tiếp nêu thay đổi, target và ngày hiệu lực; không
      suy ra từ độ giống nhau, thứ tự văn bản, metadata hay valid_to còn trống.
    - earlier ID phải thuộc earlier_records; later ID nếu có phải thuộc
      later_records. Mỗi ID chỉ dùng một lần trong một result.
    - Amendment/Replacement cần later rule; RepealOrSupersession chỉ được để
      later null khi nguồn trực tiếp chứng minh không có phiên bản thay thế.
    - Chỉ dùng evidence_id đã cung cấp. effective_on và provision path phải do
      nguồn hỗ trợ; không tạo identity, exception hay concept relation khác.
    - Nếu nguồn chưa đủ, trả proposals rỗng. Trả đúng một result cho mỗi candidate_id.

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
