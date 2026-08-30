class ExceptionCitationProposalEntry {
  citation OntologyEvidenceReference @description(#"Source reference for the citation wording, not a normalized locator."#)
  target_population_ids string[] @description(#"
    One or more version-specific rule IDs from target_records whose grounded provisions resolve this grouped citation.
  "#)
}

class LegalExceptionProposalEntry {
  general_population_id string @description(#"Exactly one rule ID from general_records."#)
  exception_provision_path string @description(#"Fullest structural path of the provision stating the exception."#)
  evidence OntologyEvidenceReference[] @description(#"Source references that directly state the complete exception."#)
  citations ExceptionCitationProposalEntry[] @description(#"
    Grouped provision citations resolved to target rule versions. Return no proposal when citation targets cannot be
    supported from the supplied target records.
  "#)
  rationale_vi string @description(#"Giải thích ngắn gọn vì sao source chứng minh exception và từng target."#)
}

class ExceptionResolutionWindowInput {
  candidate_id string
  source_document_number string
  general_records OntologyPopulationSummary[]
  target_records OntologyPopulationSummary[]
  source_evidence_sources OntologyEvidenceSource[]
}

class ExceptionResolutionWindowResult {
  candidate_id string
  proposals LegalExceptionProposalEntry[]
}

function ProposeLegalExceptionResolutions(windows: ExceptionResolutionWindowInput[]) -> ExceptionResolutionWindowResult[] {
  client OpenAICompatibleProvider
  prompt #"
    {{ _.role("user") }}

    Bạn là chuyên gia ontology pháp luật. Xác định LegalException và
    ProvisionCitation cho các rule version trong từng candidate window.

    Nguyên tắc bắt buộc:
    - Chỉ đề xuất khi toàn ngữ cảnh trực tiếp chứng minh ngoại lệ làm hẹp một
      general rule; không quyết định bằng keyword hoặc độ giống câu chữ.
    - General ID phải thuộc general_records. Citation chỉ dùng source evidence
      đã cung cấp; target IDs phải thuộc target_records và phù hợp locator.
    - Giữ citation, provision và target rule là các đối tượng khác nhau. Không
      redirect theo similarity, thứ tự văn bản hay lịch sử thay đổi.
    - Provision path phải được nguồn hỗ trợ. Không đánh giá priority hoặc một
      vụ việc cụ thể. Nếu chưa đủ căn cứ, trả proposals rỗng.
    - Trả đúng một result cho mỗi candidate_id và không lặp cùng exception/general rule.

    {% for window in windows %}
    === candidate_id={{ window.candidate_id }} ===
    source_document={{ window.source_document_number }}
    general_records={{ window.general_records }}
    target_records={{ window.target_records }}
    source_evidence_sources={{ window.source_evidence_sources }}
    === end candidate_id={{ window.candidate_id }} ===
    {% endfor %}

    {{ ctx.output_format }}
  "#
}
