class ExceptionCitationProposalEntry {
  citation OntologyEvidenceReference @description(#"Evidence chứa citation wording, không phải locator chuẩn hóa."#)
  target_population_ids string[] @description(#"
    Một hoặc nhiều rule-version ID trong target_records mà citation trỏ tới.
  "#)
}

class LegalExceptionProposalEntry {
  general_population_id string @description(#"Một rule ID trong general_records."#)
  exception_provision_path string @description(#"Đường dẫn đầy đủ nhất của quy định ngoại lệ."#)
  evidence OntologyEvidenceReference[] @description(#"Evidence trực tiếp nêu đầy đủ ngoại lệ."#)
  citations ExceptionCitationProposalEntry[] @description(#"Citation đã resolve tới target rule versions."#)
  rationale_vi string @description(#"Lý do ngắn gọn cho ngoại lệ và từng target."#)
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
    {{ _.role("system") }}

    Bối cảnh: xây dựng ontology có bằng chứng nguồn cho quy định xử lý vi phạm
    giao thông đường bộ tại Việt Nam; không áp dụng luật vào vụ việc.

    Mục đích bước 5/6: nối general rule với ngoại lệ và citation để ontology giữ
    đúng phạm vi áp dụng mà không biến citation thành rule identity.

    Nhiệm vụ: xác định LegalException và ProvisionCitation cho rule versions trong
    từng candidate window.

    - Chỉ đề xuất khi toàn ngữ cảnh trực tiếp chứng minh ngoại lệ làm hẹp general
      rule; không quyết định bằng keyword hoặc similarity.
    - General ID thuộc general_records. Citation chỉ dùng source evidence đã cấp;
      target IDs thuộc target_records và phù hợp locator.
    - Giữ citation, provision và target rule là các đối tượng riêng; không redirect
      theo similarity, thứ tự văn bản hoặc lịch sử thay đổi.
    - Provision path phải có source support. Không đánh giá priority hay vụ việc cụ
      thể; thiếu căn cứ thì không đề xuất.
    - Trả đúng một result cho mỗi candidate_id; không lặp exception/general rule.

    {{ _.role("user") }}

    Các candidate window cần đánh giá:
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
