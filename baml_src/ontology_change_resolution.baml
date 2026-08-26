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

    Bạn là chuyên gia ontology pháp luật. Hãy đề xuất các LegalChange event giữa rule version của hai văn bản khác nhau.
    Candidate window được tạo exhaustive theo thứ tự văn bản và batch có giới hạn; việc hai record xuất hiện cùng window
    không chứng minh chúng là cùng rule hoặc có quan hệ kế thừa.

    Nguyên tắc bắt buộc:
    - Chỉ đề xuất khi change_evidence_sources chứa phát biểu pháp lý trực tiếp về sửa đổi, thay thế, bãi bỏ hoặc làm hết
      hiệu lực rule/provision trước, cùng bằng chứng đủ để xác định target và ngày hiệu lực.
    - Không suy ra change từ conduct, role, vehicle, condition, consequence, provision path hoặc câu chữ tương tự.
    - Không suy ra change chỉ từ thứ tự ban hành, ngày hiệu lực, hay việc valid_to đang null.
    - earlier_population_id phải thuộc earlier_records. later_population_id nếu có phải thuộc later_records.
    - Amendment/Replacement thông thường phải chỉ ra later rule. RepealOrSupersession có thể không có later rule chỉ khi
      source trực tiếp chứng minh điều đó. Nếu candidate chưa chứa later rule cần thiết, không tự chọn record gần giống.
    - evidence chỉ dùng evidence_id từ change_evidence_sources; không dùng knowledge ngoài source.
    - effective_on và change_provision_path phải được source hỗ trợ. Không tự dùng metadata date làm kết luận pháp lý.
    - Không tạo enduring rule identity, concept equivalence, exception redirect hay transitional applicability.
    - Nếu evidence chưa đủ, trả proposals rỗng. Human reviewer quyết định cuối cùng.
    - Trả đúng một result cho mỗi candidate_id; trong một result không dùng cùng earlier hoặc later rule cho nhiều proposal.

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
