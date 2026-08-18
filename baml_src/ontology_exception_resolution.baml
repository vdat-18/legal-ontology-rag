class ExceptionCitationProposalEntry {
  citation OntologyEvidenceEntry @description(#"Exact citation wording and its source chunk, not a normalized locator."#)
  target_population_ids string[] @description(#"
    One or more version-specific rule IDs from target_records whose grounded provisions resolve this grouped citation.
  "#)
}

class LegalExceptionProposalEntry {
  general_population_id string @description(#"Exactly one rule ID from general_records."#)
  exception_provision_path string @description(#"Fullest structural path of the provision stating the exception."#)
  evidence OntologyEvidenceEntry[] @description(#"Exact source quotations that directly state the complete exception."#)
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
  source_chunks ChangeSourceChunkInput[]
}

class ExceptionResolutionWindowResult {
  candidate_id string
  proposals LegalExceptionProposalEntry[]
}

function ProposeLegalExceptionResolutions(windows: ExceptionResolutionWindowInput[]) -> ExceptionResolutionWindowResult[] {
  client OpenAICompatibleProvider
  prompt #"
    {{ _.role("user") }}

    Bạn là chuyên gia ontology pháp luật. Hãy đề xuất LegalException và grouped ProvisionCitation cho các rule version
    đã canonicalize. Candidate window được tạo exhaustive theo cấu trúc và batch; membership không chứng minh exception,
    citation resolution hay redirect.

    Nguyên tắc bắt buộc:
    - Chỉ tạo proposal khi source_chunks trực tiếp chứng minh một ngoại lệ pháp lý làm hẹp general rule cụ thể.
    - Không phân loại chỉ vì có token như "trừ", "ngoại trừ", phủ định, hoặc câu chữ tương tự. Phải đọc toàn ngữ cảnh.
    - general_population_id phải thuộc general_records và là rule version thực sự bị ngoại lệ làm hẹp.
    - Mỗi citation phải copy nguyên văn từ source_chunks, giữ đúng chunk_id và grouping của locator nguồn.
    - target_population_ids phải thuộc target_records. Chỉ chọn rule version có provision và validity phù hợp với dẫn chiếu.
    - Không đồng nhất citation string, cited provision và target rule. Một citation có thể resolve nhiều target rules.
    - Không redirect theo lexical similarity, conduct label, provision path gần giống, document order hoặc change history.
    - evidence phải là exact quotation nêu đầy đủ exception; exception_provision_path phải được source hỗ trợ.
    - Nếu thiếu general rule, thiếu target candidate, citation chưa resolve chắc chắn hoặc source chưa đủ, trả proposals rỗng.
    - Không đánh giá priority hoặc applicability cho occurrence cụ thể. Human reviewer quyết định cuối cùng.
    - Trả đúng một result cho mỗi candidate_id. Trong một result không lặp cùng exception statement/general rule.

    {% for window in windows %}
    === candidate_id={{ window.candidate_id }} ===
    source_document={{ window.source_document_number }}
    general_records={{ window.general_records }}
    target_records={{ window.target_records }}
    source_chunks={{ window.source_chunks }}
    === end candidate_id={{ window.candidate_id }} ===
    {% endfor %}

    {{ ctx.output_format }}
  "#
}
