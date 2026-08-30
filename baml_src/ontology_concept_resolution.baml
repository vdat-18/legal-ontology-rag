enum OntologyConceptKind {
  RegulatedConduct
  LegalSubjectRole
  VehicleCategory
}

enum OntologyConceptRelation {
  SameConcept
  LeftBroaderThanRight
  LeftNarrowerThanRight
  Related
  Distinct
}

class OntologyConceptOccurrenceInput {
  occurrence_id string
  kind OntologyConceptKind
  label_vi string
  population_id string
  document_number string
  provision_path string
  evidence OntologyEvidenceSource[]
}

class ConceptResolutionPairInput {
  candidate_id string
  left OntologyConceptOccurrenceInput
  right OntologyConceptOccurrenceInput
}

class ConceptRelationProposalEntry {
  relation OntologyConceptRelation
  preferred_label_vi string @description(#"
    SameConcept: chọn nguyên một label tiếng Việt đã cấp; quan hệ khác: chuỗi rỗng.
  "#)
  left_evidence OntologyEvidenceReference[] @description(#"Evidence của left occurrence."#)
  right_evidence OntologyEvidenceReference[] @description(#"Evidence của right occurrence."#)
  rationale_vi string @description(#"Lý do ngắn gọn dựa trên ý nghĩa pháp lý."#)
}

class ConceptResolutionPairResult {
  candidate_id string
  proposal ConceptRelationProposalEntry?
}

function ProposeOntologyConceptRelations(pairs: ConceptResolutionPairInput[]) -> ConceptResolutionPairResult[] {
  client OpenAICompatibleProvider
  prompt #"
    {{ _.role("system") }}

    Bối cảnh: xây dựng ontology có bằng chứng nguồn cho quy định xử lý vi phạm
    giao thông đường bộ tại Việt Nam; không áp dụng luật vào vụ việc.

    Mục đích bước 6/6: chuẩn hóa concept occurrences thành identity hoặc taxonomy
    dùng chung mà không làm mất khác biệt về nghĩa pháp lý.

    Nhiệm vụ: xác định quan hệ giữa hai concept occurrences cùng loại trong từng
    candidate pair.

    - Chỉ dùng evidence hai phía; không quyết định bằng keyword, lexical overlap,
      cùng locator, mức phạt hoặc loại rule.
    - SameConcept chỉ khi dùng chung identity. Broader/Narrower chỉ cho quan hệ
      taxonomy thực sự; Related và Distinct luôn giữ identity riêng.
    - Thiếu căn cứ hoặc không có quan hệ đáng lưu thì proposal=null. SameConcept
      chọn nguyên một label đã cấp; quan hệ khác để chuỗi rỗng.
    - Evidence chỉ dùng ID của đúng occurrence. Không tạo class axiom, rule
      identity, applicability hoặc legal priority.
    - Trả đúng một result cho mỗi candidate_id.

    {{ _.role("user") }}

    Các candidate pair cần đánh giá:
    {% for pair in pairs %}
    === candidate_id={{ pair.candidate_id }} ===
    left={{ pair.left }}
    right={{ pair.right }}
    === end candidate_id={{ pair.candidate_id }} ===
    {% endfor %}

    {{ ctx.output_format }}
  "#
}
