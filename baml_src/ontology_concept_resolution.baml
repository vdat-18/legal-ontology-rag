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
    For SameConcept, exactly one supplied left/right Vietnamese label selected as the canonical preferred label.
    For every other relation, return an empty string.
  "#)
  left_evidence OntologyEvidenceReference[] @description(#"Evidence references supplied with the left occurrence."#)
  right_evidence OntologyEvidenceReference[] @description(#"Evidence references supplied with the right occurrence."#)
  rationale_vi string @description(#"Giải thích ngắn gọn dựa trên ý nghĩa pháp lý, không chỉ dựa trên từ giống nhau."#)
}

class ConceptResolutionPairResult {
  candidate_id string
  proposal ConceptRelationProposalEntry?
}

function ProposeOntologyConceptRelations(pairs: ConceptResolutionPairInput[]) -> ConceptResolutionPairResult[] {
  client OpenAICompatibleProvider
  prompt #"
    {{ _.role("user") }}

    Bạn là chuyên gia ontology pháp luật. Xác định quan hệ giữa hai concept
    occurrence cùng loại trong từng candidate pair.

    Nguyên tắc bắt buộc:
    - Chỉ dùng evidence hai phía; không quyết định bằng keyword, lexical overlap,
      cùng locator, cùng mức phạt hay cùng loại rule.
    - SameConcept chỉ khi có thể dùng chung identity. Broader/Narrower chỉ cho
      taxonomy thực sự. Related và Distinct luôn giữ identity riêng.
    - Nếu chưa đủ căn cứ hoặc không có quan hệ đáng lưu, trả proposal=null.
    - SameConcept phải chọn nguyên một label đã cung cấp; quan hệ khác để label
      rỗng. Evidence chỉ dùng ID của đúng occurrence.
    - Không tạo class axiom, rule identity, applicability hay legal priority.
      Trả đúng một result cho mỗi candidate_id.

    {% for pair in pairs %}
    === candidate_id={{ pair.candidate_id }} ===
    left={{ pair.left }}
    right={{ pair.right }}
    === end candidate_id={{ pair.candidate_id }} ===
    {% endfor %}

    {{ ctx.output_format }}
  "#
}
