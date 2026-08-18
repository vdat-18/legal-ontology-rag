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
  evidence OntologyEvidenceEntry[]
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
  left_evidence OntologyEvidenceEntry[] @description(#"Exact evidence supplied with the left occurrence."#)
  right_evidence OntologyEvidenceEntry[] @description(#"Exact evidence supplied with the right occurrence."#)
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

    Bạn là chuyên gia ontology pháp luật. Hãy đánh giá quan hệ giữa hai concept occurrence cùng loại. Candidate pairs
    được tạo exhaustive; việc hai occurrence xuất hiện trong một pair không chứng minh chúng tương đương hay liên quan.

    Nguyên tắc bắt buộc:
    - Chỉ đề xuất khi evidence của cả hai phía đủ để xác định ý nghĩa pháp lý của từng concept.
    - Không quyết định chỉ bằng lexical overlap, keyword, cùng provision path, cùng mức phạt hoặc cùng loại rule.
    - SameConcept chỉ dùng khi hai occurrence biểu đạt cùng một concept pháp lý có thể dùng chung canonical identity.
    - LeftBroaderThanRight/LeftNarrowerThanRight chỉ dùng cho quan hệ phân cấp pháp lý thực sự, không dùng cho hai rule
      có phạm vi khác nhau nhưng concept không có quan hệ taxonomy.
    - Related giữ hai identity riêng; Distinct ghi nhận một phân biệt pháp lý quan trọng. Nếu evidence chưa đủ hoặc không
      có quan hệ đáng ghi nhận, trả proposal=null.
    - preferred_label_vi của SameConcept phải copy chính xác label_vi của left hoặc right. Quan hệ khác phải để rỗng.
    - left_evidence và right_evidence phải là exact quotation từ evidence đã cung cấp đúng occurrence/chunk_id.
    - Không tạo owl:equivalentClass, rdfs:subClassOf, rule identity, applicability hoặc legal priority.
    - Human reviewer quyết định cuối cùng. Trả đúng một result cho mỗi candidate_id.

    {% for pair in pairs %}
    === candidate_id={{ pair.candidate_id }} ===
    left={{ pair.left }}
    right={{ pair.right }}
    === end candidate_id={{ pair.candidate_id }} ===
    {% endfor %}

    {{ ctx.output_format }}
  "#
}
