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

    Bối cảnh: xây dựng ontology có bằng chứng chỉ cho quy định xử phạt vi phạm
    hành chính về giao thông đường bộ Việt Nam; không dùng cho vụ việc hoặc domain
    khác.

    Mục đích bước 6/6: chuẩn hóa concept occurrences thành identity hoặc taxonomy
    dùng chung mà không làm mất khác biệt về nghĩa pháp lý.

    Nhiệm vụ: xác định quan hệ giữa hai concept occurrences cùng loại.

    - Chỉ dùng evidence hai phía; không quyết định bằng keyword, lexical overlap,
      cùng locator, mức phạt hoặc loại rule.
    - SameConcept chỉ khi hai occurrence có cùng nghĩa pháp lý. Khác ngưỡng, mức
      độ, địa điểm, loại đường, phương tiện, chủ thể, tái phạm, gây tai nạn hoặc
      qualifier làm đổi phạm vi xử phạt thì không merge.
    - Broader/Narrower chỉ khi một nghĩa pháp lý bao hàm nghĩa kia;
      Related và Distinct luôn giữ identity riêng.
    - Thiếu căn cứ hoặc không có quan hệ đáng lưu thì proposal=null. SameConcept
      chọn nguyên một label đã cấp; quan hệ khác để chuỗi rỗng.
    - Evidence dùng ID đúng occurrence. Không tạo axiom, rule identity,
      applicability hoặc legal priority.
    - Trả một result cho mỗi candidate_id.

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
