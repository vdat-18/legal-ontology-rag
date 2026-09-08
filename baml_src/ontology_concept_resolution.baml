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

class ConceptResolutionInventoryInput {
  candidate_id string
  kind OntologyConceptKind
  occurrences OntologyConceptOccurrenceInput[]
}

class ConceptRelationProposalEntry {
  left_occurrence_id string
  right_occurrence_id string
  relation OntologyConceptRelation
  preferred_label_vi string @description(#"
    SameConcept: chọn nguyên một label tiếng Việt đã cấp; quan hệ khác: chuỗi rỗng.
  "#)
  left_evidence OntologyEvidenceReference[] @description(#"Evidence của left occurrence."#)
  right_evidence OntologyEvidenceReference[] @description(#"Evidence của right occurrence."#)
  rationale_vi string @description(#"Lý do ngắn gọn dựa trên ý nghĩa pháp lý."#)
}

class ConceptResolutionInventoryResult {
  candidate_id string
  proposals ConceptRelationProposalEntry[]
}

function ProposeOntologyConceptRelations(inventories: ConceptResolutionInventoryInput[]) -> ConceptResolutionInventoryResult[] {
  client OpenAICompatibleProvider
  prompt #"
    {{ _.role("system") }}

    Bối cảnh: xây dựng ontology có bằng chứng chỉ cho quy định xử phạt vi phạm
    hành chính về giao thông đường bộ Việt Nam; không dùng cho vụ việc hoặc domain
    khác.

    Mục đích bước 6/6: chuẩn hóa concept occurrences thành identity hoặc taxonomy
    dùng chung mà không làm mất khác biệt về nghĩa pháp lý.

    Nhiệm vụ: chuẩn hóa toàn bộ nhãn trong mỗi inventory cùng loại và chỉ trả
    các quan hệ có giá trị cho controlled vocabulary dùng chung.

    - Xem xét toàn cục mọi occurrence trong inventory. Chỉ dùng meaning và
      evidence hai phía; không quyết định chỉ bằng lexical overlap, cùng locator,
      mức phạt hoặc loại rule.
    - SameConcept chỉ khi hai occurrence có cùng nghĩa pháp lý. Khác ngưỡng, mức
      độ, địa điểm, loại đường, phương tiện, chủ thể, tái phạm, gây tai nạn hoặc
      qualifier làm đổi phạm vi xử phạt thì không merge.
    - Broader/Narrower chỉ khi một nghĩa pháp lý bao hàm nghĩa kia;
      Related luôn giữ identity riêng; cặp phân biệt hoặc không liên quan thì bỏ.
    - Chỉ xuất SameConcept, Broader/Narrower hoặc Related khi có căn cứ rõ.
      Không xuất Distinct và không xuất pair không có quan hệ đáng lưu.
      SameConcept chọn nguyên một label đã cấp; quan hệ khác để chuỗi rỗng.
      Mọi SameConcept trong cùng một nhóm bắc cầu phải chọn cùng preferred label.
    - Mỗi unordered pair xuất tối đa một lần. Evidence và occurrence ID phải
      thuộc đúng inventory. Không tạo axiom, rule identity, applicability hoặc
      legal priority.
    - Trả một result cho mỗi candidate_id, với proposals=[] nếu không có quan hệ.

    {{ _.role("user") }}

    Các controlled-vocabulary inventory cần đánh giá:
    {% for inventory in inventories %}
    === candidate_id={{ inventory.candidate_id }} kind={{ inventory.kind }} ===
    occurrences={{ inventory.occurrences }}
    === end candidate_id={{ inventory.candidate_id }} ===
    {% endfor %}

    {{ ctx.output_format }}
  "#
}
