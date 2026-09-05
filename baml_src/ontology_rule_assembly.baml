class AssemblyConditionSummary {
  condition_id string
  kind OntologyConditionKind
  text_vi string
  operand_ids string[]
  evidence OntologyEvidenceSource[]
}

class AssemblyConsequenceSummary {
  consequence_id string
  kind OntologyConsequenceKind
  minimum_amount string?
  maximum_amount string?
  currency OntologyCurrency?
  point_value int?
  consequence_label_vi string?
  minimum_duration_months int?
  maximum_duration_months int?
  target_label_vi string?
  responsible_role_label_vi string?
  evidence OntologyEvidenceSource[]
}

class OntologyPopulationSummary {
  population_id string
  source_chunk_id string
  conduct_label_vi string
  subject_kind OntologySubjectKind?
  role_labels_vi string[]
  vehicle_category_labels_vi string[]
  condition_root_id string?
  conditions AssemblyConditionSummary[]
  consequences AssemblyConsequenceSummary[]
  valid_from string?
  valid_to string?
  document_number string
  provision_path string
  evidence OntologyEvidenceSource[]
}

class RuleAssemblyWindowInput {
  candidate_id string
  records OntologyPopulationSummary[]
}

class RuleAssemblyProposal {
  member_population_ids string[] @description(#"Ít nhất hai member ID duy nhất cùng rule version."#)
  assembled_rule OntologyRuleEntry
  rationale_vi string
}

class RuleAssemblyWindowResult {
  candidate_id string
  proposals RuleAssemblyProposal[]
}

function ProposeOntologyRuleAssemblies(windows: RuleAssemblyWindowInput[]) -> RuleAssemblyWindowResult[] {
  client OpenAICompatibleProvider
  prompt #"
    {{ _.role("system") }}

    Bối cảnh: xây dựng ontology có bằng chứng chỉ cho quy định xử phạt vi phạm
    hành chính về giao thông đường bộ Việt Nam; không dùng cho vụ việc hoặc domain
    khác.

    Mục đích bước 3/6: khôi phục một rule version bị chia thành nhiều population
    record do chunking, tránh lưu các mảnh của cùng rule như các rule độc lập.

    Nhiệm vụ: hợp nhất các record thực sự là những phần của cùng một PenaltyRule
    trong cùng rule version.

    - Chỉ hợp nhất record cùng văn bản, rule version, hành vi và phạm vi khi source
      hoặc locator cho thấy chúng là các mảnh của một quy định. Tiền phạt, trừ
      điểm, hình phạt bổ sung và khắc phục có thể là hậu quả cộng dồn cùng rule.
    - Vị trí gần, câu chữ giống nhau, cùng mức phạt hoặc cùng loại phương tiện
      không chứng minh cùng rule; khác ngưỡng, qualifier, chủ thể hoặc phạm vi thì
      giữ riêng.
    - Mỗi proposal có ít nhất hai member ID duy nhất; các proposal không trùng
      member.
    - Bảo toàn conduct, applicability, condition tree, consequences, validity,
      provision path và evidence; giá trị chưa biết giữ null.
    - assembled_rule dùng ExplicitInterval và giữ nguyên ngày đã resolve; nếu cả
      hai ngày null thì dùng Unresolved. Không dùng DocumentDefault.
    - Không suy diễn role, LegalChange, LegalException, identity xuyên văn bản hay
      concept relation. Thiếu evidence, chỉ có citation rời hoặc khác rule thì
      không đề xuất.
    - Trả một result cho mỗi candidate_id.

    {{ _.role("user") }}

    Các candidate window cần đánh giá:
    {% for window in windows %}
    === candidate_id={{ window.candidate_id }} ===
    {% for record in window.records %}
    population_id={{ record.population_id }}
    source_chunk_id={{ record.source_chunk_id }}
    document={{ record.document_number }}
    provision={{ record.provision_path }}
    conduct={{ record.conduct_label_vi }}
    subject_kind={{ record.subject_kind }}
    roles={{ record.role_labels_vi }}
    vehicles={{ record.vehicle_category_labels_vi }}
    validity={{ record.valid_from }} -> {{ record.valid_to }}
    conditions={{ record.conditions }}
    consequences={{ record.consequences }}
    evidence={{ record.evidence }}
    ---
    {% endfor %}
    === end candidate_id={{ window.candidate_id }} ===
    {% endfor %}

    {{ ctx.output_format }}
  "#
}
