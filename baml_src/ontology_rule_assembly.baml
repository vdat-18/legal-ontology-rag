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
  member_population_ids string[] @description(#"
    Ít nhất hai population_id khác nhau từ đúng candidate window. Các member
    phải cùng cấu thành một rule phiên bản cụ thể.
  "#)
  assembled_rule OntologyRuleEntry @description(#"
    Rule hợp nhất đầy đủ, giữ source evidence cho mọi condition và consequence.
  "#)
  rationale_vi string @description(#"Giải thích ngắn gọn vì sao source chứng minh đây là cùng một rule."#)
}

class RuleAssemblyWindowResult {
  candidate_id string
  proposals RuleAssemblyProposal[] @description(#"
    Mảng rỗng nếu các record là rule khác nhau hoặc bằng chứng chưa đủ. Các
    proposal trong cùng result không được dùng trùng member_population_ids.
  "#)
}

function ProposeOntologyRuleAssemblies(windows: RuleAssemblyWindowInput[]) -> RuleAssemblyWindowResult[] {
  client OpenAICompatibleProvider
  prompt #"
    {{ _.role("user") }}

    Bạn là chuyên gia ontology pháp luật. Xác định record nào trong từng window
    thực sự là các phần của cùng một PenaltyRule trong cùng phiên bản văn bản.

    Nguyên tắc bắt buộc:
    - Chỉ dùng dữ liệu và evidence trong window; vị trí gần hoặc câu chữ giống
      nhau không đủ chứng minh cùng rule.
    - Member phải là ID duy nhất trong window, có ít nhất hai ID và không chồng
      lấn giữa các proposal.
    - Rule hợp nhất phải bảo toàn conduct, applicability, condition tree,
      consequences, validity, provision path và source evidence.
    - Giữ null cho dữ liệu chưa biết; không tự suy diễn role, thay đổi pháp lý,
      ngoại lệ, identity xuyên văn bản hay concept equivalence.
    - Nếu nguồn chưa đủ hoặc các record là rule riêng, trả proposals rỗng.
    - Trả đúng một result cho mỗi candidate_id.

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
