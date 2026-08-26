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
    Rule hợp nhất đầy đủ, giữ exact evidence riêng cho mọi condition và
    consequence. Không bỏ provenance chỉ vì hai record được hợp nhất.
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

    Bạn là chuyên gia ontology pháp luật. Nhiệm vụ là đề xuất hợp nhất các
    population record được trích từ những chunk gần nhau trong cùng một văn
    bản thành các PenaltyRule hoàn chỉnh. Candidate window chỉ là retrieval
    theo vị trí; proximity không chứng minh các record là cùng rule.

    Nguyên tắc bắt buộc:
    - Chỉ đề xuất assembly khi exact evidence cho thấy các record là những
      thành phần của cùng một rule phiên bản cụ thể.
    - Không hợp nhất chỉ vì conduct, subject, vehicle, mức tiền hoặc câu chữ
      tương tự. Không dùng kiến thức ngoài evidence.
    - Mỗi member_population_ids phải lấy từ đúng window, có ít nhất hai ID
      khác nhau. Các proposal trong một window không được chồng lấn member.
    - assembled_rule phải giữ toàn bộ conduct, applicability, condition tree,
      consequence, validity, provision path và exact evidence cần thiết.
    - Không biến thiếu dữ liệu thành 0, false, vô hạn hoặc không áp dụng.
    - Owner, driver, operator và custodian là các role khác nhau nếu source
      không trực tiếp đồng nhất.
    - Không tạo amendment, replacement, repeal, cross-document identity,
      exception redirect hoặc concept equivalence.
    - Nếu bằng chứng chưa đủ hoặc các record là các rule riêng, trả proposals
      rỗng. Human reviewer sẽ quyết định cuối cùng; output này chỉ là proposal.
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
