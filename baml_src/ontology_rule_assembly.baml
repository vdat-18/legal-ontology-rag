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
  candidate_id string @description(#"Sao chép nguyên văn candidate_id của input tương ứng."#)
  proposals RuleAssemblyProposal[] @description(#"Danh sách rỗng khi candidate không có nhóm record hợp lệ để hợp nhất."#)
}

function ProposeOntologyRuleAssemblies(windows: RuleAssemblyWindowInput[]) -> RuleAssemblyWindowResult[] {
  client OpenAICompatibleProvider
  prompt #"
    {{ _.role("system") }}

    Bối cảnh: xây dựng ontology có bằng chứng chỉ cho quy định xử phạt vi phạm
    hành chính về giao thông đường bộ Việt Nam.

    Mục đích bước 3/6: khôi phục một rule version bị chia thành nhiều population
    record do chunking, tránh lưu các mảnh của cùng rule như các rule độc lập.

    - Chỉ hợp nhất khi đồng thời: cùng một hành vi pháp lý đơn nhất; cùng chủ thể,
      vai trò, phương tiện và cây điều kiện tương đương về nghĩa; record chỉ bổ
      sung mảnh hậu quả/evidence bị tách.
    - Cây conditions của một rule chi phối mọi consequence trong rule đó. Vì vậy,
      không hợp nhất record vô điều kiện với record có điều kiện, hoặc các record
      có điều kiện khác nhau, dù cùng conduct và locator. Đặc biệt phải giữ riêng
      hậu quả kèm `nếu`, `trong trường hợp`, `trừ trường hợp` hoặc điều kiện tương
      đương; không được làm tiền phạt hay hậu quả khác kế thừa nhầm điều kiện ấy.
      Chỉ khác condition_id nhưng cùng cấu trúc và cùng nghĩa thì vẫn được xem là
      tương đương.
    - Giữ riêng các động từ/hành động pháp lý độc lập, kể cả khi chúng dùng chung
      đối tượng, bổ ngữ, Điểm/Khoản, mức phạt, danh sách, từ "hoặc" hoặc dấu chấm
      phẩy. Phần bổ ngữ bị lược để tránh lặp không biến hai hành động thành mảnh
      của cùng conduct; không nối nhiều hành vi thành một conduct label.
    - Không hợp nhất NaturalPerson với Organization; không dùng subject_kind=null
      để thay cho nhiều loại chủ thể. Khác role, vehicle, ngưỡng, qualifier hoặc
      phạm vi thì giữ riêng.
    - Cùng locator, source chunk hoặc hậu quả không chứng minh cùng rule; không
      hợp nhất chỉ để khử mức phạt lặp.
    - Mỗi proposal có ít nhất hai member ID duy nhất; các proposal không trùng
      member.
    - Bảo toàn conduct, applicability, conditions, consequences, validity,
      provision và evidence; giá trị chưa biết giữ null. Trước khi đề xuất, audit
      từng consequence của assembled_rule: chính cây conditions chung phải áp dụng
      đúng cho consequence đó trong source. Sai một consequence thì không hợp nhất
      các member ấy.
    - assembled_rule dùng ExplicitInterval và giữ nguyên ngày đã resolve; nếu cả
      hai ngày null thì dùng Unresolved. Không dùng DocumentDefault.
    - Không suy diễn role, LegalChange, LegalException, identity xuyên văn bản hay
      concept relation. Thiếu evidence, chỉ có citation rời hoặc khác rule thì
      không đề xuất.
    - Luôn trả đúng một RuleAssemblyWindowResult cho từng candidate window và sao
      chép candidate_id nguyên văn, không bỏ, đổi, rút gọn hoặc tạo ID mới.
    - Nếu một candidate không có nhóm record hợp lệ để hợp nhất, vẫn trả result
      của candidate đó với `proposals=[]`. Không được trả danh sách kết quả tổng
      thể rỗng khi input có candidate; kể cả một input cũng phải có một result.

    {{ _.role("user") }}

    Các candidate window cần đánh giá:
    {% for window in windows %}
    === candidate_id={{ window.candidate_id }} ===
    records={{ window.records }}
    === end candidate_id={{ window.candidate_id }} ===
    {% endfor %}

    {{ ctx.output_format }}
  "#
}
