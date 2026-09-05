class RuleAssemblyConflictProposalInput {
  proposal_id string
  member_population_ids string[]
  assembled_rule OntologyPopulationSummary
  rationale_vi string
}

class RuleAssemblyConflictInput {
  component_id string
  source_records OntologyPopulationSummary[]
  competing_proposals RuleAssemblyConflictProposalInput[]
}

class RuleAssemblyConflictResult {
  component_id string
  proposals RuleAssemblyProposal[]
}

function ResolveOntologyRuleAssemblyConflicts(components: RuleAssemblyConflictInput[]) -> RuleAssemblyConflictResult[] {
  client OpenAICompatibleProvider
  prompt #"
    {{ _.role("system") }}

    Bối cảnh: xây dựng ontology có bằng chứng chỉ cho quy định xử phạt vi phạm
    hành chính về giao thông đường bộ Việt Nam.

    Mục đích bước 3: review các đề xuất hợp nhất theo toàn bộ proposal component,
    loại over-merge và tạo kết quả nhất quán toàn cục. Component có thể chứa một
    hoặc nhiều proposal cạnh tranh.

    - Chỉ hợp nhất source record bị tách nhưng cùng mô tả một hành vi pháp lý đơn
      nhất, cùng chủ thể, vai trò, phương tiện và điều kiện áp dụng. Record bổ sung
      hậu quả hoặc evidence cho đúng hành vi đó có thể thuộc cùng rule.
    - Giữ riêng các động từ/hành động pháp lý độc lập, kể cả khi chúng dùng chung
      đối tượng, bổ ngữ, Điểm/Khoản, mức phạt, danh sách, từ "hoặc" hoặc dấu chấm
      phẩy. Phần bổ ngữ bị lược để tránh lặp không biến hai hành động thành mảnh
      của cùng conduct; không nối nhiều hành vi thành một conduct label.
    - Không hợp nhất NaturalPerson với Organization; không dùng subject_kind=null
      để thay nhiều loại chủ thể. Khác role, vehicle, ngưỡng, qualifier hoặc phạm
      vi thì giữ riêng.
    - Với từng output proposal, xác nhận các record trùng/lặp nghĩa cùng một hành
      động, hoặc chỉ bổ sung hậu quả/evidence còn thiếu cho chính hành động đó;
      đồng thời toàn bộ applicability phải giống nhau. Sai một điều thì loại.
    - Proposal đầu vào chỉ là gợi ý. Có thể giữ, sửa, thay hoặc loại; trả [] nếu
      chưa đủ căn cứ.
    - Mỗi proposal có ít nhất hai member ID duy nhất từ source_records; các
      proposal trong một component không trùng member.
    - Bảo toàn conduct, applicability, condition tree, consequences, validity,
      provision path và evidence; không suy diễn ngoài nguồn.
    - Cùng locator, source chunk, câu chữ gần nhau hoặc cùng mức phạt không chứng
      minh cùng rule.
    - Dùng ExplicitInterval và ngày đã resolve; nếu cả hai ngày null thì dùng
      Unresolved. Không dùng DocumentDefault.
    - Trả đúng một result cho mỗi component_id.

    {{ _.role("user") }}

    Các conflict component:
    {% for component in components %}
    === component_id={{ component.component_id }} ===
    source_records={{ component.source_records }}
    competing_proposals={{ component.competing_proposals }}
    === end component_id={{ component.component_id }} ===
    {% endfor %}

    {{ ctx.output_format }}
  "#
}
