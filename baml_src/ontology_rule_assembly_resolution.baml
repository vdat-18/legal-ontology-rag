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
      nhất, cùng chủ thể, vai trò, phương tiện và cây điều kiện tương đương về
      nghĩa. Record bổ sung hậu quả hoặc evidence cho đúng hành vi đó chỉ có thể
      thuộc cùng rule khi chính điều kiện chung áp dụng cho mọi hậu quả được gộp.
    - Cây conditions của một rule chi phối mọi consequence trong rule đó. Không
      hợp nhất record vô điều kiện với record có điều kiện, hoặc các record có điều
      kiện khác nhau, dù cùng conduct, locator hay cùng câu dẫn chiếu. Hậu quả kèm
      `nếu`, `trong trường hợp`, `trừ trường hợp` hoặc điều kiện tương đương phải
      giữ ở rule riêng; không được làm tiền phạt hay hậu quả khác kế thừa nhầm điều
      kiện. Chỉ khác condition_id nhưng cùng cấu trúc và cùng nghĩa thì vẫn được
      xem là tương đương.
    - Giữ riêng các động từ/hành động pháp lý độc lập, kể cả khi chúng dùng chung
      đối tượng, bổ ngữ, Điểm/Khoản, mức phạt, danh sách, từ "hoặc" hoặc dấu chấm
      phẩy. Phần bổ ngữ bị lược để tránh lặp không biến hai hành động thành mảnh
      của cùng conduct; không nối nhiều hành vi thành một conduct label.
    - So sánh conduct leaf của từng source record, không chỉ câu evidence chung.
      Các nhánh độc lập được Step 1 tách từ cùng một câu vẫn là các conduct khác
      nhau. Không hợp nhất hai leaf khác vị ngữ hoặc khác phương án lựa chọn, và
      không mở rộng assembled conduct thành hợp của các leaf thành viên. Ví dụ
      tổng quát, `không thực hiện A hoặc B` tạo hai leaf A và B; việc chúng cùng
      source, locator và hậu quả không cho phép ghép lại thành `A hoặc B`.
    - Cổng bắt buộc trước khi giữ proposal: so sánh từng conduct_label_vi như một
      hành vi độc lập, không mở rộng label từ evidence. Mọi member phải mô tả cùng
      một vị ngữ pháp lý và cùng một phương án đối tượng. Nếu assembled conduct có
      bất kỳ động từ hoặc phương án nào không có trong conduct label của từng
      member, proposal là over-merge và phải bị loại. Hai member chỉ khác nhau ở
      hậu quả/evidence vẫn phải giữ nguyên cùng conduct leaf; hai label bổ sung
      các hành động khác nhau cho nhau không phải các mảnh của một rule.
    - Không hợp nhất NaturalPerson với Organization; không dùng subject_kind=null
      để thay nhiều loại chủ thể. Khác role, vehicle, ngưỡng, qualifier hoặc phạm
      vi thì giữ riêng.
    - Với từng output proposal, xác nhận các record trùng/lặp nghĩa cùng một hành
      động, hoặc chỉ bổ sung hậu quả/evidence còn thiếu cho chính hành động đó;
      đồng thời toàn bộ applicability phải giống nhau. Audit từng consequence của
      assembled_rule: cây conditions chung phải được source áp dụng đúng cho chính
      consequence đó. Nếu một consequence có phạm vi điều kiện khác, tách member
      theo phạm vi tương đương hoặc loại proposal; sai một điều thì loại.
    - Proposal đầu vào chỉ là gợi ý. Có thể giữ, sửa, thay hoặc loại; trả [] nếu
      chưa đủ căn cứ. Conduct leaf không đồng nhất là không đủ căn cứ. Chỉ sửa
      assembled_rule trong phạm vi nghĩa chung có mặt ở mọi member, không dùng
      source rộng hơn để đưa sibling conduct trở lại.
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
