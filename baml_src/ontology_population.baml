enum OntologySubjectKind {
  NaturalPerson
  Organization
}

enum OntologyConditionKind {
  Atomic
  AllOf
  AnyOf
  Not
}

enum OntologyConsequenceKind {
  MonetaryPenalty
  PointDeduction
  AdditionalSanction
  RemedialMeasure
}

enum OntologyCurrency {
  VND
}

class OntologyEvidenceSource {
  evidence_id string @description(#"Định danh source evidence được cung cấp trong input hiện tại."#)
  chunk_id string
  text string @description(#"Đoạn source nguyên văn; model không được sửa hoặc diễn giải lại."#)
}

class OntologyEvidenceReference {
  evidence_id string @description(#"Phải trùng với một evidence_id được cung cấp trong input hiện tại."#)
  start_char int? @description(#"Offset Unicode 0-based, inclusive; null để dùng toàn source."#)
  end_char int? @description(#"Offset Unicode 0-based, exclusive; null để dùng toàn source."#)
}

class OntologyConditionEntry {
  condition_id string @description(#"Định danh cục bộ duy nhất trong một rule, ví dụ condition_1."#)
  kind OntologyConditionKind
  text_vi string @description(#"Nội dung điều kiện ngắn gọn bằng tiếng Việt."#)
  operand_ids string[] @description(#"
    ID của các condition con. Atomic phải rỗng; AllOf/AnyOf có ít nhất hai;
    Not có đúng một.
  "#)
  evidence OntologyEvidenceReference[] @description(#"
    Mọi expression, kể cả AllOf, AnyOf và Not, phải có ít nhất một source
    evidence trực tiếp. Có thể dùng lại evidence của operand khi cùng wording.
  "#)
}

class OntologyConsequenceEntry {
  consequence_id string @description(#"Định danh cục bộ duy nhất trong một rule."#)
  kind OntologyConsequenceKind
  minimum_amount string? @description(#"Số thập phân không có dấu phân cách hàng nghìn; chỉ dùng cho MonetaryPenalty."#)
  maximum_amount string? @description(#"Số thập phân không có dấu phân cách hàng nghìn; chỉ dùng cho MonetaryPenalty."#)
  currency OntologyCurrency? @description(#"Đơn vị tiền tệ được source nêu; bắt buộc cho MonetaryPenalty."#)
  point_value int? @description(#"Số điểm nguyên dương; chỉ dùng cho PointDeduction."#)
  consequence_label_vi string? @description(#"
    Tên tác động pháp lý được source nêu. Bắt buộc cho AdditionalSanction và
    RemedialMeasure.
  "#)
  minimum_duration_months int? @description(#"
    Cận dưới thời hạn tính theo tháng khi source nêu một khoảng thời hạn;
    chỉ dùng cho AdditionalSanction.
  "#)
  maximum_duration_months int? @description(#"
    Cận trên thời hạn tính theo tháng khi source nêu một khoảng thời hạn;
    chỉ dùng cho AdditionalSanction.
  "#)
  target_label_vi string? @description(#"Đối tượng chịu tác động, ví dụ giấy phép lái xe hoặc biển số."#)
  responsible_role_label_vi string? @description(#"
    Vai trò phải thực hiện RemedialMeasure. Bắt buộc cho RemedialMeasure và
    phải được source xác định trực tiếp hoặc qua ngữ cảnh cấu trúc trong window.
  "#)
  evidence OntologyEvidenceReference[]
}

class OntologyRuleEntry {
  conduct_label_vi string @description(#"
    Một hành vi được điều chỉnh có thể đối chiếu độc lập. Không gộp nhiều
    hành vi thay thế trong cùng điểm thành một nhãn bao quát. Chỉ giữ nội
    dung cấu thành hành vi và qualifier nội tại; ngoại lệ giới hạn
    phạm vi áp dụng phải nằm trong condition tree, không chỉ nằm trong nhãn.
  "#)
  subject_kind OntologySubjectKind?
  role_labels_vi string[] @description(#"Vai trò áp dụng được văn bản nêu rõ; không suy diễn owner thành driver."#)
  vehicle_category_labels_vi string[] @description(#"Loại phương tiện được văn bản nêu rõ."#)
  condition_root_id string? @description(#"
    ID condition gốc. Chỉ null khi source không có điều kiện hoặc ngoại lệ
    riêng giới hạn phạm vi áp dụng của rule.
  "#)
  conditions OntologyConditionEntry[] @description(#"
    Cây điều kiện giữ đầy đủ qualifier và ngoại lệ áp dụng có source evidence.
    Ngoại lệ áp dụng được biểu diễn bằng Not, không chỉ chép vào conduct label.
  "#)
  consequences OntologyConsequenceEntry[] @description(#"Ít nhất một hậu quả pháp lý có nguồn hỗ trợ."#)
  valid_from string? @description(#"YYYY-MM-DD, chỉ điền khi window nêu hiệu lực riêng cho rule."#)
  valid_to string? @description(#"YYYY-MM-DD, null nghĩa là chưa biết chứ không phải vô hạn."#)
  provision_path string @description(#"
    Đường dẫn định danh phải ghi đầy đủ tên cấp tiếng Việt theo thứ tự Chương >
    Mục > Điều > Khoản > Điểm. Mỗi cấp gồm tên cấp và số hoặc ký hiệu, không
    gồm tiêu đề; bỏ cấp không tồn tại.
  "#)
  evidence OntologyEvidenceReference[] @description(#"Bằng chứng chung cho conduct, applicability và rule."#)
}

class OntologyPopulationBatchResult {
  target_chunk_id string @description(#"Phải trùng với target_chunk_id đầu vào."#)
  rules OntologyRuleEntry[]
}

class OntologyTargetConductInput {
  target_chunk_id string
  structural_context string @description(#"Metadata cấu trúc của Target; không phải quoted evidence."#)
  evidence_sources OntologyEvidenceSource[]
}

class OntologyTargetConductEntry {
  conduct_id string @description(#"ID cục bộ duy nhất trong Target."#)
  conduct_label_vi string @description(#"
    Một conduct nguyên tử được nêu cụ thể trong Target, giữ đúng phủ định và
    qualifier tiếng Việt chi phối nó.
  "#)
  provision_path string @description(#"Đường dẫn đầy đủ tên cấp tiếng Việt của conduct."#)
  evidence OntologyEvidenceReference[] @description(#"Evidence chỉ từ Target."#)
}

class OntologyTargetConductBatchResult {
  target_chunk_id string
  conducts OntologyTargetConductEntry[] @description(#"
    Danh sách đầy đủ mọi conduct nguyên tử được Target trực tiếp mô tả; rỗng
    chỉ khi đã audit toàn bộ Target và không có conduct hợp lệ.
  "#)
}

class OntologyChunkWindowEntry {
  role ChunkWindowRole
  chunk_id string
  structural_context string @description(#"Metadata cấu trúc; không phải source evidence."#)
  evidence_sources OntologyEvidenceSource[]
}

class OntologyPopulationBindingItem {
  target_chunk_id string
  window OntologyChunkWindowEntry[]
  target_conducts OntologyTargetConductEntry[]
}

class OntologyBoundRuleEntry {
  target_conduct_id string @description(#"Phải trùng conduct_id được cung cấp trong target_conducts."#)
  rule OntologyRuleEntry
}

class OntologyPopulationBindingBatchResult {
  target_chunk_id string
  rules OntologyBoundRuleEntry[] @description(#"
    Mọi rule variant có đủ source support cho closed target conduct inventory.
  "#)
}

function ExtractOntologyTargetConductBatch(items: OntologyTargetConductInput[]) -> OntologyTargetConductBatchResult[] {
  client OpenAICompatibleProvider
  prompt #"
    {{ _.role("user") }}

    Lập closed inventory mọi hành vi vi phạm được Target trực tiếp mô tả. Đây là
    pha inventory, chưa gắn hậu quả pháp lý và không kết luận vụ việc thực tế.

    Nguồn và phạm vi
    - Chỉ dùng Target evidence và structural context của chính item. Structural
      context chỉ xác định ancestry; không phải evidence. Không dùng kiến thức
      ngoài hoặc suy ra nội dung của provision được dẫn chiếu.
    - Audit Target theo thứ tự nguồn, đến hết mọi Khoản/Điểm và mọi evidence
      source. Chỉ trả conducts rỗng sau khi toàn bộ Target không có conduct hợp lệ.

    Quyết định conduct
    - Tạo entry cho hành động, không hành động hoặc nghĩa vụ cụ thể có thể độc
      lập cấu thành vi phạm và được Target diễn đạt trực tiếp.
    - Không tạo entry từ định nghĩa, thủ tục, thẩm quyền, tên hậu quả, đối tượng
      chịu hậu quả hoặc clause chỉ nói `hành vi quy định tại <provision>`.
    - Clause hậu quả vẫn có thể trực tiếp định nghĩa một variant nếu nó nhắc lại
      đầy đủ conduct cùng qualifier như tái phạm hoặc gây hậu quả. Clause chỉ
      dẫn chiếu locator mà không nhắc lại conduct vẫn là catalog.
    - Trong mỗi Điểm/Khoản, tách các lựa chọn có thể vi phạm độc lập. Không tách
      danh sách đối tượng, cách diễn đạt tương đương hoặc các thành phần phải kết
      hợp mới tạo thành một hành vi.
    - Giữ đúng phủ định, qualifier cấu thành và phạm vi ngữ pháp tiếng Việt.
      Ngoại lệ chỉ giới hạn applicability không phải một phần của conduct label.
    - Giữ nguyên Unicode và ký hiệu locator pháp lý; `d` và `đ` là khác nhau.
      provision_path dùng đủ cấp có trong source theo dạng `Chương > Mục > Điều
      > Khoản > Điểm`, không kèm tiêu đề.
    - Evidence phải thuộc Target và trực tiếp chứa wording của conduct. Trước khi
      trả, đối chiếu lại từng entry với source và loại mọi entry chỉ dựa vào dẫn chiếu.

    {% for item in items %}
    === target_chunk_id={{ item.target_chunk_id }} ===
    Structural context (không phải source evidence):
    {{ item.structural_context }}
    Target evidence sources:
    {% for evidence in item.evidence_sources %}
    [{{ evidence.evidence_id }}] chunk_id={{ evidence.chunk_id }}
    {{ evidence.text }}
    {% endfor %}
    === end target_chunk_id={{ item.target_chunk_id }} ===
    {% endfor %}

    {{ ctx.output_format }}
  "#
}

function BindOntologyPopulationBatch(items: OntologyPopulationBindingItem[]) -> OntologyPopulationBindingBatchResult[] {
  client OpenAICompatibleProvider
  prompt #"
    {{ _.role("user") }}

    Gắn hậu quả pháp lý và applicability vào closed target conduct inventory để
    tạo các PenaltyRule có source support. Không kết luận vụ việc thực tế.

    Closed inventory
    - Không thêm, bỏ, tách, gộp hoặc đổi label/provision_path của target_conducts.
      Mỗi output phải trỏ đúng target_conduct_id; context không được tạo conduct mới.
    - Nếu inventory rỗng thì trả rules rỗng. Chỉ bỏ một locked conduct khi toàn
      reading window thực sự không cung cấp hậu quả pháp lý cho nó.

    Nguồn và identity
    - Chỉ dùng reading window của cùng item. Structural context chỉ xác định
      ancestry, subject, role và vehicle; không phải evidence. Chỉ trả evidence_id
      được cung cấp và không trộn các target window.
    - Sao chép nguyên văn locked conduct label, path và Target evidence. Giữ
      nguyên Unicode/ký hiệu locator pháp lý và không đồng nhất concept từ câu chữ.

    Bind consequence
    - Với từng conduct, tìm heading hậu quả chính còn hiệu lực, kể cả heading mở
      ở Preceding, rồi audit mọi catalog trực tiếp hoặc dẫn chiếu trong cùng Điều.
    - Giải mỗi dẫn chiếu thành tuple Điều/Khoản/Điểm đầy đủ. Chỉ bind khi tuple
      khớp chính xác locked provision; cùng Khoản, vị trí gần hoặc ký hiệu tương
      tự không đủ. Khi nhiều clause chồng lấn, dùng clause trực tiếp và cụ thể nhất.
    - Giữ mọi consequence cộng dồn được source hỗ trợ. Catalog chỉ bổ sung hậu
      quả cho conduct đã khóa; không tự tạo PenaltyRule. Mỗi consequence giữ
      evidence riêng và tuân thủ các field constraints trong output schema.
    - Hậu quả phụ thuộc điều kiện tạo variant riêng. Không trộn hậu quả thay thế,
      và mỗi variant có tối đa một MonetaryPenalty cùng một PointDeduction.

    Applicability và conditions
    - Qualifier cấu thành vẫn thuộc locked conduct. Ngoại lệ giới hạn phạm vi áp
      dụng nằm trong condition tree và giữ đúng AND/OR/NOT cùng source evidence.
    - Lặp condition/exception dùng chung cho mọi conduct bị chi phối; ngoại lệ
      cục bộ chỉ gắn đúng phần tử. Không suy ra exception chỉ từ một từ đơn lẻ.
    - Atomic không có operand; AllOf/AnyOf có ít nhất hai; Not có đúng một. Mọi
      condition có evidence trực tiếp, ID duy nhất, chỉ tham chiếu node cùng cây,
      không chu trình và mọi node phải đi được từ condition_root_id.
    - Chỉ điền subject, role, vehicle, validity và consequence values khi source
      hoặc structural context cho phép. Giá trị thiếu giữ null, không điền 0.

    Completion audit
    - Audit lần lượt mọi target_conduct_id, primary heading, catalog reference,
      variant, condition và evidence trước khi trả. Trả đúng một result cho mỗi
      target_chunk_id; không tạo LegalChange, concept equivalence hay occurrence.

    {% for item in items %}
    === target_chunk_id={{ item.target_chunk_id }} ===
    Locked target conducts:
    {% for conduct in item.target_conducts %}
    [{{ conduct.conduct_id }}]
    conduct_label_vi={{ conduct.conduct_label_vi }}
    provision_path={{ conduct.provision_path }}
    {% for evidence in conduct.evidence %}
    target_evidence={{ evidence.evidence_id }}
    {% endfor %}
    {% endfor %}
    {% for entry in item.window %}
    [{{ entry.role }}] chunk_id={{ entry.chunk_id }}
    {% if entry.structural_context %}
    Structural context (không phải source evidence):
    {{ entry.structural_context }}
    {% endif %}
    Evidence sources:
    {% for evidence in entry.evidence_sources %}
    [{{ evidence.evidence_id }}] chunk_id={{ evidence.chunk_id }}
    {{ evidence.text }}
    {% endfor %}
    ---
    {% endfor %}
    === end target_chunk_id={{ item.target_chunk_id }} ===
    {% endfor %}

    {{ ctx.output_format }}
  "#
}
