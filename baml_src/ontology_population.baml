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

    Lập closed inventory mọi hành vi vi phạm được Target trực tiếp mô tả; chưa
    gắn hậu quả và không kết luận vụ việc thực tế.

    - Chỉ dùng Target evidence; structural context chỉ hỗ trợ xác định cấu trúc.
      Không dùng kiến thức ngoài hoặc suy ra nội dung của điều khoản được dẫn chiếu.
    - Tách các hành vi có thể cấu thành vi phạm độc lập. Không tạo conduct từ định
      nghĩa, thủ tục, thẩm quyền, hậu quả hoặc dẫn chiếu không nhắc lại hành vi.
    - Không tách danh sách đối tượng hay các thành phần phải kết hợp mới thành một
      hành vi. Giữ đúng phủ định, qualifier và phạm vi ngữ pháp tiếng Việt.
    - provision_path giữ đủ cấp có trong nguồn theo `Chương > Mục > Điều > Khoản
      > Điểm`, bỏ cấp không tồn tại và không thêm tiêu đề.
    - Evidence phải thuộc Target. Chỉ trả rỗng sau khi đã đọc hết Target.

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

    Gắn hậu quả và applicability vào closed target conduct inventory để tạo các
    PenaltyRule có nguồn hỗ trợ. Không kết luận vụ việc thực tế.

    - Không thêm, bỏ, tách, gộp hay đổi label/path của target_conducts. Mỗi rule
      phải trỏ đúng target_conduct_id; inventory rỗng thì rules rỗng.
    - Chỉ dùng reading window của cùng item. Structural context hỗ trợ cấu trúc,
      subject, role và vehicle nhưng không thay thế evidence.
    - Gắn consequence khi nguồn hoặc dẫn chiếu cấu trúc xác định rõ conduct bị
      chi phối. Không dùng vị trí gần hay câu chữ tương tự để nối dữ liệu.
    - Giữ mọi consequence cộng dồn; tách variant khi hậu quả phụ thuộc điều kiện.
      Mỗi consequence giữ evidence riêng và tuân thủ kiểu dữ liệu output.
    - Applicability và ngoại lệ cục bộ nằm trong condition tree, giữ đúng
      AND/OR/NOT. Atomic không có operand; AllOf/AnyOf có ít nhất hai; Not có một.
      ID phải duy nhất, không chu trình và mọi node đi được từ root.
    - Chỉ điền giá trị được nguồn hỗ trợ; dữ liệu chưa biết giữ null. Không tạo
      LegalChange, concept equivalence hay occurrence.
    - Trả đúng một result cho mỗi target_chunk_id sau khi kiểm tra mọi locked conduct.

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
