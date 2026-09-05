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

enum OntologyValidityMode {
  DocumentDefault
  ExplicitInterval
  Unresolved
}

class OntologyEvidenceSource {
  evidence_id string @description(#"ID evidence trong input hiện tại."#)
  chunk_id string
  text string @description(#"Source nguyên văn; không sửa hoặc diễn giải lại."#)
}

class OntologyEvidenceReference {
  evidence_id string @description(#"Một evidence_id được cung cấp trong input hiện tại."#)
  start_char int? @description(#"Offset Unicode 0-based, inclusive; null để dùng toàn source."#)
  end_char int? @description(#"Offset Unicode 0-based, exclusive; null để dùng toàn source."#)
}

class OntologyConditionEntry {
  condition_id string @description(#"ID cục bộ duy nhất trong rule."#)
  kind OntologyConditionKind
  text_vi string @description(#"Nội dung điều kiện ngắn gọn bằng tiếng Việt."#)
  operand_ids string[] @description(#"ID condition con: Atomic rỗng; AllOf/AnyOf >=2; Not đúng 1."#)
  evidence OntologyEvidenceReference[] @description(#"Ít nhất một evidence trực tiếp cho expression."#)
}

class OntologyConsequenceEntry {
  consequence_id string @description(#"ID cục bộ duy nhất trong rule."#)
  kind OntologyConsequenceKind
  minimum_amount string? @description(#"Số thập phân không phân cách hàng nghìn; chỉ cho MonetaryPenalty."#)
  maximum_amount string? @description(#"Số thập phân không phân cách hàng nghìn; chỉ cho MonetaryPenalty."#)
  currency OntologyCurrency? @description(#"Bắt buộc cho MonetaryPenalty khi source nêu đơn vị."#)
  point_value int? @description(#"Số nguyên dương; chỉ cho PointDeduction."#)
  consequence_label_vi string? @description(#"Tên tác động; bắt buộc cho AdditionalSanction/RemedialMeasure."#)
  minimum_duration_months int? @description(#"Cận dưới theo tháng; chỉ cho AdditionalSanction."#)
  maximum_duration_months int? @description(#"Cận trên theo tháng; chỉ cho AdditionalSanction."#)
  target_label_vi string? @description(#"Đối tượng chịu tác động được source nêu."#)
  responsible_role_label_vi string? @description(#"Vai trò thực hiện RemedialMeasure; bắt buộc cho loại này."#)
  evidence OntologyEvidenceReference[]
}

class OntologyRuleEntry {
  conduct_label_vi string @description(#"
    Một hành vi độc lập; không gộp các hành vi thay thế. Giữ qualifier nội tại;
    ngoại lệ áp dụng thuộc condition tree.
  "#)
  subject_kind OntologySubjectKind?
  role_labels_vi string[] @description(#"Vai trò áp dụng được source nêu rõ."#)
  vehicle_category_labels_vi string[] @description(#"Loại phương tiện được văn bản nêu rõ."#)
  condition_root_id string? @description(#"ID condition gốc; null khi rule không có điều kiện/ngoại lệ riêng."#)
  conditions OntologyConditionEntry[] @description(#"Cây AND/OR/NOT giữ qualifier và ngoại lệ có evidence."#)
  consequences OntologyConsequenceEntry[] @description(#"Ít nhất một hậu quả pháp lý có evidence."#)
  validity_mode OntologyValidityMode @description(#"Nguồn hiệu lực: chung, riêng có ngày, hoặc chưa xác định."#)
  valid_from string? @description(#"YYYY-MM-DD, chỉ điền khi window nêu hiệu lực riêng cho rule."#)
  valid_to string? @description(#"YYYY-MM-DD; null là chưa biết, không có nghĩa vô hạn."#)
  provision_path string @description(#"Các cấp có thật theo Chương > Mục > Điều > Khoản > Điểm; không thêm tiêu đề."#)
  evidence OntologyEvidenceReference[] @description(#"Evidence chung cho conduct, applicability và rule."#)
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
  conduct_label_vi string @description(#"Một conduct nguyên tử, giữ đúng phủ định và qualifier tiếng Việt."#)
  provision_path string @description(#"Đường dẫn đầy đủ các cấp tiếng Việt của conduct."#)
  evidence OntologyEvidenceReference[] @description(#"Evidence chỉ từ Target."#)
}

class OntologyTargetConductBatchResult {
  target_chunk_id string
  conducts OntologyTargetConductEntry[] @description(#"Đầy đủ conduct nguyên tử trực tiếp trong Target; rỗng sau khi audit toàn Target."#)
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
  temporal_overrides DocumentTemporalOverride[]
}

class OntologyBoundRuleEntry {
  target_conduct_id string @description(#"Một conduct_id trong target_conducts."#)
  rule OntologyRuleEntry
}

class OntologyPopulationBindingBatchResult {
  target_chunk_id string
  rules OntologyBoundRuleEntry[] @description(#"Mọi rule variant đủ source support cho locked conduct inventory."#)
}

function ExtractOntologyTargetConductBatch(items: OntologyTargetConductInput[]) -> OntologyTargetConductBatchResult[] {
  client OpenAICompatibleProvider
  prompt #"
    {{ _.role("system") }}

    Bối cảnh: xây dựng ontology có bằng chứng chỉ cho quy định xử phạt vi phạm
    hành chính về giao thông đường bộ Việt Nam; không dùng cho vụ việc hoặc domain
    khác.

    Mục đích bước 1/6: khóa danh mục đóng các hành vi trực tiếp trong từng Target
    để context sau không tự thêm conduct; chưa gắn phạm vi hay hậu quả.

    - Conduct nguyên tử là hành vi cùng qualifier có thể làm căn cứ xử phạt. Tách
      hành vi thay thế, nhưng không tách phần phải kết hợp mới thành vi phạm.
    - Quyết định theo nghĩa pháp lý toàn câu, không theo dấu câu; dùng cùng độ chi
      tiết cho cấu trúc tương đương.
    - Không tạo conduct từ hậu quả, định nghĩa, thủ tục hoặc thẩm quyền. Trong câu
      hậu quả/dẫn chiếu, chỉ tạo khi Target nêu đủ hành vi; locator hoặc cụm đối
      tượng không đủ và không được suy lại nội dung được dẫn chiếu.
    - Chỉ dùng Target evidence; structural context chỉ định vị. Không dùng kiến
      thức ngoài. Giữ nguyên phủ định, qualifier và phạm vi tiếng Việt.
    - provision_path chỉ gồm cấp có thật theo `Chương > Mục > Điều > Khoản > Điểm`.
      Trả đúng một result mỗi target_chunk_id; conducts chỉ rỗng sau khi audit.

    {{ _.role("user") }}

    Các Target cần trích conduct:
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
    {{ _.role("system") }}

    Bối cảnh: xây dựng ontology có bằng chứng chỉ cho quy định xử phạt vi phạm
    hành chính về giao thông đường bộ Việt Nam; không dùng cho vụ việc hoặc domain
    khác.

    Mục đích bước 2/6: gắn phạm vi và hậu quả có source support vào conduct đã khóa.

    - Giữ nguyên target conducts, trỏ đúng target_conduct_id; inventory rỗng thì rules
      rỗng. Không gộp hoặc tạo conduct; chỉ dùng reading window cùng item.
    - Chỉ nối hậu quả khi source, locator hoặc cấu trúc xác định conduct bị chi
      phối; gần nhau, giống chữ hoặc cùng mức phạt không đủ. Giữ hậu quả cộng dồn;
      tách variant khi phạm vi hoặc hậu quả phụ thuộc điều kiện khác nhau.
    - Phân loại tiền phạt, trừ điểm, hình phạt bổ sung và khắc phục; biểu diễn phạm
      vi/ngoại lệ bằng cây AND/OR/NOT có evidence. Thiếu source support thì giữ null.
    - Chọn validity_mode theo temporal_overrides: DocumentDefault nếu không áp
      override (để hai ngày null); ExplicitInterval nếu provision có ngày riêng;
      Unresolved nếu căn cứ riêng chưa cho ngày (để hai ngày null). Đối chiếu đúng
      Điều/Khoản/Điểm, không lan sang provision khác hoặc suy từ kiến thức ngoài.
    - Không tạo LegalChange, concept relation hay occurrence. Trả một result mỗi
      target_chunk_id sau khi xử lý mọi locked conduct.

    {{ _.role("user") }}

    Các reading window cần bind:
    {% for item in items %}
    === target_chunk_id={{ item.target_chunk_id }} ===
    Temporal overrides:
    {% for override in item.temporal_overrides %}
    provisions={{ override.affected_provision_paths }}
    validity={{ override.valid_from }} -> {{ override.valid_to }}
    unresolved_basis={{ override.unresolved_basis_vi }}
    ---
    {% endfor %}
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
