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
  Warning
  PointDeduction
  AdditionalSanction
  RemedialMeasure
  ReferencedPenalty
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
  evidence_id string
  chunk_id string
  text string @description(#"Source nguyên văn."#)
}

class OntologyEvidenceReference {
  evidence_id string
  start_char int? @description(#"Offset Unicode inclusive; null dùng toàn source."#)
  end_char int? @description(#"Offset Unicode exclusive; null dùng toàn source."#)
}

class OntologyConditionEntry {
  condition_id string
  kind OntologyConditionKind
  text_vi string @description(#"Nội dung điều kiện ngắn gọn bằng tiếng Việt."#)
  operand_ids string[] @description(#"ID condition con: Atomic rỗng; AllOf/AnyOf >=2; Not đúng 1."#)
  evidence OntologyEvidenceReference[]
}

class OntologyConsequenceEntry {
  consequence_id string @description(#"ID cục bộ duy nhất trong rule."#)
  kind OntologyConsequenceKind
  minimum_amount string? @description(#"Số thập phân; chỉ MonetaryPenalty."#)
  maximum_amount string? @description(#"Số thập phân; chỉ MonetaryPenalty."#)
  currency OntologyCurrency? @description(#"Đơn vị tiền source nêu."#)
  point_value int? @description(#"Số nguyên dương; chỉ cho PointDeduction."#)
  consequence_label_vi string?
  minimum_duration_months int? @description(#"Cận dưới tháng của AdditionalSanction."#)
  maximum_duration_months int? @description(#"Cận trên tháng của AdditionalSanction."#)
  target_label_vi string? @description(#"Đối tượng chịu tác động được source nêu."#)
  responsible_role_label_vi string? @description(#"Vai trò thực hiện RemedialMeasure; bắt buộc cho loại này."#)
  evidence OntologyEvidenceReference[]
}

class OntologyRuleEntry {
  conduct_label_vi string @description(#"Hành vi đã khóa; giữ qualifier nội tại."#)
  subject_kind OntologySubjectKind?
  role_labels_vi string[] @description(#"Vai trò áp dụng được source nêu rõ."#)
  vehicle_category_labels_vi string[] @description(#"Loại phương tiện được văn bản nêu rõ."#)
  condition_root_id string? @description(#"ID gốc; null khi không có điều kiện riêng."#)
  conditions OntologyConditionEntry[] @description(#"Cây AND/OR/NOT có evidence."#)
  consequences OntologyConsequenceEntry[] @description(#"Ít nhất một hậu quả pháp lý có evidence."#)
  validity_mode OntologyValidityMode @description(#"Nguồn hiệu lực."#)
  valid_from string? @description(#"YYYY-MM-DD khi có ngày riêng."#)
  valid_to string? @description(#"YYYY-MM-DD; null khi chưa biết."#)
  provision_path string @description(#"Chương > Mục > Điều > Khoản > Điểm có căn cứ."#)
  evidence OntologyEvidenceReference[] @description(#"Evidence chung của rule."#)
}

class OntologyPopulationBatchResult {
  target_chunk_id string @description(#"Phải trùng với target_chunk_id đầu vào."#)
  rules OntologyRuleEntry[]
}

class OntologyTargetConductInput {
  target_chunk_id string
  structural_context string
  locator_context string
  evidence_sources OntologyEvidenceSource[]
}

class OntologyTargetConductEntry {
  conduct_id string @description(#"ID cục bộ duy nhất trong Target."#)
  conduct_label_vi string
  provision_path string
  evidence OntologyEvidenceReference[]
}

class OntologyTargetConductBatchResult {
  target_chunk_id string
  conducts OntologyTargetConductEntry[]
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

    - Tách lựa chọn chỉ khi từng nhánh tự là hành vi độc lập. Giữ cụm phối hợp khi
      source xem toàn cụm là một hành vi qua tham chiếu số ít hoặc
      predicate/qualifier chung; không tách phần phải kết hợp mới thành vi phạm.
    - Quyết định theo nghĩa pháp lý toàn câu, không theo dấu câu; dùng cùng độ chi
      tiết cho cấu trúc tương đương.
    - Không tạo conduct từ định nghĩa, thủ tục, thẩm quyền, nguyên tắc xử lý,
      sửa/bãi bỏ hoặc điều khoản chỉ ánh xạ sang provision khác; trả rỗng dù nó
      nhắc/mô tả lại hành vi. Chỉ tạo khi Target trực tiếp quy định hoặc gắn xử
      phạt với hành vi đủ; locator hoặc cụm đối tượng không đủ.
    - Chỉ dùng Target evidence; structural context chỉ định vị. Không dùng kiến
      thức ngoài. Giữ nguyên phủ định, qualifier và phạm vi tiếng Việt.
    - Locator context chỉ khôi phục Khoản/Điểm, không cấp conduct/evidence.
      provision_path giữ mọi cấp có căn cứ theo
      `Chương > Mục > Điều > Khoản > Điểm`; không bỏ hoặc đoán cấp.
      Nếu Target mở đầu bằng Điểm rồi mới xuất hiện Khoản mới, Điểm mở đầu vẫn
      thuộc Khoản gần nhất trong locator context; mốc mới chỉ chi phối phần sau nó.
    - Trả đúng một result mỗi target_chunk_id; conducts chỉ rỗng sau khi audit.

    {{ _.role("user") }}

    Các Target cần trích conduct:
    {% for item in items %}
    === target_chunk_id={{ item.target_chunk_id }} ===
    Structural context (không phải source evidence):
    {{ item.structural_context }}
    Locator context (không phải evidence):
    {{ item.locator_context }}
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

    - Giữ nguyên target conducts: trả ít nhất một rule cho từng target_conduct_id,
      giữ đúng conduct_label_vi và provision_path. Inventory rỗng thì rules rỗng;
      không bỏ, gộp hoặc tạo conduct.
    - Audit toàn window cho từng conduct. Chỉ bind khi source/locator xác định nó
      bị chi phối; gần nhau hoặc giống chữ không đủ. Thu đủ mọi hậu quả cộng dồn,
      kể cả điều khoản liệt kê locator; không dừng sau tiền phạt hay bỏ trừ điểm,
      bổ sung hoặc khắc phục. Mỗi rule phải có hậu quả.
    - Khi source nêu phạm vi hoặc mức khác nhau cho cá nhân và tổ chức, tạo rule
      NaturalPerson và Organization riêng; không để subject_kind null, không dùng
      target_label_vi thay subject. Chỉ null khi source thật sự không xác định;
      tách thêm variant khi điều kiện/phạm vi/hậu quả khác nhau.
    - Phân loại pháp lý: Warning cho cảnh cáo; ReferencedPenalty cho dẫn chiếu
      mức phạt (target_label_vi là đích dẫn chiếu); AdditionalSanction chỉ khi
      source nêu hình thức xử phạt bổ sung. RemedialMeasure phải giữ đối tượng
      được khôi phục/sửa chữa trong target_label_vi và vai trò thực hiện nếu có.
      Điều kiện dùng cây AND/OR/NOT.
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
