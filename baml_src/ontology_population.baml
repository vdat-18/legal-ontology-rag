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
  text string
}

class OntologyEvidenceReference {
  evidence_id string
  start_char int? @description(#"Offset Unicode inclusive; null dùng toàn source."#)
  end_char int? @description(#"Offset Unicode exclusive; null dùng toàn source."#)
}

class OntologyConditionEntry {
  condition_id string
  kind OntologyConditionKind
  text_vi string
  operand_ids string[] @description(#"Atomic rỗng; AllOf/AnyOf >=2; Not đúng 1."#)
  evidence OntologyEvidenceReference[]
}

class OntologyConsequenceEntry {
  consequence_id string
  kind OntologyConsequenceKind
  minimum_amount string? @description(#"Số thập phân; chỉ tiền phạt."#)
  maximum_amount string? @description(#"Số thập phân; chỉ tiền phạt."#)
  currency OntologyCurrency?
  point_value int? @description(#"Số nguyên dương; chỉ PointDeduction."#)
  consequence_label_vi string?
  minimum_duration_months int?
  maximum_duration_months int?
  target_label_vi string? @description(#"Đối tượng chịu tác động."#)
  responsible_role_label_vi string? @description(#"Vai trò thực hiện RemedialMeasure."#)
  evidence OntologyEvidenceReference[]
}

class OntologyRuleEntry {
  conduct_label_vi string @description(#"Hành vi đã khóa."#)
  subject_kind OntologySubjectKind?
  role_labels_vi string[]
  vehicle_category_labels_vi string[]
  condition_root_id string? @description(#"ID gốc; null khi không có điều kiện riêng."#)
  conditions OntologyConditionEntry[]
  consequences OntologyConsequenceEntry[]
  validity_mode OntologyValidityMode @description(#"Nguồn hiệu lực."#)
  valid_from string? @description(#"YYYY-MM-DD khi có ngày riêng."#)
  valid_to string? @description(#"YYYY-MM-DD; null khi chưa biết."#)
  provision_path string
  evidence OntologyEvidenceReference[]
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

    Bối cảnh: xây dựng ontology chỉ cho quy định xử phạt vi phạm
    hành chính về giao thông đường bộ Việt Nam.

    Mục đích bước 1/6: danh mục đóng conduct.

    Thứ tự ưu tiên bắt buộc: (1) bao phủ đủ Target từ đầu đến cuối; (2) giữ đúng
    chủ thể, hành động, đối tượng và điều kiện nguồn; (3) phân rã thành conduct lá;
    (4) gắn locator. Không được bỏ một nhánh nguồn để làm label ngắn hoặc dễ tách.

    - Cổng bắt buộc: Target trực tiếp (a) gắn conduct với cảnh cáo, tiền phạt, trừ
      điểm, bổ sung, khắc phục/dẫn chiếu mức phạt; hoặc (b) sửa/thay điều khoản xác
      định bằng danh sách conduct vi phạm giao thông. (b) không cần lặp hậu quả.
    - Loại kỷ luật, hình sự/dân sự, cưỡng chế/thi hành, thủ tục/thẩm quyền, nghĩa vụ
      đúng, quyền/nghĩa vụ chung và điều cấm bị phạt nơi khác. Không tự đảo nghĩa vụ
      tích cực thành vi phạm; "xử lý theo pháp luật" không đủ căn cứ.
    - Quét Target từ ký tự đầu đến cuối. Audit riêng đoạn mở đầu trước heading mới;
      nếu nó tiếp nối Điểm/Khoản/danh sách, Locator context được xác nhận tiêu đề xử
      phạt/cấu trúc chi phối. Không bỏ đoạn mở đầu vì phía sau có heading mới;
      conduct/evidence vẫn chỉ từ Target.
    - Audit từng Điểm/nhánh, không xuất audit: đánh dấu mọi hành động, không hành
      động, trạng thái vi phạm; mỗi đoạn mở đầu, Điểm và nhánh trực tiếp bị xử phạt
      phải có conduct tương ứng hoặc được loại theo cổng bằng căn cứ rõ ràng. Kiểm
      tra lại coverage sau khi phân rã; không bỏ sót khi Target chuyển Khoản/Điều.
    - Trước khi phân rã, xác định chủ thể chịu xử phạt từ câu dẫn mức phạt/cấu trúc
      chi phối. Mọi label phải giữ nguyên chủ thể đó và đủ loại phương tiện/phạm vi;
      không lấy công cụ, phần mềm, giấy tờ, phương tiện hoặc người thực hiện hành
      động phụ làm chủ thể nếu họ không phải đối tượng bị xử phạt.
    - Mỗi phương án tự đủ bị phạt là một conduct lá. Khôi phục phần dùng chung rồi
      tách: (1) mọi vị ngữ độc lập, kể cả động từ dùng như bổ nghĩa bị động;
      `S V1, V2 O` -> `S V1 O`, `S V2 O`, `O V1, V2 trái quy định` -> hai lá;
      (2) các bổ ngữ lựa chọn độc lập cùng một vị ngữ: `S V O1 hoặc O2` ->
      `S V O1`, `S V O2` nếu thay lần lượt O1/O2 đều tạo câu vi phạm tự đủ.
      `P không đúng A, không đúng B` cũng thành hai lá.
    - Giữ chung danh sách dấu phẩy sau một lần xuất hiện của vị ngữ khi source không
      lặp vị ngữ và không dùng liên từ lựa chọn để tạo nhánh: `V tại A, B`,
      `làm thay đổi A, B`, `P bằng A/B/C` là một lá. Danh sách có thể gồm thành phần,
      thuộc tính, vị trí, phương thức, phạm vi hoặc ngưỡng; không tách chỉ vì từng
      phần có thể ghép thành câu ngắn. Hai đối tượng pháp lý độc lập nối bằng `hoặc`
      vẫn phải tách.
    - Trước khi trả kết quả, đọc lại từng label theo chủ thể-vị ngữ-bổ ngữ. Chỉ tách
      khi có nhiều vị ngữ hoặc các nhánh lựa chọn độc lập trong source; không dùng
      khả năng viết lại thành câu ngắn làm căn cứ duy nhất. Cụm danh từ chứa nhiều
      động từ cùng nhận một qualifier pháp lý, `O V1, V2 trái quy định`, bắt buộc
      thành `O V1 trái quy định` và `O V2 trái quy định`; quy tắc này ưu tiên hơn
      quy tắc giữ danh sách dấu phẩy. Ví dụ tổng quát: `phương tiện chế tạo, cải tạo
      trái quy định` phải thành hai conduct cho `chế tạo` và `cải tạo`. Không được
      trả label vẫn chứa hai động từ cùng đứng trước `trái quy định`.
    - Với `S cho phép/chở/để T V1, V2 hoặc V3`, giữ S là chủ thể chịu xử phạt nhưng
      tách V1/V2/V3 nếu từng hành động của T tự tạo thành phương án bị cấm. Chỉ gộp
      các hành động hoặc khả năng khi pháp lý buộc chúng cùng cấu thành một yêu cầu.
      Áp dụng tương tự cho động từ lồng sau mục đích/khả năng: `S P O để/thực hiện
      việc V1, V2, V3` -> lặp toàn bộ khung `S P O để V1`, `... để V2`, `... để V3`
      khi từng V là một khả năng pháp lý độc lập. Động từ lồng vẫn là vị ngữ cần audit.
    - Chủ thể label là chủ thể chịu xử phạt trong mệnh đề chi phối, không phải người
      thực hiện hành động phụ. Với `S tự V hoặc chở/cho phép/để T V`, tạo `S tự V`
      và `S chở/cho phép/để T V`; không đổi nhánh thứ hai thành `T V`. Kiểm tra lại
      mọi label đều bắt đầu từ S, kể cả khi câu nguồn bắt đầu bằng thuộc tính của O.
    - Nhánh rút gọn phải tự đủ nghĩa: khôi phục cả chủ thể và vị ngữ chi phối bị
      lược, cùng phủ định, đối tượng, qualifier/phạm vi dùng chung. `S V O1; O2`
      mặc định tạo `S V O1`, `S V O2` nếu O2 không mở một vị ngữ chính mới; phải
      lặp lại V trong label, không đổi `S V O` thành `S O` hay hành động khác của S.
    - Bảo toàn toàn bộ điều kiện xác định đối tượng, kể cả phần trong ngoặc hoặc
      mệnh đề cuối áp dụng cho nhiều nhánh; ưu tiên đủ nghĩa hơn label ngắn. Với cụm
      danh từ pháp lý được lặp lại hoặc rút gọn, mỗi lá giữ đủ giấy tờ, loại xe, bên
      giữ/cấp và phạm vi. Không thêm động từ/hành động để nối chủ thể với hành vi;
      chỉ dùng nội dung được source trực tiếp hỗ trợ. Không rút ngắn phần trong ngoặc
      nếu việc bỏ nó làm thay đổi đối tượng hoặc điều kiện áp dụng của leaf. Phân loại
      chủ thể để bước 2 bind.
    - Loại định nghĩa, nguyên tắc và locator thuần túy. Dẫn chiếu phải trực tiếp xác
      định conduct và mức/điều khoản xử phạt. Đoạn chỉ thi hành, khắc phục hoặc tính
      hậu quả/số lợi từ conduct dẫn chiếu không tạo conduct mới.
    - Không dùng kiến thức ngoài. Với văn bản sửa đổi, provision_path là điều khoản
      được sửa/thay; Điều sửa đổi chỉ là provenance. provision_path chỉ gồm mã
      locator theo `Chương > Mục > Điều > Khoản > Điểm`, đúng dấu ` > `; không kèm
      tên/tiêu đề Chương, Mục, Điều và không đoán/lặp locator. Context không cấp
      evidence hay conduct.
    {{ _.role("user") }}

    {% for item in items %}
    === target_chunk_id={{ item.target_chunk_id }} ===
    Cấu trúc:
    {{ item.structural_context }}
    Locator context:
    {{ item.locator_context }}
    Evidence:
    {% for evidence in item.evidence_sources %}
    [{{ evidence.evidence_id }}] chunk_id={{ evidence.chunk_id }}
    {{ evidence.text }}
    {% endfor %}
    === end ===
    {% endfor %}

    {{ ctx.output_format }}
  "#
}

function BindOntologyPopulationBatch(items: OntologyPopulationBindingItem[]) -> OntologyPopulationBindingBatchResult[] {
  client OpenAICompatibleProvider
  prompt #"
    {{ _.role("system") }}

    Bối cảnh: xây dựng ontology có bằng chứng chỉ cho quy định xử phạt vi phạm
    hành chính về giao thông đường bộ Việt Nam.

    Mục đích bước 2/6: gắn phạm vi và hậu quả có source support vào conduct đã khóa.

    - Giữ nguyên target conducts: trả ít nhất một rule cho từng target_conduct_id,
      giữ đúng conduct_label_vi và provision_path. Inventory rỗng thì rules rỗng;
      không bỏ, gộp hoặc tạo conduct.
    - Audit toàn window cho từng conduct. Chỉ bind khi source hoặc locator xác định
      nó bị chi phối; gần nhau hay giống chữ không đủ. Với từng clause hậu quả,
      đối chiếu toàn bộ danh sách locator rồi thu đủ tiền phạt, cảnh cáo, trừ điểm,
      bổ sung, khắc phục và dẫn chiếu áp dụng cộng dồn. Không dừng sau hậu quả đầu;
      mỗi rule phải có hậu quả.
    - Suy ra subject_kind theo bản chất chủ thể được source nêu: vai trò chỉ một
      con người thực hiện được là NaturalPerson, dù không có từ "cá nhân"; vai trò
      chỉ một pháp nhân/tổ chức thực hiện được là Organization. Nếu source nêu mức
      hoặc phạm vi riêng, tạo đủ hai variant. Chỉ null khi chủ thể thật sự có thể
      thuộc nhiều loại mà source không phân biệt; không dùng target_label_vi thay
      subject. Tách thêm variant khi điều kiện, phạm vi hoặc hậu quả khác nhau.
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
    === end ===
    {% endfor %}

    {{ ctx.output_format }}
  "#
}
