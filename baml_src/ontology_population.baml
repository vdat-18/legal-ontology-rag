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

class OntologyEvidenceEntry {
  chunk_id string @description(#"Phải trùng với chunk_id trong reading window."#)
  quoted_span string @description(#"
    Đoạn trích nguyên văn liên tục, chính xác từng ký tự từ chunk tương ứng.
  "#)
}

class OntologyConditionEntry {
  condition_id string @description(#"Định danh cục bộ duy nhất trong một rule, ví dụ condition_1."#)
  kind OntologyConditionKind
  text_vi string @description(#"Nội dung điều kiện ngắn gọn bằng tiếng Việt."#)
  operand_ids string[] @description(#"
    ID của các condition con. Atomic phải rỗng; AllOf/AnyOf có ít nhất hai;
    Not có đúng một.
  "#)
  evidence OntologyEvidenceEntry[]
}

class OntologyConsequenceEntry {
  consequence_id string @description(#"Định danh cục bộ duy nhất trong một rule."#)
  kind OntologyConsequenceKind
  minimum_amount string? @description(#"Số thập phân không có dấu phân cách hàng nghìn; chỉ dùng cho MonetaryPenalty."#)
  maximum_amount string? @description(#"Số thập phân không có dấu phân cách hàng nghìn; chỉ dùng cho MonetaryPenalty."#)
  currency OntologyCurrency? @description(#"Đơn vị tiền tệ được source nêu; bắt buộc cho MonetaryPenalty."#)
  point_value int? @description(#"Số điểm nguyên dương; chỉ dùng cho PointDeduction."#)
  target_label_vi string? @description(#"Đối tượng chịu tác động, ví dụ giấy phép lái xe hoặc biển số."#)
  responsible_role_label_vi string? @description(#"Vai trò phải thực hiện RemedialMeasure nếu văn bản nêu rõ."#)
  evidence OntologyEvidenceEntry[]
}

class OntologyRuleEntry {
  conduct_label_vi string @description(#"Khái niệm hành vi được điều chỉnh, ngắn gọn bằng tiếng Việt."#)
  subject_kind OntologySubjectKind?
  role_labels_vi string[] @description(#"Vai trò áp dụng được văn bản nêu rõ; không suy diễn owner thành driver."#)
  vehicle_category_labels_vi string[] @description(#"Loại phương tiện được văn bản nêu rõ."#)
  condition_root_id string? @description(#"ID condition gốc, hoặc null khi không có điều kiện riêng."#)
  conditions OntologyConditionEntry[]
  consequences OntologyConsequenceEntry[] @description(#"Ít nhất một hậu quả pháp lý có nguồn hỗ trợ."#)
  valid_from string? @description(#"YYYY-MM-DD, chỉ điền khi window nêu hiệu lực riêng cho rule."#)
  valid_to string? @description(#"YYYY-MM-DD, null nghĩa là chưa biết chứ không phải vô hạn."#)
  provision_path string @description(#"Đường dẫn cấu trúc đầy đủ nhất được source hỗ trợ."#)
  evidence OntologyEvidenceEntry[] @description(#"Bằng chứng chung cho conduct, applicability và rule."#)
}

class OntologyPopulationBatchResult {
  target_chunk_id string @description(#"Phải trùng với target_chunk_id đầu vào."#)
  rules OntologyRuleEntry[]
}

function ExtractOntologyPopulationBatch(items: FactExtractionBatchItem[]) -> OntologyPopulationBatchResult[] {
  client OpenAICompatibleProvider
  prompt #"
    {{ _.role("user") }}

    Bạn xây dựng dữ liệu instance cho ontology pháp luật giao thông đường bộ
    từ văn bản quy phạm pháp luật Việt Nam. Hãy trích xuất cấu trúc quy tắc,
    không đưa ra kết luận áp dụng cho một vụ việc cụ thể.

    Quy tắc bắt buộc:
    - Chỉ dùng nội dung có bằng chứng trực tiếp trong reading window. Không
      bổ sung kiến thức sẵn có và không đoán phần văn bản bị thiếu.
    - Mỗi quoted_span phải là đoạn liên tục, nguyên văn, chính xác từng ký tự
      trong chunk được dẫn. Một assertion cần nhiều đoạn thì dùng nhiều entry.
    - Target là chunk chính; Adjacent chỉ cung cấp ngữ cảnh liền kề. Không trộn
      dữ liệu giữa các window.
    - Chỉ tạo rule khi có đủ conduct, applicability/provenance và ít nhất một
      legal consequence. Nếu chưa đủ, trả rules rỗng.
    - subject_kind chỉ phân loại NaturalPerson hoặc Organization khi source
      nói rõ. Vai trò như người điều khiển, chủ xe, người quản lý phải nằm ở
      role_labels_vi và không được tự đồng nhất với nhau.
    - conduct và vehicle category là các concept candidate theo đúng nghĩa
      source; không gộp chỉ vì câu chữ tương tự.
    - condition tree giữ nguyên AND/OR/NOT của source. condition_id chỉ là ID
      cục bộ duy nhất trong rule; operand_ids phải trỏ tới condition trong
      cùng rule và không tạo chu trình.
    - Phân loại hậu quả thành MonetaryPenalty, PointDeduction,
      AdditionalSanction hoặc RemedialMeasure chỉ khi source trực tiếp hỗ trợ.
      Nội dung biểu mẫu hoặc phụ lục không tự chứng minh RemedialMeasure.
    - MonetaryPenalty phải tách minimum_amount và maximum_amount thành số
      thập phân không có dấu phân cách và ghi currency. Không biến thiếu dữ
      liệu thành 0 hoặc tự đoán đơn vị tiền tệ.
    - PointDeduction phải có point_value và target_label_vi. AdditionalSanction
      và RemedialMeasure phải có target_label_vi; RemedialMeasure phải có
      responsible_role_label_vi khi văn bản xác định chủ thể thực hiện.
    - valid_to null nghĩa là chưa biết. Không diễn giải null thành vô hạn.
    - Không tạo ConductOccurrence, LegalChange, amendment link, replacement
      link hoặc cross-document rule identity từ lexical similarity.
    - Trả đúng một kết quả cho mỗi target_chunk_id. Không có rule là kết quả
      hợp lệ.

    {% for item in items %}
    === target_chunk_id={{ item.target_chunk_id }} ===
    {% for entry in item.window %}
    [{{ entry.role }}] chunk_id={{ entry.chunk_id }}
    {{ entry.text }}
    ---
    {% endfor %}
    === end target_chunk_id={{ item.target_chunk_id }} ===
    {% endfor %}

    {{ ctx.output_format }}
  "#
}
