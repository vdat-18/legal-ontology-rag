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
  evidence OntologyEvidenceEntry[]
}

class OntologyRuleEntry {
  conduct_label_vi string @description(#"
    Một hành vi được điều chỉnh có thể đối chiếu độc lập. Không gộp nhiều
    hành vi thay thế trong cùng điểm thành một nhãn bao quát. Chỉ giữ nội
    dung tích cực cấu thành hành vi và qualifier nội tại; ngoại lệ giới hạn
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

    Mục tiêu
    Xây dựng instance ontology từ văn bản pháp luật giao thông đường bộ Việt
    Nam. Chỉ trích xuất PenaltyRule gắn conduct cụ thể với hậu quả pháp lý;
    không kết luận cho vụ việc thực tế.

    Quy trình nội bộ
    1. Đọc Target theo thứ tự đến hết, chia theo từng heading Khoản/Điểm và kiểm
       kê mọi conduct, lựa chọn hành vi, qualifier và ngoại lệ. Không dừng khi
       gặp heading mới trong cùng Target.
    2. Với mỗi conduct, xác định provision_path và heading hậu quả chính đang
       chi phối, kể cả heading mở ở Preceding.
    3. Đọc toàn reading window, lập bản đồ mọi hậu quả trực tiếp hoặc dẫn chiếu
       tới provision đó và giữ riêng điều kiện của từng hậu quả.
    4. Tách conduct nguyên tử và rule variant; sao chép đúng applicability,
       consequence, condition, exception và evidence dùng chung.
    5. Audit coverage theo từng heading trong Target: mỗi conduct cụ thể phải có
       rule hoặc được xác nhận không đủ điều kiện tạo rule; mọi heading chính,
       dẫn chiếu, điều kiện và ngoại lệ đã đúng phạm vi; các hậu quả loại trừ
       nhau không cùng variant. Không xuất dữ liệu kiểm kê hoặc audit.

    Nguồn và evidence
    - Chỉ dùng reading window; không dùng kiến thức ngoài hoặc đoán phần thiếu.
      Không trộn dữ liệu giữa các target window.
    - Target chứa conduct cần tạo rule. Preceding/Following hoàn chỉnh ngữ cảnh
      liền kề; ArticleContext hoàn chỉnh heading, scope và dẫn chiếu trong cùng
      Điều. Context không được biến conduct chỉ xuất hiện ngoài Target thành rule.
    - Structural context chỉ xác định ancestry, subject/role, vehicle và
      provision_path. Nếu có nhiều VARIANT, đối chiếu Source text và thứ tự cấu
      trúc để giữ đúng ranh giới. Không dùng structural context làm quoted_span.
    - Mỗi quoted_span phải liên tục và khớp chính xác từng ký tự trong Source
      text của chunk được dẫn; dùng nhiều evidence entry cho nhiều đoạn.
    - Dấu và ký hiệu tiếng Việt là một phần của định danh pháp lý. Giữ nguyên
      chính tả source: `d` và `đ` là hai Điểm khác nhau; không bỏ dấu, đổi chữ
      hoặc ghép các ký hiệu gần nhau khi xác định Điều/Khoản/Điểm.
    - Mọi conduct phải có evidence trong Target. Subject, role, vehicle, path,
      ngày hiệu lực và giá trị consequence phải được source hoặc structural
      context cho phép ở trên xác định; không tự đồng nhất vai trò hay concept
      chỉ vì câu chữ tương tự. valid_to null chỉ có nghĩa là chưa biết.

    PenaltyRule và consequence
    - Chỉ tạo PenaltyRule khi xác định đủ conduct cụ thể, applicability,
      provenance và ít nhất một consequence. Quy định chung, định nghĩa, nguyên
      tắc, thẩm quyền, thủ tục, hồ sơ, tạm giữ/thu hồi, danh mục consequence và
      nghĩa vụ của người thi hành công vụ không tự tạo PenaltyRule.
    - Target thuần consequence hoặc dẫn chiếu trả rules rỗng; consequence đó chỉ
      gắn vào rule của conduct provision tương ứng. Không dùng mô tả vi phạm
      chung làm conduct.
    - Với Target bắt đầu giữa danh sách, lần ngược theo thứ tự tới heading hậu
      quả chính gần nhất còn mở và gắn nó cho mọi conduct thuộc phạm vi.
      Preceding/Following là bắt buộc khi trực tiếp hoàn chỉnh rule.
    - Đọc toàn bộ các danh mục consequence trong ArticleContext. Giải nén mỗi
      dẫn chiếu thành tuple đầy đủ (Điều, Khoản, Điểm), kế thừa đúng cấp dùng
      chung và kiểm tra từng tuple bằng ký hiệu nguyên văn. Chỉ bind khi tuple
      khớp chính xác provision của rule; không suy từ vị trí gần, thứ tự chữ cái
      hoặc ký hiệu tương tự. Khi phạm vi chồng lấn, chọn clause trực tiếp và cụ
      thể nhất, không lấy consequence của provision khác.
    - Giữ mọi consequence được window hỗ trợ. PointDeduction,
      AdditionalSanction và RemedialMeasure không thay thế hậu quả chính. Các
      consequence không định lượng chỉ cùng tồn tại khi source quy định cộng dồn.
    - Consequence có điều kiện tạo variant riêng: giữ lại Target evidence của
      conduct, thêm condition/evidence tương ứng và chỉ giữ hậu quả của variant.
      Không chuyển hậu quả có điều kiện sang variant cơ bản hoặc giữ hậu quả cơ
      bản khi source quy định hậu quả thay thế.
    - Mỗi variant có tối đa một MonetaryPenalty và một PointDeduction. Giá trị
      khác nhau theo điều kiện phải nằm ở variant khác nhau.
    - Chỉ phân loại consequence khi source hỗ trợ. MonetaryPenalty cần min/max
      dạng số không dấu phân cách và currency, không điền 0 hay đoán đơn vị.
      PointDeduction cần point_value và target_label_vi. AdditionalSanction cần
      consequence_label_vi, target_label_vi và đủ hai cận tháng nếu source nêu
      khoảng. RemedialMeasure cần consequence_label_vi, target_label_vi và
      responsible_role_label_vi; không suy ra từ biểu mẫu hoặc phụ lục.

    Atomicity, conditions và output
    - Mỗi rule chứa đúng một conduct có thể đối chiếu độc lập. Tách các hành vi
      thay thế nếu từng hành vi riêng đã đủ cấu thành; không dùng nhãn bao quát.
      Giữ cùng conduct các qualifier, danh sách đối tượng/phương tiện, cách diễn
      đạt tương đương hoặc thành phần phải kết hợp mới có nghĩa. Quyết định theo
      ngữ nghĩa, không theo dấu câu hay từ khóa.
    - Khi từ phủ định hoặc qualifier tiếng Việt chi phối nhiều động từ/lựa chọn,
      giữ phạm vi chi phối đó trong từng conduct sau khi tách. Không được bỏ một
      lựa chọn chỉ vì từ phủ định hoặc chủ ngữ không được lặp lại trước nó.
    - Phân biệt qualifier cấu thành conduct với ngoại lệ giới hạn applicability
      theo ý nghĩa và phạm vi quy phạm. conduct_label_vi chỉ mô tả hành vi tích
      cực cùng qualifier nội tại; ngoại lệ applicability nằm trong condition tree.
    - Giữ nguyên logic AND/OR/NOT. Biểu diễn ngoại lệ applicability bằng Atomic
      có exact evidence và Not trỏ tới Atomic; kết hợp AllOf/AnyOf khi cần.
      condition_id là cục bộ, operand_ids chỉ trỏ trong rule và không tạo chu trình.
    - Lặp đầy đủ cây ngoại lệ chung cho mọi atomic rule bị chi phối. Ngoại lệ cục
      bộ chỉ gắn phần tử thuộc phạm vi; tách rule nếu các hành vi, đối tượng hoặc
      địa điểm có scope ngoại lệ khác nhau. Phần loại trừ chỉ thu hẹp đối tượng
      cấu thành conduct vẫn là qualifier, không tự động là applicability exception.
    - provision_path bắt buộc dùng dạng `Chương <id> > Mục <id> > Điều <id> >
      Khoản <id> > Điểm <id>`, không được chỉ ghi chuỗi ID. Bỏ cấp không có trong
      source, giữ nguyên dấu của ký hiệu và không kèm tiêu đề.
    - Không tạo ConductOccurrence, LegalChange, liên kết sửa đổi/thay thế,
      equivalence hoặc cross-document identity.
    - Trả đúng một result cho mỗi target_chunk_id. Chỉ trả rules rỗng sau khi
      audit toàn Target xác nhận không conduct nào đủ applicability và consequence
      trực tiếp hoặc dẫn chiếu trong reading window.

    {% for item in items %}
    === target_chunk_id={{ item.target_chunk_id }} ===
    {% for entry in item.window %}
    [{{ entry.role }}] chunk_id={{ entry.chunk_id }}
    {% if entry.context_text %}
    Structural context (không dùng làm quoted evidence):
    {{ entry.context_text }}
    {% endif %}
    Source text:
    {{ entry.text }}
    ---
    {% endfor %}
    === end target_chunk_id={{ item.target_chunk_id }} ===
    {% endfor %}

    {{ ctx.output_format }}
  "#
}
