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
    ID của các condition con. Atomic phải rỗng; AllOf/AnyOf có ít nhất một;
    Not có đúng một.
  "#)
  evidence OntologyEvidenceEntry[] @description(#"
    Atomic phải có evidence trực tiếp. AllOf, AnyOf và Not có thể để rỗng khi
    evidence đã nằm đầy đủ ở các operand bên dưới.
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
  evidence OntologyEvidenceEntry[]
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
  evidence OntologyEvidenceEntry[] @description(#"Bằng chứng chung cho conduct, applicability và rule."#)
}

class OntologyPopulationBatchResult {
  target_chunk_id string @description(#"Phải trùng với target_chunk_id đầu vào."#)
  rules OntologyRuleEntry[]
}

class OntologyTargetConductInput {
  target_chunk_id string
  structural_context string @description(#"Metadata cấu trúc của Target; không phải quoted evidence."#)
  target_text string
}

class OntologyTargetConductEntry {
  conduct_id string @description(#"ID cục bộ duy nhất trong Target."#)
  conduct_label_vi string @description(#"
    Một conduct nguyên tử được nêu cụ thể trong Target, giữ đúng phủ định và
    qualifier tiếng Việt chi phối nó.
  "#)
  provision_path string @description(#"Đường dẫn đầy đủ tên cấp tiếng Việt của conduct."#)
  evidence OntologyEvidenceEntry[] @description(#"Exact evidence chỉ từ Target."#)
}

class OntologyTargetConductBatchResult {
  target_chunk_id string
  conducts OntologyTargetConductEntry[]
}

class OntologyPopulationBindingItem {
  target_chunk_id string
  window ChunkWindowEntry[]
  target_conducts OntologyTargetConductEntry[]
}

class OntologyBoundRuleEntry {
  target_conduct_id string @description(#"Phải trùng conduct_id được cung cấp trong target_conducts."#)
  rule OntologyRuleEntry
}

class OntologyPopulationBindingBatchResult {
  target_chunk_id string
  rules OntologyBoundRuleEntry[]
}

function ExtractOntologyTargetConductBatch(items: OntologyTargetConductInput[]) -> OntologyTargetConductBatchResult[] {
  client OpenAICompatibleProvider
  prompt #"
    {{ _.role("user") }}

    Lập inventory conduct cho ontology pháp luật giao thông đường bộ Việt Nam.
    Đây là danh sách đóng dùng ở pha gắn hậu quả pháp lý sau đó.

    - Chỉ đọc Target text và structural context của chính item. Không dùng kiến
      thức ngoài, không suy ra nội dung của điều khoản được dẫn chiếu.
    - Chỉ tạo conduct khi Target trực tiếp mô tả hành vi hoặc nghĩa vụ cụ thể.
      Cụm chỉ nói "hành vi quy định tại" một provision khác không phải conduct,
      kể cả khi Target nêu hậu quả áp dụng cho provision đó.
    - Clause consequence chỉ xác định phạm vi bằng dẫn chiếu vẫn là catalog,
      không phải conduct source. Không dựng conduct từ tên hình phạt, đối tượng
      bị tịch thu/thu hồi/tước quyền, biện pháp khắc phục hoặc wording mô tả đối
      tượng của hậu quả. provision_path phải trỏ tới clause trực tiếp định nghĩa
      conduct, không trỏ tới clause catalog đang dẫn chiếu conduct đó.
    - Nếu Target trộn clause conduct trực tiếp với catalog dẫn chiếu, chỉ tạo
      entry cho clause conduct trực tiếp và bỏ qua phần catalog khi lập inventory.
    - Đọc Target đến hết và xử lý mọi heading Khoản/Điểm xuất hiện trong chunk.
    - Mỗi entry là một hành vi có thể độc lập cấu thành vi phạm. Tách lựa chọn
      hành vi độc lập; giữ cùng entry danh sách đối tượng, phương tiện, cách diễn
      đạt tương đương hoặc các thành phần phải kết hợp mới có nghĩa.
    - Giữ đúng phạm vi ngữ pháp tiếng Việt. Khi một từ phủ định hoặc qualifier
      chi phối nhiều động từ/lựa chọn, lặp lại ý nghĩa đó trong từng label sau
      khi tách; không biến hành động tuân thủ thành hành vi vi phạm.
    - `d` và `đ` là hai ký hiệu Điểm khác nhau. Không bỏ dấu hoặc đổi ký hiệu.
    - provision_path bắt buộc dùng tên cấp tiếng Việt, dạng `Chương <id> > Mục
      <id> > Điều <id> > Khoản <id> > Điểm <id>` và bỏ cấp không tồn tại.
    - Mỗi evidence phải có chunk_id bằng target_chunk_id và quoted_span liên
      tục, khớp chính xác Target text. Evidence phải trực tiếp chứa wording tạo
      conduct; một dẫn chiếu chung không đủ làm conduct evidence.
    - Trả đúng một result cho mỗi target_chunk_id; conducts rỗng nếu Target chỉ
      chứa consequence catalog, dẫn chiếu, thủ tục hoặc quy định chung. Trước
      khi trả, audit từng entry và loại entry không có direct conduct clause.

    {% for item in items %}
    === target_chunk_id={{ item.target_chunk_id }} ===
    Structural context (không dùng làm quoted evidence):
    {{ item.structural_context }}
    Target text:
    {{ item.target_text }}
    === end target_chunk_id={{ item.target_chunk_id }} ===
    {% endfor %}

    {{ ctx.output_format }}
  "#
}

function BindOntologyPopulationBatch(items: OntologyPopulationBindingItem[]) -> OntologyPopulationBindingBatchResult[] {
  client OpenAICompatibleProvider
  prompt #"
    {{ _.role("user") }}

    Mục tiêu
    Gắn hậu quả pháp lý vào inventory conduct đã được khóa cho từng Target để
    xây dựng PenaltyRule. Không kết luận cho vụ việc thực tế.

    Quy trình nội bộ
    1. Dùng target_conducts làm danh sách đóng. Không thêm, bỏ, đổi label hoặc
       đổi provision_path; mỗi output rule phải trỏ tới đúng target_conduct_id.
    2. Với mỗi conduct, xác định heading hậu quả chính đang chi phối, kể cả
       heading mở ở Preceding.
    3. Đọc toàn reading window, lập bản đồ mọi hậu quả trực tiếp hoặc dẫn chiếu
       tới provision đó và giữ riêng điều kiện của từng hậu quả.
    4. Tạo rule variant và sao chép đúng locked conduct, applicability,
       consequence, condition, exception và evidence dùng chung.
    5. Audit từng target_conduct_id: tạo đủ variant nếu có consequence, hoặc bỏ
       nếu reading window chưa đủ consequence; mọi heading, dẫn chiếu, điều kiện
       và ngoại lệ phải đúng phạm vi. Không xuất dữ liệu audit.

    Nguồn và evidence
    - Chỉ dùng reading window; không dùng kiến thức ngoài hoặc đoán phần thiếu.
      Không trộn dữ liệu giữa các target window.
    - target_conducts chứa toàn bộ conduct được phép. Preceding/Following hoàn chỉnh ngữ cảnh
      liền kề; ArticleContext hoàn chỉnh heading, scope và dẫn chiếu trong cùng
      Điều. Context chỉ cung cấp applicability, condition hoặc consequence,
      không được tạo conduct mới.
    - Structural context chỉ xác định ancestry, subject/role, vehicle và
      provision_path. Nếu có nhiều VARIANT, đối chiếu Source text và thứ tự cấu
      trúc để giữ đúng ranh giới. Không dùng structural context làm quoted_span.
    - Mỗi quoted_span phải liên tục và khớp chính xác từng ký tự trong Source
      text của chunk được dẫn; dùng nhiều evidence entry cho nhiều đoạn.
    - Dấu và ký hiệu tiếng Việt là một phần của định danh pháp lý. Giữ nguyên
      chính tả source: `d` và `đ` là hai Điểm khác nhau; không bỏ dấu, đổi chữ
      hoặc ghép các ký hiệu gần nhau khi xác định Điều/Khoản/Điểm.
    - Sao chép nguyên văn conduct_label_vi, provision_path và Target evidence từ
      target_conducts vào rule. Subject, role, vehicle,
      ngày hiệu lực và giá trị consequence phải được source hoặc structural
      context cho phép ở trên xác định; không tự đồng nhất vai trò hay concept
      chỉ vì câu chữ tương tự. valid_to null chỉ có nghĩa là chưa biết.

    PenaltyRule và consequence
    - Chỉ tạo PenaltyRule khi xác định đủ conduct cụ thể, applicability,
      provenance và ít nhất một consequence. Quy định chung, định nghĩa, nguyên
      tắc, thẩm quyền, thủ tục, hồ sơ, tạm giữ/thu hồi, danh mục consequence và
      nghĩa vụ của người thi hành công vụ không tự tạo PenaltyRule.
    - Nếu target_conducts rỗng thì trả rules rỗng. Không mở rộng một dẫn chiếu
      chung thành các conduct lấy từ context.
    - Với Target bắt đầu giữa danh sách, lần ngược theo thứ tự tới heading hậu
      quả chính gần nhất còn mở và gắn nó cho mọi conduct thuộc phạm vi.
      Preceding/Following là bắt buộc khi trực tiếp hoàn chỉnh rule.
    - Đọc toàn bộ các danh mục consequence trong ArticleContext. Giải nén mỗi
      dẫn chiếu thành tuple đầy đủ (Điều, Khoản, Điểm), kế thừa đúng cấp dùng
      chung và kiểm tra từng tuple bằng ký hiệu nguyên văn. Chỉ bind khi tuple
      khớp chính xác provision của rule; không suy từ vị trí gần, thứ tự chữ cái
      hoặc ký hiệu tương tự. Khi phạm vi chồng lấn, chọn clause trực tiếp và cụ
      thể nhất, không lấy consequence của provision khác.
    - Trước khi giữ một consequence, kiểm tra chính clause evidence của nó sau
      khi giải nén có chứa exact provision_path của locked conduct. Clause chỉ
      chứa cùng Khoản nhưng liệt kê Điểm khác không hỗ trợ consequence đó.
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
    - Inventory đã quyết định atomicity. Mỗi rule dùng nguyên một locked conduct;
      không gộp, tách hoặc diễn giải lại label ở pha này.
    - Phân biệt qualifier cấu thành conduct với ngoại lệ giới hạn applicability
      theo ý nghĩa và phạm vi quy phạm. conduct_label_vi giữ nguyên wording cấu
      thành cùng qualifier nội tại; ngoại lệ applicability nằm trong condition tree.
    - Giữ nguyên logic AND/OR/NOT. Biểu diễn ngoại lệ applicability bằng Atomic
      có source evidence và Not trỏ tới Atomic. AllOf/AnyOf có ít nhất một
      operand; Not có đúng một. Logical operator có thể để evidence rỗng khi
      evidence đã nằm đầy đủ ở operands. condition_id là cục bộ, operand_ids chỉ
      trỏ trong rule và không tạo chu trình.
    - Lặp đầy đủ cây ngoại lệ chung cho mọi atomic rule bị chi phối. Ngoại lệ cục
      bộ chỉ gắn phần tử thuộc phạm vi; tách rule nếu các hành vi, đối tượng hoặc
      địa điểm có scope ngoại lệ khác nhau. Phần loại trừ chỉ thu hẹp đối tượng
      cấu thành conduct vẫn là qualifier, không tự động là applicability exception.
    - provision_path bắt buộc dùng dạng `Chương <id> > Mục <id> > Điều <id> >
      Khoản <id> > Điểm <id>`, không được chỉ ghi chuỗi ID. Bỏ cấp không có trong
      source, giữ nguyên dấu của ký hiệu và không kèm tiêu đề.
    - Không tạo ConductOccurrence, LegalChange, liên kết sửa đổi/thay thế,
      equivalence hoặc cross-document identity.
    - Trả đúng một result cho mỗi target_chunk_id. Mỗi target_conduct_id có thể
      có nhiều rule variant nhưng không được trỏ tới conduct ngoài inventory.

    {% for item in items %}
    === target_chunk_id={{ item.target_chunk_id }} ===
    Locked target conducts:
    {% for conduct in item.target_conducts %}
    [{{ conduct.conduct_id }}]
    conduct_label_vi={{ conduct.conduct_label_vi }}
    provision_path={{ conduct.provision_path }}
    {% for evidence in conduct.evidence %}
    target_evidence chunk_id={{ evidence.chunk_id }}: {{ evidence.quoted_span }}
    {% endfor %}
    {% endfor %}
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
