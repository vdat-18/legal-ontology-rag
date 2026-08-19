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
    Đường dẫn định danh theo thứ tự Chương > Mục > Điều > Khoản > Điểm, chỉ
    gồm số hoặc ký hiệu của cấp, không gồm tiêu đề. Bỏ cấp không tồn tại.
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

    Bạn xây dựng dữ liệu instance cho ontology pháp luật giao thông đường bộ
    từ văn bản quy phạm pháp luật Việt Nam. Hãy trích xuất quy tắc xử phạt
    gắn với hành vi được mô tả cụ thể, không đưa ra kết luận áp dụng cho một
    vụ việc cụ thể.

    Thực hiện nội bộ theo đúng thứ tự sau trước khi tạo output:
    1. Đọc toàn bộ Target từ đầu đến cuối và kiểm kê mọi conduct cụ thể. Conduct
       ở đầu Target vẫn có thể thuộc heading mức phạt nằm trong Previous.
    2. Với từng conduct trong Target, truy vết tất cả consequence trực tiếp và
       consequence dẫn chiếu rõ ràng trong toàn reading window, kể cả Previous
       và Following.
    3. Tách conduct nguyên tử, dựng applicability/exception tree và lặp lại mọi
       consequence dùng chung cho từng rule bị chi phối.
    4. Tự kiểm tra coverage trước khi trả lời: không bỏ conduct chỉ vì heading
       hoặc consequence nằm ở Adjacent; không bỏ consequence chỉ vì nó dẫn
       chiếu tới provision thay vì lặp lại conduct. Không xuất checklist này.

    Quy tắc bắt buộc:
    - Chỉ dùng nội dung có bằng chứng trực tiếp trong reading window. Không
      bổ sung kiến thức sẵn có và không đoán phần văn bản bị thiếu.
    - Structural context do parser lấy từ metadata của source. Dùng nó để xác
      định Chương, Mục, Điều, provision_path, vai trò và loại phương tiện áp
      dụng. Không trích structural context làm quoted_span; evidence chỉ được
      lấy nguyên văn từ Source text của chunk.
    - Nếu structural context có nhiều VARIANT thì chunk đi qua ranh giới cấu
      trúc. Đối chiếu thứ tự và heading trong Source text để không gán nội dung
      thuộc Điều trước sang Điều sau hoặc ngược lại.
    - Mỗi quoted_span phải là đoạn liên tục, nguyên văn, chính xác từng ký tự
      trong chunk được dẫn. Một assertion cần nhiều đoạn thì dùng nhiều entry.
    - Target là chunk chính; Adjacent cung cấp ngữ cảnh liền kề trong cùng
      window. Không trộn dữ liệu giữa các window.
    - Chỉ tạo rule cho conduct có quoted evidence nằm trong Target. Adjacent và
      structural context có thể cung cấp mức phạt, phạm vi áp dụng và ancestry,
      nhưng không được làm một conduct chỉ xuất hiện ở Adjacent trở thành rule
      của Target.
    - Previous và Following không phải nguồn tùy chọn khi chúng trực tiếp hoàn
      chỉnh rule của conduct trong Target. Nếu heading mức phạt nằm ở Previous
      và danh sách conduct tiếp tục ở đầu Target, phải gắn mức phạt đó cho mọi
      conduct thuộc danh sách. Không được chỉ xử lý danh sách có heading nằm
      ngay trong Target.
    - Nếu Following hoặc Previous nêu consequence bổ sung và dẫn chiếu rõ tới
      khoản, điểm chứa conduct trong Target, phải gắn consequence đó cho mọi
      rule được dẫn chiếu. Một rule phải giữ tất cả consequence được window hỗ
      trợ, không chỉ consequence xuất hiện gần conduct nhất.
    - PenaltyRule là quy tắc gắn một hành vi bị cấm hoặc nghĩa vụ cụ thể với
      hậu quả pháp lý áp dụng cho chủ thể vi phạm. Chỉ tạo rule khi window xác
      định được conduct cụ thể, applicability/provenance và ít nhất một legal
      consequence. Nếu chưa đủ, trả rules rỗng.
    - Không dùng các cụm chung như "vi phạm hành chính" hoặc "thực hiện hành
      vi vi phạm" làm conduct để biến một quy định chung thành PenaltyRule.
    - Trả rules rỗng cho danh mục chung về hình thức xử phạt hoặc biện pháp
      khắc phục; nguyên tắc xử lý nhiều vi phạm; định nghĩa; thẩm quyền, hồ sơ,
      tạm giữ hoặc thu hồi giấy phép; và nghĩa vụ thủ tục của người thi hành
      công vụ. Các nội dung này không phải PenaltyRule chỉ vì có nhắc tới một
      loại hậu quả pháp lý.
    - Một hậu quả đứng riêng chỉ là thành phần của rule khi window đồng thời
      định danh được hành vi cụ thể mà hậu quả đó áp dụng. Không tạo rule bằng
      cách gắn danh mục hậu quả chung cho mọi hành vi trong lĩnh vực.
    - subject_kind chỉ phân loại NaturalPerson hoặc Organization khi source
      nói rõ. Vai trò như người điều khiển, chủ xe, người quản lý phải nằm ở
      role_labels_vi và không được tự đồng nhất với nhau.
    - conduct và vehicle category là các concept candidate theo đúng nghĩa
      source; không gộp chỉ vì câu chữ tương tự.
    - Mỗi rule phải biểu diễn đúng một conduct có thể đối chiếu độc lập. Khi
      một điểm liệt kê nhiều hành vi thay thế cùng dẫn tới một hậu quả, tách
      từng hành vi thành rule riêng và lặp lại applicability, consequence và
      evidence dùng chung. Giữ đầy đủ qualifier như địa điểm, phương tiện,
      điều kiện "gây tai nạn" và ngoại lệ cho từng conduct.
    - Không tách các thành phần chỉ là danh sách đối tượng, phương tiện, cách
      diễn đạt tương đương, hoặc các yếu tố phải kết hợp mới tạo thành một
      hành vi có nghĩa. Việc tách là quyết định ngữ nghĩa, không chỉ dựa vào
      dấu phẩy, dấu chấm phẩy hay từ "hoặc".
    - Ví dụ: "Vượt xe trong trường hợp không được vượt; không có tín hiệu
      trước khi vượt; vượt bên phải trong trường hợp không được phép" phải
      thành ba rules. Không gộp thành một conduct "Vượt xe trái quy định".
    - Theo granularity của artifact này, "lạng lách, đánh võng" là hai conduct
      có thể đối chiếu độc lập và phải thành hai rules khi source dùng chúng
      như các hành vi thay thế cùng chịu một consequence.
    - condition tree giữ nguyên AND/OR/NOT của source. condition_id chỉ là ID
      cục bộ duy nhất trong rule; operand_ids phải trỏ tới condition trong
      cùng rule và không tạo chu trình.
    - Phân biệt bằng ngữ nghĩa giữa ngoại lệ giới hạn phạm vi áp dụng của rule
      và qualifier nội tại mô tả chính đối tượng hoặc hành vi. Không quyết định
      chỉ vì xuất hiện từ "trừ" hoặc một mẫu câu cụ thể.
    - Mọi ngoại lệ giới hạn phạm vi áp dụng phải được biểu diễn trong condition
      tree. Tạo một Atomic mô tả chính xác trường hợp hoặc nhóm quy định bị
      loại trừ, sau đó tạo Not trỏ tới Atomic đó; nếu còn điều kiện khác thì
      kết hợp bằng AllOf/AnyOf đúng logic của source. Giữ exact evidence của
      mệnh đề ngoại lệ trong các condition liên quan.
    - Không để ngoại lệ áp dụng chỉ nằm trong conduct_label_vi. Nhãn conduct
      mô tả hành vi tích cực có thể đối chiếu độc lập; condition tree mô tả
      khi rule được áp dụng hoặc không được áp dụng.
    - Khi tách một điểm thành nhiều conduct nguyên tử, lặp lại đầy đủ cùng cây
      ngoại lệ cho từng rule chịu ngoại lệ đó. condition_id là cục bộ nên tạo
      lại trong từng rule; không được giữ ngoại lệ cho chỉ một conduct.
    - Ngoại lệ chỉ gắn với một phần tử trong danh sách phải giữ đúng phạm vi của
      phần tử đó. Nếu các địa điểm hoặc hành vi trong cùng điểm có ngoại lệ cục
      bộ khác nhau, tách thành các rules riêng theo phạm vi; mỗi rule giữ ngoại
      lệ cục bộ của chính nó và mọi ngoại lệ chung áp dụng cho toàn điểm. Không
      gộp chúng thành một conduct rồi chỉ giữ ngoại lệ chung.
    - Ví dụ, nếu source nêu ba cách chuyển làn rồi loại trừ hành vi thuộc một
      điểm khác, tạo ba rules chuyển làn. Mỗi rule có một Atomic mô tả nhóm bị
      loại trừ, một Not trỏ tới Atomic đó, condition_root_id trỏ tới Not và
      condition evidence trích đúng mệnh đề loại trừ từ source.
    - Nếu phần loại trừ chỉ thu hẹp chính loại đối tượng cấu thành hành vi, ví
      dụ loại thiết bị không thuộc hành vi bị cấm, giữ nó như qualifier của
      conduct thay vì tự động biến thành ngoại lệ áp dụng. Việc phân loại này
      phải dựa trên ý nghĩa toàn câu và phạm vi quy phạm.
    - Phân loại hậu quả thành MonetaryPenalty, PointDeduction,
      AdditionalSanction hoặc RemedialMeasure chỉ khi source trực tiếp hỗ trợ.
      Nội dung biểu mẫu hoặc phụ lục không tự chứng minh RemedialMeasure.
    - MonetaryPenalty phải tách minimum_amount và maximum_amount thành số
      thập phân không có dấu phân cách và ghi currency. Không biến thiếu dữ
      liệu thành 0 hoặc tự đoán đơn vị tiền tệ.
    - PointDeduction phải có point_value và target_label_vi. AdditionalSanction
      và RemedialMeasure phải có target_label_vi. Mọi RemedialMeasure phải có
      responsible_role_label_vi được source hoặc ngữ cảnh cấu trúc trong
      window xác định; nếu không xác định được thì không xuất candidate đó.
    - valid_to null nghĩa là chưa biết. Không diễn giải null thành vô hạn.
    - provision_path dùng đúng format `Chương II > Mục 1 > Điều 6 > Khoản 5 >
      Điểm a`. Viết hoa đúng tên cấp, dùng separator ` > `, không thêm tiêu đề
      Chương/Mục/Điều, và bỏ cấp không có trong source.
    - Không tạo ConductOccurrence, LegalChange, amendment link, replacement
      link hoặc cross-document rule identity từ lexical similarity.
    - Trả đúng một kết quả cho mỗi target_chunk_id. Rules rỗng chỉ hợp lệ sau
      khi đã kiểm kê toàn bộ Target và xác nhận không conduct nào trong Target
      có đủ applicability cùng consequence trực tiếp hoặc dẫn chiếu trong
      reading window.

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
