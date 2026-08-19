enum ChunkWindowRole {
  Preceding
  Target
  Following
  ArticleContext
}

class ChunkWindowEntry {
  chunk_id string
  role ChunkWindowRole @description(#"
    Target là chunk chính đang được đọc để trích fact. Preceding và Following
    là chunk liền trước và liền sau theo thứ tự văn bản. ArticleContext là
    chunk khác trong cùng Điều, chỉ dùng khi quy tắc hoặc hậu quả nằm xa.
  "#)
  context_text string @description(#"
    Ngữ cảnh cấu trúc do parser lấy từ metadata tài liệu. Có thể dùng để hiểu
    Chương, Mục, Điều và phạm vi áp dụng, nhưng không phải source text để trích
    evidence.quoted_span.
  "#)
  text string
}

class FactExtractionBatchItem {
  target_chunk_id string
  window ChunkWindowEntry[]
}

class EvidenceEntry {
  chunk_id string @description(#"Phải trùng với một chunk_id trong window."#)
  quoted_span string @description(#"
    Trích nguyên văn, chính xác từng ký tự, từ text của chunk có chunk_id
    tương ứng. Không diễn giải lại, không tóm tắt.
  "#)
}

class LegalFact {
  conduct string @description(#"Hành vi vi phạm, diễn đạt ngắn gọn bằng tiếng Việt."#)
  subject string @description(#"Đối tượng hoặc loại phương tiện áp dụng."#)
  conditions string[] @description(#"Điều kiện áp dụng bổ sung, nếu có. Mảng rỗng nếu không có."#)
  monetary_penalty string?
  license_points_deducted string?
  additional_penalties string[] @description(#"Hình phạt bổ sung, ví dụ tước giấy phép. Mảng rỗng nếu không có."#)
  remedial_measures string[] @description(#"Biện pháp khắc phục hậu quả. Mảng rỗng nếu không có."#)
  valid_from string? @description(#"Định dạng YYYY-MM-DD. Chỉ điền khi window nêu ngày hiệu lực riêng của điều khoản này."#)
  valid_to string? @description(#"Định dạng YYYY-MM-DD. Chỉ điền khi window nêu ngày hết hiệu lực riêng của điều khoản này."#)
  provision_path string @description(#"Ví dụ: Điều 6 khoản 9 điểm b."#)
  evidence EvidenceEntry[] @description(#"Ít nhất một phần tử. Mọi nội dung của fact phải có căn cứ ở đây."#)
}

class FactExtractionBatchResult {
  target_chunk_id string @description(#"Phải trùng với một target_chunk_id trong danh sách đầu vào."#)
  facts LegalFact[]
}

function ExtractLegalFactBatch(items: FactExtractionBatchItem[]) -> FactExtractionBatchResult[] {
  client OpenAICompatibleProvider
  prompt #"
    {{ _.role("user") }}

    Bạn là chuyên gia pháp lý, trích xuất fact xử phạt vi phạm giao thông
    đường bộ có căn cứ xác thực từ văn bản quy phạm pháp luật Việt Nam.

    Đọc từng reading window dưới đây và trích xuất các fact pháp lý nguyên
    tử về xử phạt vi phạm giao thông đường bộ tại Việt Nam. Trong mỗi window,
    chunk vai trò Target là trọng tâm; các chunk khác chỉ là ngữ cảnh hỗ trợ.

    Nguyên tắc bắt buộc:
    - Chỉ trích xuất nội dung có bằng chứng trực tiếp trong các đoạn dưới đây.
      Không dùng kiến thức pháp luật sẵn có của bạn để bổ sung hoặc suy diễn.
    - Mỗi evidence.quoted_span phải là một đoạn trích nguyên văn, chính xác
      từng ký tự, từ đúng chunk có evidence.chunk_id tương ứng.
    - Mỗi quoted_span phải là một đoạn liên tục trong source text. Nếu fact
      cần hai đoạn cách nhau bởi nội dung khác, tạo hai evidence entry; không
      được ghép hai đoạn rồi bỏ phần nằm giữa.
    - Nếu một fact cần thông tin từ nhiều chunk trong window, liệt kê đủ các
      evidence từ tất cả chunk đó.
    - Nếu window không chứa đủ căn cứ để tạo một fact hoàn chỉnh nào, trả về
      danh sách rỗng thay vì đoán.
    - Một Target có thể sinh ra không, một, hoặc nhiều fact, ví dụ một hàng
      bảng liệt kê nhiều hành vi khác nhau.
    - Trả về đúng một kết quả cho mỗi target_chunk_id, giữ nguyên target_chunk_id.
      Mảng facts rỗng là kết quả hợp lệ.
    - Xử lý từng window độc lập. Không trộn nội dung hoặc evidence giữa các window.
    - Nếu structural context có nhiều VARIANT thì chunk đi qua ranh giới cấu
      trúc. Đối chiếu thứ tự và heading trong Source text để không gán nội dung
      thuộc Điều trước sang Điều sau hoặc ngược lại.
    Ví dụ:

    Window ví dụ 1:
    [Target] chunk_id=chunk_042
    Điều 6 khoản 9 điểm b: Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng
    đối với người điều khiển xe ô tô không chấp hành hiệu lệnh của đèn tín
    hiệu giao thông.
    ---
    Kết quả: một fact với conduct="Không chấp hành hiệu lệnh của đèn tín
    hiệu giao thông", monetary_penalty="4.000.000 đồng đến 6.000.000 đồng",
    evidence=[{chunk_id: "chunk_042", quoted_span: "Phạt tiền từ 4.000.000
    đồng đến 6.000.000 đồng đối với người điều khiển xe ô tô không chấp
    hành hiệu lệnh của đèn tín hiệu giao thông"}]. quoted_span trích nguyên
    văn, không diễn giải lại.

    Window ví dụ 2:
    [Target] chunk_id=chunk_003
    Điều 3. Giải thích từ ngữ
    Trong Nghị định này, các từ ngữ dưới đây được hiểu như sau: "Người điều
    khiển xe" là người đang lái, vận hành phương tiện tham gia giao thông
    đường bộ.
    ---
    Kết quả: danh sách rỗng — đây là điều khoản giải thích từ ngữ, không nêu
    hành vi vi phạm hay mức phạt cụ thể nào để trích xuất fact.

    Bây giờ đọc các window sau:

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
