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
    Đoạn căn cứ từ chunk tương ứng. Ưu tiên trích nguyên văn; có thể chuẩn hóa
    nhẹ cách xuống dòng hoặc diễn đạt nếu vẫn giữ đúng nội dung pháp lý.
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

    Bạn trích xuất các fact nguyên tử về xử phạt vi phạm giao thông đường bộ
    từ từng reading window. Target là chunk cần trích xuất; Preceding và
    Following chỉ giúp hoàn chỉnh quy định bị chia chunk.

    Quy tắc bắt buộc:
    - Chỉ tạo fact cho hành vi, nghĩa vụ hoặc quy tắc có căn cứ trực tiếp
      trong Target. Context có thể bổ sung subject, conditions, chế tài và
      provision_path, nhưng không được tạo fact chỉ xuất hiện ngoài Target.
    - Mỗi fact biểu diễn một hành vi/quy tắc nguyên tử với đúng đối tượng và
      điều kiện áp dụng. Tách riêng các hành vi hoặc đối tượng có chế tài khác nhau.
    - Chỉ điền trường có căn cứ trực tiếp trong window; không dùng kiến thức
      bên ngoài, không suy đoán. Không đủ căn cứ cho fact hoàn chỉnh thì bỏ fact.
    - Mọi nội dung của fact phải được phủ bởi evidence từ đúng chunk_id trong
      window. quoted_span cần thể hiện đúng căn cứ pháp lý; không bắt buộc
      khớp từng ký tự với Source text.
    - Structural context chỉ dùng để hiểu cấu trúc và provision_path, không
      được dùng làm quoted evidence. Với nhiều VARIANT, đối chiếu Source text
      và thứ tự window để không trộn quy định qua ranh giới Điều.
    - Xử lý từng window độc lập. Trả đúng một kết quả cho mỗi target_chunk_id,
      giữ nguyên id; facts=[] là hợp lệ.

    Tự kiểm tra trước khi trả lời: conduct có evidence trong Target; subject,
    conditions và mọi chế tài đúng với conduct; evidence đủ và đúng intent.

    Ví dụ ngắn:
    - Target nêu "không chấp hành hiệu lệnh của đèn tín hiệu" và window nêu
      mức phạt tương ứng -> tạo một fact với cả evidence hành vi và mức phạt.
    - Target chỉ giải thích thuật ngữ, không nêu quy tắc xử phạt -> facts=[].

    Các reading window:

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
