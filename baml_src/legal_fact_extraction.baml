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
    chunk bổ sung trong cùng Điều để hoàn chỉnh phạm vi và dẫn chiếu pháp lý.
  "#)
  context_text string @description(#"
    Ngữ cảnh cấu trúc do parser lấy từ metadata tài liệu. Có thể dùng để hiểu
    Chương, Mục, Điều và phạm vi áp dụng, nhưng không phải source text để làm
    căn cứ cho fact.
  "#)
  text string
}

class FactExtractionBatchItem {
  target_chunk_id string
  window ChunkWindowEntry[]
}

class EvidenceEntry {
  chunk_id string @description(#"Phải trùng với một chunk_id trong window."#)
}

class LegalFact {
  statement string @description(#"
    Một mệnh đề pháp lý nguyên tử, đầy đủ và tự hiểu được, bao quát toàn bộ
    nội dung chính của fact bất kể đó là hành vi, chế tài, thẩm quyền, thủ tục,
    ngoại lệ hoặc nghĩa vụ.
  "#)
  conduct string? @description(#"Hành vi vi phạm nếu fact có nêu; null nếu không áp dụng."#)
  subject string? @description(#"Đối tượng hoặc loại phương tiện áp dụng nếu có; null nếu không nêu."#)
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

    Bạn trích xuất các fact pháp lý nguyên tử về xử lý vi phạm giao thông đường bộ
    từ từng reading window. Target là chunk cần trích xuất; Preceding và
    Following chỉ giúp hoàn chỉnh quy định bị chia chunk.

    Quy tắc bắt buộc:
    - Chỉ tạo fact cho quy định có căn cứ trực tiếp trong Target. Context có
      thể hoàn chỉnh statement, subject, conditions, hậu quả và provision_path,
      nhưng không được tạo fact chỉ xuất hiện ngoài Target.
    - statement là một mệnh đề pháp lý nguyên tử, đầy đủ và tự hiểu được. Nó
      phải giữ được nội dung kể cả khi quy định không phù hợp field chuyên biệt.
    - Tách riêng các quy tắc, hành vi hoặc đối tượng có điều kiện hay hậu quả khác nhau.
      Chỉ điền conduct và subject khi thật sự áp dụng; nếu không thì trả null.
    - Chỉ điền trường có căn cứ trực tiếp trong window; không dùng kiến thức
      bên ngoài, không suy đoán. Không đủ căn cứ cho fact hoàn chỉnh thì bỏ fact.
    - Mọi nội dung của fact phải được phủ bởi evidence từ đúng chunk_id trong
      window. Chỉ tham chiếu chunk thực sự làm căn cứ cho fact.
    - Structural context chỉ dùng để hiểu cấu trúc và provision_path, không
      được dùng làm evidence. Với nhiều VARIANT, đối chiếu Source text
      và thứ tự window để không trộn quy định qua ranh giới Điều.
    - Xử lý từng window độc lập. Trả đúng một kết quả cho mỗi target_chunk_id,
      giữ nguyên id; facts=[] là hợp lệ.

    Tự kiểm tra trước khi trả lời: statement có căn cứ trong Target; các field
    chuyên biệt nhất quán với statement; các chunk evidence đủ và đúng intent.

    Ví dụ ngắn:
    - Target nêu "không chấp hành hiệu lệnh của đèn tín hiệu" và window nêu
      mức phạt tương ứng -> statement bao quát hành vi và mức phạt, đồng thời
      điền các field conduct, subject và monetary_penalty phù hợp.
    - Target quy định thẩm quyền xử phạt -> statement giữ trọn quy định;
      conduct có thể null nếu không có hành vi cụ thể.
    - Target chỉ giải thích thuật ngữ, không nêu quy tắc xử phạt -> facts=[].

    Các reading window:

    {% for item in items %}
    === target_chunk_id={{ item.target_chunk_id }} ===
    {% for entry in item.window %}
    [{{ entry.role }}] chunk_id={{ entry.chunk_id }}
    {% if entry.context_text %}
    Structural context (không dùng làm evidence):
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
