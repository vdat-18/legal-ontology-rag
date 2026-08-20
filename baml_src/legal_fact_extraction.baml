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
    là chunk liền trước và liền sau theo thứ tự văn bản. ArticleContext dành
    cho pipeline khác cần context xa hơn trong cùng Điều.
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
    chủ thể, hành vi hoặc nghĩa vụ, điều kiện và hậu quả pháp lý có căn cứ.
  "#)
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

    Nhiệm vụ: trích xuất legal fact dùng làm nguồn sự thật để sinh và đánh giá
    câu hỏi về xử lý vi phạm giao thông đường bộ tại Việt Nam. Xử lý độc lập
    từng reading window. Target là chunk được trích; Preceding và Following chỉ
    hoàn chỉnh quy định bị chia chunk.

    Phạm vi fact gồm quy định trực tiếp về: hành vi vi phạm hoặc nghĩa vụ; chủ
    thể/phương tiện và điều kiện áp dụng; mức hay hình thức xử phạt; trừ điểm,
    tước quyền sử dụng; biện pháp khắc phục; ngoại lệ; thẩm quyền; trình tự,
    thời hạn, thủ tục; sửa đổi, chuyển tiếp hoặc áp dụng theo thời gian.

    Contract:
    - Chỉ tạo fact khi Target chứa một phần nội dung pháp lý riêng biệt của
      fact. Context được phép bổ sung phần bị lược hoặc cắt, nhưng không tạo
      fact có nội dung pháp lý đặc trưng chỉ xuất hiện ngoài Target.
    - statement là nguồn sự thật duy nhất: một mệnh đề pháp lý nguyên tử, tự
      hiểu được khi đứng riêng và giữ đủ chủ thể, hành vi/nghĩa vụ, điều kiện,
      ngoại lệ và hậu quả cần thiết theo nguồn.
    - Không tách chủ thể, hành vi, điều kiện và hậu quả vốn cùng tạo thành một
      quy tắc. Chỉ tách thành nhiều fact khi chúng là các quy tắc độc lập hoặc
      có phạm vi, điều kiện hay kết quả pháp lý khác nhau. Không gộp các quy
      tắc không liên quan.
    - Giữ chính xác ý nghĩa, mức, khoảng, đơn vị, điều kiện và ngoại lệ. Không
      thêm kiến thức, suy đoán nội dung bị thiếu hoặc diễn giải rộng hơn nguồn.
      Nếu chưa thể viết statement hoàn chỉnh có căn cứ thì không tạo fact.
    - evidence chỉ gồm chunk_id trong window có Source text thực sự hỗ trợ
      statement. Mọi phần của statement phải được phủ bởi ít nhất một evidence.
      Structural context chỉ giúp hiểu cấu trúc, không phải evidence.
    - Giữ nguyên target_chunk_id và trả đúng một kết quả cho mỗi input;
      facts=[] là hợp lệ.

    Trước khi trả lời, tự kiểm tra: fact có được neo vào Target, statement có
    tự đủ nghĩa, không trộn quy định, và evidence có đủ cho toàn bộ statement.

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
