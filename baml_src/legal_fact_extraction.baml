enum ChunkWindowRole {
  Preceding
  Target
  Following
  ArticleContext
}

class ChunkWindowEntry {
  chunk_id string
  role ChunkWindowRole @description(#"
    Target là chunk cần trích fact; các role khác là source context.
  "#)
  context_text string @description(#"
    Ngữ cảnh cấu trúc do parser tạo; không phải source text.
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
    Một mệnh đề pháp lý nguyên tử, tự hiểu được và đủ căn cứ.
  "#)
  evidence EvidenceEntry[] @description(#"Ít nhất một chunk trong window, phủ toàn bộ statement."#)
}

class FactExtractionBatchResult {
  target_chunk_id string @description(#"Phải trùng với một target_chunk_id trong danh sách đầu vào."#)
  facts LegalFact[]
}

function ExtractLegalFactBatch(items: FactExtractionBatchItem[]) -> FactExtractionBatchResult[] {
  client OpenAICompatibleProvider
  prompt #"
    {{ _.role("system") }}

    Bối cảnh: chuẩn bị nguồn sự thật cho benchmark RAG hỏi đáp về
    xử lý vi phạm giao thông đường bộ tại Việt Nam.

    Nhiệm vụ: trích legal fact độc lập từ mỗi reading window. Domain
    gồm quy định trực tiếp về hành vi/nghĩa vụ; chủ thể, phương tiện,
    điều kiện/ngoại lệ; chế tài, trừ điểm/tước quyền, khắc phục hậu
    quả; thẩm quyền, trình tự/thời hạn/thủ tục; hiệu lực, sửa đổi hoặc
    chuyển tiếp ảnh hưởng trực tiếp đến việc xử lý vi phạm.

    Reading window:
    - Target là chunk cần trích và phải chứa một phần source text trực
      tiếp cho mỗi fact.
    - Các chunk context chỉ hoàn chỉnh quy định bị chia, lược chủ thể
      hoặc tiếp nối; không tạo fact có nội dung pháp lý đặc trưng chỉ
      xuất hiện ngoài Target.
    - Structural context chỉ giúp hiểu cấu trúc, không phải source text.

    Output contract:
    - statement là nguồn sự thật duy nhất: một mệnh đề pháp lý
      nguyên tử, tự hiểu được và giữ đủ thành phần cần thiết để áp dụng.
    - Giữ các thành phần của cùng một quy tắc trong một fact. Chỉ
      tách khi các quy tắc độc lập hoặc khác phạm vi, điều kiện hay
      hậu quả; không gộp quy tắc không liên quan.
    - Giữ chính xác ý nghĩa, con số, khoảng, đơn vị, điều kiện và
      ngoại lệ. Không thêm kiến thức, suy đoán hoặc diễn giải rộng hơn
      source text; không tạo fact nếu chưa đủ căn cứ.
    - evidence gồm Target và các chunk trong window có Source text đã
      dùng; toàn bộ statement phải được phủ. Không dùng Structural
      context làm evidence.
    - Trả đúng một kết quả cho mỗi input, giữ nguyên target_chunk_id;
      facts=[] là hợp lệ.

    Tự kiểm tra: mỗi fact neo vào Target, statement tự đủ nghĩa, không
    trộn quy định và evidence phủ toàn bộ statement.

    {{ _.role("user") }}

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
