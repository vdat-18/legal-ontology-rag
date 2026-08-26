enum ChunkWindowRole {
  Preceding
  PrecedingTarget
  Target
  Following
  FollowingTarget
  ArticleContext
}

class ChunkWindowEntry {
  chunk_id string
  role ChunkWindowRole @description(#"
    Target là chunk cần trích fact. PrecedingTarget và FollowingTarget
    là context cũng được xử lý độc lập như target ở request khác.
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
    - Mỗi fact thuộc về Target chứa nội dung pháp lý đặc trưng của quy
      tắc như hành vi, chủ thể, điều kiện, ngoại lệ hoặc hậu quả; tiêu
      đề, câu dẫn, đánh số hay dẫn chiếu chung không đủ để sở hữu fact.
    - Nếu Target bắt đầu bằng phần tiếp nối của quy tắc đã bắt đầu ở
      PrecedingTarget, không trích lại quy tắc đó; chỉ trích các quy
      tắc độc lập bắt đầu trong Target. PrecedingTarget đã sở hữu quy
      tắc và có Following context để hoàn chỉnh nó.
    - Nếu phần đầu cần thiết nằm ở Preceding thường, dùng context đó
      để khôi phục đầy đủ chủ thể, quy tắc chung, điều kiện và các phần
      không thể tách vì chunk này không được xử lý độc lập.
    - Chỉ dùng Following để khép lại câu, khoản hoặc danh sách đã có
      nội dung pháp lý đặc trưng trong Target. Nếu Target chỉ giới
      thiệu quy tắc hoặc danh sách còn nội dung đặc trưng bắt đầu ở
      Following thì không trích nội dung đó cho Target.
    - Không trích lại cùng một quy tắc dưới dạng gộp và dạng tách.
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

    Tự kiểm tra: Target sở hữu nội dung đặc trưng của fact; không lặp
    quy tắc đã bắt đầu ở PrecedingTarget; statement tự đủ nghĩa, không
    thiếu phần đầu, không trích sang Following và evidence phủ toàn bộ
    statement.

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
