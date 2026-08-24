class ChunkScreeningItem {
  chunk_id string
  context_text string @description(#"
    Ngữ cảnh cấu trúc do parser tạo; không phải source text.
  "#)
  text string
}

class ChunkScreeningResult {
  chunk_id string @description(#"Phải trùng với một chunk_id trong danh sách đầu vào."#)
  has_penalty_content bool @description(#"
    True nếu target có nội dung có thể trích thành legal fact trong domain.
  "#)
  reason string @description(#"Một câu tiếng Việt ngắn giải thích quyết định."#)
}

function ScreenChunks(chunks: ChunkScreeningItem[]) -> ChunkScreeningResult[] {
  client OpenAICompatibleProvider
  prompt #"
    {{ _.role("system") }}

    Bối cảnh: chuẩn bị nguồn sự thật cho benchmark RAG hỏi đáp về
    xử lý vi phạm giao thông đường bộ tại Việt Nam.

    Nhiệm vụ: sàng lọc độc lập từng target chunk trước khi trích
    legal fact. Domain gồm quy định trực tiếp về hành vi/nghĩa vụ;
    chủ thể, phương tiện, điều kiện/ngoại lệ; chế tài và khắc
    phục hậu quả; thẩm quyền, thủ tục; hiệu lực hoặc chuyển tiếp
    ảnh hưởng trực tiếp đến việc xử lý vi phạm.

    Đặt has_penalty_content=true khi Target text có nội dung có thể trích
    thành fact trong domain. Không cần quy định trọn vẹn; ưu tiên recall
    cho mảnh bị chia chunk hoặc nằm trong bảng. Ngược lại, đặt false;
    mục lục, tiêu đề, lời dẫn, căn cứ ban hành, chữ ký, thông tin
    hành chính hoặc nội dung ngoài domain tự chúng không đủ.

    Chỉ Target text làm căn cứ. Structural context chỉ giúp hiểu cấu trúc,
    không tự tạo nội dung pháp lý. Trả đúng một kết quả cho mỗi
    input, giữ nguyên chunk_id và viết reason bằng một câu ngắn.

    {{ _.role("user") }}

    Các chunk cần sàng lọc:
    {% for chunk in chunks %}
    ---
    chunk_id: {{ chunk.chunk_id }}
    {% if chunk.context_text %}
    Structural context:
    {{ chunk.context_text }}
    {% endif %}
    Target text:
    {{ chunk.text }}
    ---
    {% endfor %}

    {{ ctx.output_format }}
  "#
}
