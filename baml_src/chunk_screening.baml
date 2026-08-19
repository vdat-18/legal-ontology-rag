class ChunkScreeningItem {
  chunk_id string
  context_text string @description(#"
    Ngữ cảnh cấu trúc do parser tạo, dùng để hiểu Chương, Mục và Điều của
    target text. Không tự coi context này là nội dung chế tài.
  "#)
  text string
}

class ChunkScreeningResult {
  chunk_id string @description(#"Phải trùng với một chunk_id trong danh sách đầu vào."#)
  has_penalty_content bool @description(#"
    true nếu đoạn văn bản chứa quy định cụ thể về xử phạt vi phạm giao thông
    đường bộ tại Việt Nam (hành vi, mức phạt, thẩm quyền, thủ tục, hoặc đối
    tượng áp dụng), dạng điều khoản hoặc hàng của một bảng chế tài. false nếu
    đoạn văn bản chỉ là mục lục, tiêu đề chương, đoạn dẫn nhập, điều khoản thi
    hành, định nghĩa chung không kèm chế tài, hoặc nội dung ngoài phạm vi xử
    phạt giao thông đường bộ.
  "#)
  reason string @description(#"
    Giải thích ngắn gọn bằng tiếng Việt cho quyết định has_penalty_content.
  "#)
}

function ScreenChunks(chunks: ChunkScreeningItem[]) -> ChunkScreeningResult[] {
  client OpenAICompatibleProvider
  prompt #"
    {{ _.role("user") }}

    Bạn sàng lọc chunk của văn bản Việt Nam về xử phạt vi phạm giao thông
    đường bộ. Với mỗi chunk, trả về đúng một kết quả và giữ nguyên chunk_id.

    Đặt has_penalty_content=true khi Target text chứa ít nhất một quy định cụ
    thể phục vụ trích xuất chế tài: hành vi/nghĩa vụ, mức hoặc hình thức xử
    phạt, trừ điểm, biện pháp khắc phục, đối tượng áp dụng, thẩm quyền hoặc thủ
    tục xử phạt; bao gồm điều khoản và hàng bảng. Đặt false nếu chỉ là mục
    lục, heading, dẫn nhập, phạm vi điều chỉnh, điều khoản thi hành, hoặc định
    nghĩa chung không gắn với quy định xử phạt cụ thể.

    Quy tắc:
    - Đánh giá từng chunk độc lập; không lấy nội dung chunk khác để suy diễn.
    - Structural context chỉ xác định Chương/Mục/Điều và phạm vi của Target;
      không tự nó làm kết quả thành true.
    - Dựa vào nội dung Target, ưu tiên recall khi Target là một mảnh của quy
      định cụ thể dù mức phạt nằm ở chunk liền kề.
    - reason ngắn gọn bằng tiếng Việt và nêu căn cứ chính của quyết định.

    Ví dụ:
    - Target="b) Không chấp hành hiệu lệnh của đèn tín hiệu giao thông."
      trong Điều xử phạt -> true vì nêu một hành vi vi phạm cụ thể.
    - Target="Điều 1. Phạm vi điều chỉnh..." -> false vì chỉ mô tả phạm vi
      chung, chưa nêu quy định xử phạt cụ thể.

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
