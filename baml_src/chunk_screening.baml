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
    Tên field được giữ để tương thích schema. true nếu target chứa nội dung
    pháp lý có thể trích thành fact về xử lý vi phạm giao thông đường bộ.
  "#)
  reason string @description(#"
    Giải thích ngắn gọn bằng tiếng Việt cho quyết định has_penalty_content.
  "#)
}

function ScreenChunks(chunks: ChunkScreeningItem[]) -> ChunkScreeningResult[] {
  client OpenAICompatibleProvider
  prompt #"
    {{ _.role("user") }}

    Nhiệm vụ: chọn các target đáng gửi sang bước trích xuất fact cho benchmark
    hỏi đáp về xử lý vi phạm giao thông đường bộ tại Việt Nam. Xử lý độc lập
    từng chunk, trả đúng một kết quả và giữ nguyên chunk_id.

    has_penalty_content=true khi Target chứa ít nhất một nội dung có thể làm
    căn cứ trả lời câu hỏi trong domain, gồm: hành vi vi phạm hoặc nghĩa vụ;
    chủ thể/phương tiện và điều kiện áp dụng; mức hay hình thức xử phạt; trừ
    điểm, tước quyền sử dụng; biện pháp khắc phục; ngoại lệ; thẩm quyền; trình
    tự, thời hạn hoặc thủ tục xử lý; quy định sửa đổi/chuyển tiếp ảnh hưởng
    trực tiếp đến việc áp dụng. Bao gồm quy định nằm trong bảng hoặc Target là
    mảnh trực tiếp của một quy định bị chia chunk.

    has_penalty_content=false khi Target chỉ là mục lục, tiêu đề, lời dẫn,
    căn cứ ban hành, chữ ký, thông tin hành chính, phạm vi chung, định nghĩa
    không quyết định cách xử lý vi phạm, hiệu lực chung của toàn văn bản, hoặc
    nội dung ngoài giao thông đường bộ.

    Chỉ dùng Target text để quyết định. Structural context chỉ giúp hiểu vị trí
    và phạm vi; không tự tạo nội dung pháp lý. Ưu tiên recall khi Target có một
    phần quy định rõ ràng dù phần còn lại nằm ở chunk liền kề. reason viết một
    câu ngắn, nêu loại nội dung tìm thấy hoặc lý do loại.

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
