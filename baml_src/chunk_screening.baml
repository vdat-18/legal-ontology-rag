class ChunkScreeningItem {
  chunk_id string
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

    Bạn là chuyên gia pháp lý, có kinh nghiệm rà soát văn bản quy phạm pháp
    luật Việt Nam về xử phạt vi phạm giao thông đường bộ.

    Với MỖI đoạn văn bản luật dưới đây, xác định đoạn đó có chứa quy định cụ
    thể về xử phạt vi phạm giao thông đường bộ tại Việt Nam hay không. Đây là
    bước lọc chi phí thấp trước khi trích xuất fact pháp lý; mục tiêu là loại
    các đoạn không mang nội dung chế tài như mục lục, tiêu đề chương, hoặc
    đoạn dẫn nhập trống, không phải đánh giá văn phong. Trả về đúng một kết
    quả cho mỗi đoạn, giữ nguyên chunk_id để đối chiếu; đánh giá từng đoạn độc
    lập, không để nội dung đoạn này ảnh hưởng tới quyết định của đoạn khác.

    Ví dụ:

    Đoạn văn bản:
    ---
    Điều 6. Xử phạt người điều khiển xe ô tô vi phạm quy tắc giao thông đường
    bộ
    9. Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng đối với người điều khiển
    xe thực hiện một trong các hành vi vi phạm sau đây:
    b) Không chấp hành hiệu lệnh của đèn tín hiệu giao thông.
    ---
    Kết quả: has_penalty_content=true, reason="Nêu rõ hành vi vi phạm cụ thể
    (không chấp hành đèn tín hiệu) và mức phạt tiền tương ứng."

    Đoạn văn bản:
    ---
    MỤC LỤC
    Chương I. QUY ĐỊNH CHUNG .......... 1
    Chương II. XỬ PHẠT VI PHẠM HÀNH CHÍNH .......... 5
    Chương III. ĐIỀU KHOẢN THI HÀNH .......... 42
    ---
    Kết quả: has_penalty_content=false, reason="Đây là mục lục liệt kê tên
    chương, không chứa nội dung chế tài cụ thể."

    Đoạn văn bản:
    ---
    Điều 1. Phạm vi điều chỉnh
    Nghị định này quy định về xử phạt vi phạm hành chính trong lĩnh vực giao
    thông đường bộ, bao gồm: hành vi vi phạm hành chính; hình thức, mức xử
    phạt, biện pháp khắc phục hậu quả đối với từng hành vi vi phạm.
    ---
    Kết quả: has_penalty_content=false, reason="Đây là điều khoản nêu phạm vi
    điều chỉnh chung của nghị định, không quy định mức phạt cho một hành vi
    cụ thể."

    Bây giờ xác định danh sách đoạn văn bản sau:
    {% for chunk in chunks %}
    ---
    chunk_id: {{ chunk.chunk_id }}
    {{ chunk.text }}
    ---
    {% endfor %}

    {{ ctx.output_format }}
  "#
}
