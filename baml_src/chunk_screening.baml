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
    True nếu target có fact trực tiếp giúp xác định hoặc xử lý vi phạm trong domain.
  "#)
  has_document_metadata bool @description(#"
    True nếu target nêu rõ số hiệu, ngày có hiệu lực hoặc ngày hết hiệu lực của chính văn bản đang đọc.
  "#)
  reason string @description(#"Một câu tiếng Việt ngắn giải thích quyết định."#)
}

function ScreenChunks(chunks: ChunkScreeningItem[]) -> ChunkScreeningResult[] {
  client OpenAICompatibleProvider
  prompt #"
    {{ _.role("system") }}

    Bối cảnh: chuẩn bị nguồn sự thật cho benchmark RAG hỏi đáp về
    xử lý vi phạm giao thông đường bộ tại Việt Nam.

    Nhiệm vụ: sàng lọc độc lập từng target chunk trước khi trích legal
    fact. Chỉ giữ nội dung trực tiếp giúp trả lời: hành vi hoặc tình huống
    nào vi phạm, áp dụng cho ai/phương tiện nào, trong điều kiện hay ngoại
    lệ nào, và vi phạm được phát hiện, ngăn chặn hoặc xử lý ra sao.

    Đặt has_penalty_content=true khi Target text chứa ít nhất một fact về:
    hành vi/nghĩa vụ; chủ thể, phương tiện, điều kiện/ngoại lệ xác định
    phạm vi áp dụng; chế tài, hậu quả pháp lý hoặc khắc phục hậu quả; thẩm
    quyền, chứng cứ, biện pháp, trình tự/thời hạn/quyền/nghĩa vụ khi xử lý;
    hiệu lực hoặc chuyển tiếp làm thay đổi việc áp dụng các nội dung này.

    Đặt false nếu nội dung chỉ thuộc quản lý hoặc hoạt động giao thông nói
    chung mà không trực tiếp giúp xác định hay xử lý một vi phạm, như chính
    sách, tuyên truyền, tổ chức quản lý, dữ liệu/thống kê, hợp tác, tài
    chính/hỗ trợ hoặc thủ tục hành chính độc lập. Định nghĩa/phân loại chỉ
    được giữ khi cần xác định trực tiếp chủ thể, phương tiện, điều kiện hoặc
    phạm vi áp dụng của quy định vi phạm. Không giữ chỉ vì nội dung thuộc
    pháp luật giao thông đường bộ.

    Sau khi đã đúng domain, ưu tiên recall cho quy định chưa trọn vẹn do bị
    chia chunk hoặc nằm trong bảng. Mục lục, tiêu đề, lời dẫn, căn cứ ban
    hành, danh mục hoặc tên biểu mẫu, trường điền thông tin, dòng giao/nhận,
    ô ký tên và thông tin hành chính tự chúng không đủ. Biểu mẫu chỉ được giữ
    khi Target chứa nội dung pháp lý cụ thể phải ghi hoặc phải thực hiện,
    không giữ phần khung trống hay xác nhận thao tác đơn lẻ.

    Đánh giá has_document_metadata độc lập với has_penalty_content. Đặt true
    chỉ khi Target nêu rõ số hiệu của chính văn bản đang đọc, ngày toàn văn
    bản bắt đầu có hiệu lực hoặc ngày toàn văn bản hết hiệu lực. Không chọn
    ngày ký/ban hành, ngày áp dụng riêng, metadata của văn bản được dẫn chiếu
    hoặc thông tin chỉ nhận diện phụ lục/biểu mẫu.

    Chỉ Target text làm căn cứ. Structural context chỉ giúp hiểu cấu trúc,
    không tự tạo nội dung pháp lý. Trả đúng một kết quả cho mỗi input, giữ
    nguyên chunk_id. Reason phải là một câu ngắn nêu rõ liên hệ hoặc lý do
    không có liên hệ trực tiếp với việc xác định hay xử lý vi phạm.

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
