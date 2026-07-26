class ChunkQualityResult {
  clarity float @description(#"
    0.0 đến 1.0. Đoạn văn bản có diễn đạt rõ ràng, không mơ hồ.
  "#)
  depth float @description(#"
    0.0 đến 1.0. Đoạn văn bản có đủ nội dung thực chất, không phải tiêu đề
    hoặc câu dẫn nhập trống rỗng.
  "#)
  structure float @description(#"
    0.0 đến 1.0. Đoạn văn bản có cấu trúc pháp lý nhận diện được, ví dụ điều,
    khoản, điểm, hoặc hàng của một bảng chế tài.
  "#)
  relevance float @description(#"
    0.0 đến 1.0. Đoạn văn bản liên quan tới xử phạt vi phạm giao thông đường
    bộ tại Việt Nam: hành vi, chế tài, thẩm quyền, thủ tục, hoặc đối tượng áp
    dụng.
  "#)
}

function EvaluateChunkQuality(chunk_text: string) -> ChunkQualityResult {
  client OpenAICompatibleProvider
  prompt #"
    {{ _.role("user") }}

    Chấm điểm đoạn văn bản luật dưới đây theo bốn tiêu chí: rõ ràng, đủ sâu,
    có cấu trúc, và liên quan tới xử phạt vi phạm giao thông đường bộ. Đây là
    bước lọc chunk chi phí thấp trước khi trích xuất fact pháp lý; mục tiêu là
    loại các đoạn không mang nội dung chế tài như mục lục, tiêu đề chương,
    hoặc đoạn dẫn nhập trống, không phải đánh giá văn phong.

    Đoạn văn bản:
    ---
    {{ chunk_text }}
    ---

    {{ ctx.output_format }}
  "#
}
