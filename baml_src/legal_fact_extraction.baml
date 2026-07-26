enum ChunkWindowRole {
  Target
  Adjacent
  CitedProvision
  EmbeddingCandidate
}

class ChunkWindowEntry {
  chunk_id string
  role ChunkWindowRole @description(#"
    Target là chunk chính đang được đọc để trích fact. Adjacent là chunk liền
    kề bị chunker cắt ra từ cùng một điều khoản. CitedProvision là chunk được
    Target dẫn chiếu tường minh, ví dụ "theo Điều X". EmbeddingCandidate là
    chunk gợi ý theo độ tương đồng ngữ nghĩa trong cùng văn bản, chỉ dùng khi
    thật sự liên quan tới Target.
  "#)
  provision_path string?
  text string
}

class EvidenceEntry {
  chunk_id string @description(#"Phải trùng với một chunk_id trong window."#)
  quoted_span string @description(#"
    Trích nguyên văn, chính xác từng ký tự, từ text của chunk có chunk_id
    tương ứng. Không diễn giải lại, không tóm tắt.
  "#)
}

class LegalFact {
  conduct string @description(#"Hành vi vi phạm, diễn đạt ngắn gọn bằng tiếng Việt."#)
  subject string @description(#"Đối tượng hoặc loại phương tiện áp dụng."#)
  conditions string[] @description(#"Điều kiện áp dụng bổ sung, nếu có. Mảng rỗng nếu không có."#)
  monetary_penalty string?
  license_points_deducted string?
  additional_penalties string[] @description(#"Hình phạt bổ sung, ví dụ tước giấy phép. Mảng rỗng nếu không có."#)
  remedial_measures string[] @description(#"Biện pháp khắc phục hậu quả. Mảng rỗng nếu không có."#)
  valid_from string? @description(#"Định dạng YYYY-MM-DD. Null nếu window không nêu ngày hiệu lực."#)
  valid_to string? @description(#"Định dạng YYYY-MM-DD. Null nếu còn hiệu lực hoặc window không nêu."#)
  document_number string @description(#"Số hiệu văn bản chứa quy định, ví dụ 168/2024/NĐ-CP."#)
  provision_path string @description(#"Ví dụ: Điều 6 khoản 9 điểm b."#)
  evidence EvidenceEntry[] @description(#"Ít nhất một phần tử. Mọi nội dung của fact phải có căn cứ ở đây."#)
}

function ExtractLegalFacts(window: ChunkWindowEntry[]) -> LegalFact[] {
  client OpenAICompatibleProvider
  prompt #"
    {{ _.role("user") }}

    Đọc các đoạn văn bản luật dưới đây và trích xuất các fact pháp lý nguyên
    tử về xử phạt vi phạm giao thông đường bộ tại Việt Nam. Chunk vai trò
    Target là trọng tâm; các chunk khác chỉ là ngữ cảnh hỗ trợ.

    Nguyên tắc bắt buộc:
    - Chỉ trích xuất nội dung có bằng chứng trực tiếp trong các đoạn dưới đây.
      Không dùng kiến thức pháp luật sẵn có của bạn để bổ sung hoặc suy diễn.
    - Mỗi evidence.quoted_span phải là một đoạn trích nguyên văn, chính xác
      từng ký tự, từ đúng chunk có evidence.chunk_id tương ứng.
    - Nếu một fact cần thông tin từ nhiều chunk trong window, liệt kê đủ các
      evidence từ tất cả chunk đó.
    - Nếu window không chứa đủ căn cứ để tạo một fact hoàn chỉnh nào, trả về
      danh sách rỗng thay vì đoán.
    - Một Target có thể sinh ra không, một, hoặc nhiều fact, ví dụ một hàng
      bảng liệt kê nhiều hành vi khác nhau.
    - Chỉ dùng chunk vai trò EmbeddingCandidate nếu nội dung của nó thực sự
      là căn cứ cho fact; nếu không liên quan thì bỏ qua, không ép dùng.

    Window:
    {% for entry in window %}
    [{{ entry.role }}] chunk_id={{ entry.chunk_id }}{% if entry.provision_path %} ({{ entry.provision_path }}){% endif %}
    {{ entry.text }}
    ---
    {% endfor %}

    {{ ctx.output_format }}
  "#
}
