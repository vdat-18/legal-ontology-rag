enum ChunkWindowRole {
  Target
  Adjacent
  CitedProvision
}

class ChunkWindowEntry {
  chunk_id string
  role ChunkWindowRole @description(#"
    Target là chunk chính đang được đọc để trích fact. Adjacent là chunk liền
    kề bị chunker cắt ra từ cùng một điều khoản. CitedProvision là chunk được
    tra theo số điều mà Target nhắc tới, ví dụ "theo Điều X" — nhưng số điều
    này có thể trùng ngẫu nhiên với một luật hoặc nghị định khác, chỉ dùng làm
    căn cứ nếu nội dung thực sự cùng bối cảnh với Target.
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

    Bạn là chuyên gia pháp lý, trích xuất fact xử phạt vi phạm giao thông
    đường bộ có căn cứ xác thực từ văn bản quy phạm pháp luật Việt Nam.

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
    - Chỉ dùng chunk vai trò CitedProvision nếu nội dung thực sự cùng bối
      cảnh với Target (cùng nghị định hoặc văn bản đang xét); số điều có thể
      trùng ngẫu nhiên với một luật hoặc nghị định khác, nếu vậy thì bỏ qua.

    Ví dụ:

    Window ví dụ 1:
    [Target] chunk_id=chunk_042
    Điều 6 khoản 9 điểm b: Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng
    đối với người điều khiển xe ô tô không chấp hành hiệu lệnh của đèn tín
    hiệu giao thông.
    ---
    Kết quả: một fact với conduct="Không chấp hành hiệu lệnh của đèn tín
    hiệu giao thông", monetary_penalty="4.000.000 đồng đến 6.000.000 đồng",
    evidence=[{chunk_id: "chunk_042", quoted_span: "Phạt tiền từ 4.000.000
    đồng đến 6.000.000 đồng đối với người điều khiển xe ô tô không chấp
    hành hiệu lệnh của đèn tín hiệu giao thông"}]. quoted_span trích nguyên
    văn, không diễn giải lại.

    Window ví dụ 2:
    [Target] chunk_id=chunk_003
    Điều 3. Giải thích từ ngữ
    Trong Nghị định này, các từ ngữ dưới đây được hiểu như sau: "Người điều
    khiển xe" là người đang lái, vận hành phương tiện tham gia giao thông
    đường bộ.
    ---
    Kết quả: danh sách rỗng — đây là điều khoản giải thích từ ngữ, không nêu
    hành vi vi phạm hay mức phạt cụ thể nào để trích xuất fact.

    Bây giờ đọc window sau:

    Window:
    {% for entry in window %}
    [{{ entry.role }}] chunk_id={{ entry.chunk_id }}{% if entry.provision_path %} ({{ entry.provision_path }}){% endif %}
    {{ entry.text }}
    ---
    {% endfor %}

    {{ ctx.output_format }}
  "#
}
