class DocumentMetadata {
  document_number string @description(#"
    Số hiệu của chính văn bản đang đọc.
  "#)
  valid_from string? @description(#"
    Ngày toàn văn bản bắt đầu có hiệu lực, dạng YYYY-MM-DD.
  "#)
  valid_to string? @description(#"
    Ngày toàn văn bản hết hiệu lực, dạng YYYY-MM-DD.
  "#)
}

function ResolveDocumentMetadata(opening_sections: string[], closing_sections: string[]) -> DocumentMetadata {
  client OpenAICompatibleProvider
  prompt #"
    {{ _.role("system") }}

    Bối cảnh: chuẩn bị nguồn sự thật cho benchmark RAG hỏi đáp về
    xử lý vi phạm giao thông đường bộ tại Việt Nam.

    Nhiệm vụ: lấy metadata cấp văn bản để gắn thống nhất vào mọi
    legal fact. Input là các đoạn đầu/cuối của cùng văn bản, theo
    thứ tự tài liệu. Chỉ dùng source text được cung cấp.

    - document_number: số hiệu của chính văn bản đang đọc, giữ
      nguyên theo nguồn. Xác định từ phần đầu; không lấy số của
      văn bản được viện dẫn, sửa đổi, thay thế hoặc bãi bỏ.
    - valid_from: ngày toàn văn bản bắt đầu có hiệu lực; không
      nhầm với ngày ký, ban hành, đăng công báo hoặc ngày áp dụng
      riêng của một điều khoản.
    - valid_to: ngày toàn văn bản hết hiệu lực, chỉ khi source
      text nêu rõ; không suy ra từ việc một điều khoản hay văn bản
      khác bị bãi bỏ.

    Chuẩn hóa ngày thành YYYY-MM-DD; trả null nếu không nêu rõ. Không
    dùng kiến thức bên ngoài, suy đoán hoặc để dẫn chiếu ở phần cuối
    ghi đè danh tính văn bản xác định từ phần đầu.

    {{ _.role("user") }}

    Phần đầu văn bản:
    {% for section in opening_sections %}
    ---
    {{ section }}
    {% endfor %}

    Phần cuối văn bản:
    {% for section in closing_sections %}
    ---
    {{ section }}
    {% endfor %}

    {{ ctx.output_format }}
  "#
}
