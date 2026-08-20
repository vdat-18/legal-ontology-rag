class DocumentMetadata {
  document_number string @description(#"
    Số hiệu của chính văn bản đang đọc, ví dụ 168/2024/NĐ-CP hoặc 90/VBHN-VPQH.
  "#)
  valid_from string? @description(#"
    Ngày chính văn bản này bắt đầu có hiệu lực, định dạng YYYY-MM-DD. Null nếu
    các đoạn dưới đây không nêu.
  "#)
  valid_to string? @description(#"
    Ngày chính văn bản này hết hiệu lực, định dạng YYYY-MM-DD. Null nếu văn bản
    còn hiệu lực hoặc các đoạn dưới đây không nêu.
  "#)
}

function ResolveDocumentMetadata(opening_sections: string[], closing_sections: string[]) -> DocumentMetadata {
  client OpenAICompatibleProvider
  prompt #"
    {{ _.role("user") }}

    Nhiệm vụ: xác định metadata cấp văn bản để gắn thống nhất vào mọi legal
    fact được trích từ văn bản về xử lý vi phạm giao thông đường bộ này.

    Chỉ dùng các đoạn đầu/cuối được cung cấp, không dùng kiến thức bên ngoài.
    - document_number: số hiệu của chính văn bản đang đọc. Ưu tiên dòng số/ký
      hiệu ở phần đầu; không lấy số hiệu của văn bản được viện dẫn, sửa đổi,
      thay thế hoặc bãi bỏ.
    - valid_from: ngày toàn văn bản bắt đầu có hiệu lực. Không nhầm với ngày
      ký, ban hành, đăng công báo hoặc ngày áp dụng riêng của một điều khoản.
    - valid_to: ngày toàn văn bản hết hiệu lực chỉ khi nguồn nêu rõ. Việc một
      điều khoản hay văn bản khác bị bãi bỏ không làm văn bản hiện tại hết hiệu
      lực.

    Chuẩn hóa ngày thành YYYY-MM-DD. Với ngày hiệu lực không được nêu rõ, trả
    null và không suy đoán. Các dẫn chiếu ở phần cuối không được ghi đè số hiệu
    xác định từ phần đầu.

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
