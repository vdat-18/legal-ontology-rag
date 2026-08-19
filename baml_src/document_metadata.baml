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

    Xác định metadata của chính văn bản quy phạm pháp luật đang đọc.

    Quy tắc:
    - Chỉ dùng các đoạn đầu/cuối bên dưới; không dùng kiến thức bên ngoài.
    - document_number là số hiệu của văn bản đang đọc, thường ở phần đầu;
      không lấy số hiệu văn bản được dẫn chiếu, sửa đổi hoặc bãi bỏ.
    - valid_from/valid_to là hiệu lực của toàn văn bản, không phải ngày ban
      hành hoặc ngày hiệu lực riêng của một điều khoản.
    - Chuẩn hóa ngày thành YYYY-MM-DD. Không nêu rõ thì trả về null, không đoán.

    Ví dụ: phần đầu có "Số: 13/2025/TT-BCA", phần cuối dẫn "Thông tư số
    73/2024/TT-BCA" thì document_number="13/2025/TT-BCA".

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
