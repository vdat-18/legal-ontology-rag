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

    Bạn là chuyên gia pháp lý, xác định thông tin nhận dạng của một văn bản quy
    phạm pháp luật Việt Nam.

    Số hiệu văn bản nằm ở phần đầu, cạnh tên cơ quan ban hành. Ngày hiệu lực
    nằm ở điều khoản thi hành, gần cuối phần nội dung.

    Nguyên tắc bắt buộc:
    - Chỉ dùng thông tin có trong các đoạn dưới đây, không dùng kiến thức pháp
      luật sẵn có của bạn.
    - Một văn bản sửa đổi nhắc lại số hiệu của văn bản bị sửa đổi. Số hiệu cần
      trả về là của văn bản đang đọc, không phải của văn bản bị sửa đổi hay bị
      bãi bỏ.
    - Nếu một điều khoản riêng lẻ có ngày hiệu lực muộn hơn, vẫn trả về ngày
      hiệu lực của toàn văn bản.
    - Trả về null cho ngày nào không được nêu rõ, thay vì đoán.

    Ví dụ 1:
    Phần đầu: "CHÍNH PHỦ ... Số: 168/2024/NĐ-CP ... Hà Nội, ngày 26 tháng 12
    năm 2024"
    Phần cuối: "Điều 55. Hiệu lực thi hành. Nghị định này có hiệu lực thi hành
    từ ngày 01 tháng 01 năm 2025."
    Kết quả: document_number="168/2024/NĐ-CP", valid_from="2025-01-01",
    valid_to=null.

    Ví dụ 2:
    Phần đầu: "BỘ CÔNG AN ... Số: 13/2025/TT-BCA"
    Phần cuối: "Sửa đổi, bổ sung điểm b khoản 2 Điều 27 của Thông tư số
    73/2024/TT-BCA ..."
    Kết quả: document_number="13/2025/TT-BCA", không phải 73/2024/TT-BCA vì đó
    là văn bản bị sửa đổi. valid_from=null vì các đoạn đưa vào không nêu ngày
    hiệu lực.

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
