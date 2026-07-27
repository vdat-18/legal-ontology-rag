class ExtractedEffectiveDate {
  effective_date string? @description(#"
    Ngày văn bản NÀY (không phải văn bản khác được nhắc tới) chính thức có hiệu lực thi hành,
    định dạng YYYY-MM-DD. Trả về null nếu đoạn trích không nêu rõ ngày hiệu lực chung của
    toàn văn bản (ví dụ chỉ có ngày hiệu lực riêng của một quy định/khoản cụ thể).
  "#)
}

function ExtractEffectiveDate(document_text: string) -> ExtractedEffectiveDate {
  client OpenAICompatibleProvider
  prompt #"
    {{ _.role("user") }}

    Đọc đoạn trích cuối một văn bản pháp luật Việt Nam (nghị định/thông tư/quyết định/luật)
    dưới đây và tìm ngày văn bản NÀY chính thức có hiệu lực thi hành — thường nằm trong điều
    "Hiệu lực thi hành" hoặc "Điều khoản thi hành". Bỏ qua ngày hiệu lực riêng của một quy
    định/khoản cụ thể khác không phải ngày hiệu lực chung của toàn văn bản.

    Đoạn trích:
    {{ document_text }}

    {{ ctx.output_format }}
  "#
}
