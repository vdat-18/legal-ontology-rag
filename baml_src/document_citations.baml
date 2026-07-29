class CitedDocuments {
  document_numbers string[] @description(#"
    Số hiệu của mọi văn bản quy phạm pháp luật được nhắc tới trong đoạn văn bản, chuẩn hoá theo
    đúng dạng số hiệu gốc (ví dụ: "168/2024/NĐ-CP", "36/2024/QH15"). Trả về mảng rỗng nếu đoạn
    văn bản không nhắc tới văn bản nào.
  "#)
}

function ExtractCitedDocuments(text: string) -> CitedDocuments {
  client OpenAICompatibleProvider
  prompt #"
    {{ _.role("user") }}

    Liệt kê số hiệu của mọi văn bản quy phạm pháp luật Việt Nam (nghị định, thông tư, luật,
    quyết định, văn bản hợp nhất) được nhắc tới trong đoạn văn bản dưới đây. Chỉ liệt kê số
    hiệu xuất hiện trong đoạn văn bản, không suy đoán hay bổ sung văn bản không được nhắc tới.

    Đoạn văn bản:
    {{ text }}

    {{ ctx.output_format }}
  "#
}
