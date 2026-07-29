class RewrittenQuery {
  query_vi string @description(#"
    Câu hỏi đã được diễn đạt lại bằng đúng thuật ngữ pháp lý chính thức, giữ nguyên ý nghĩa
    và đối tượng của câu hỏi gốc.
  "#)
}

function RewriteQueryToLegalTerms(query_vi: string) -> RewrittenQuery {
  client OpenAICompatibleProvider
  prompt #"
    {{ _.role("user") }}

    Bạn là chuyên gia thuật ngữ pháp luật giao thông đường bộ Việt Nam.

    Diễn đạt lại câu hỏi sau đây bằng đúng thuật ngữ pháp lý chính thức thường dùng trong
    nghị định, thông tư (ví dụ: "vượt đèn đỏ" -> "không chấp hành hiệu lệnh của đèn tín hiệu
    giao thông"). Giữ nguyên ý nghĩa và đối tượng của câu hỏi gốc.

    Câu hỏi: {{ query_vi }}

    {{ ctx.output_format }}
  "#
}
