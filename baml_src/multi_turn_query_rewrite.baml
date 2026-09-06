class MultiTurnRewrittenQuery {
  query_vi string @description(#"
    Câu hỏi độc lập, đầy đủ, diễn đạt bằng đúng thuật ngữ pháp lý chính thức. Người đọc câu này
    phải hiểu được mà không cần xem lại các lượt hội thoại trước.
  "#)
}

function RewriteMultiTurnQuery(query_vi: string, history: ConversationTurn[]) -> MultiTurnRewrittenQuery {
  client OpenAICompatibleProvider
  prompt #"
    {{ _.role("user") }}

    Bạn là chuyên gia thuật ngữ pháp luật giao thông đường bộ Việt Nam.

    Nhiệm vụ: viết lại câu hỏi cuối cùng thành MỘT câu hỏi độc lập, dùng đúng thuật ngữ pháp lý.

    Hai việc phải làm:
    1. Bổ sung ngữ cảnh còn thiếu từ hội thoại trước. Câu hỏi nối tiếp thường lược bỏ chủ thể
       hoặc hành vi ("còn xe máy thì sao?", "mức đó áp dụng cho ai?", "bị trừ mấy điểm?").
       Phải khôi phục đầy đủ hành vi và đối tượng đang được nói tới.
    2. Diễn đạt bằng thuật ngữ chính thức trong nghị định, thông tư (ví dụ: "vượt đèn đỏ" ->
       "không chấp hành hiệu lệnh của đèn tín hiệu giao thông").

    Giữ nguyên ý định của người hỏi. Nếu câu hỏi đã độc lập, chỉ cần chuẩn hoá thuật ngữ.
    Nếu câu hỏi chuyển hẳn sang chủ đề khác, bỏ qua hội thoại trước.

    Ví dụ:
    Hội thoại trước: "Ô tô vượt đèn đỏ phạt bao nhiêu?" -> "Phạt 18-20 triệu đồng."
    Câu hỏi cuối cùng: "còn xe máy thì sao?"
    Viết lại: "Xe mô tô, xe gắn máy không chấp hành hiệu lệnh của đèn tín hiệu giao thông bị
    phạt bao nhiêu tiền?"

    {% if history %}
    Hội thoại trước:
    {% for turn in history %}
    Người hỏi: {{ turn.question }}
    Trả lời: {{ turn.answer }}
    {% endfor %}
    {% endif %}

    Câu hỏi cuối cùng: {{ query_vi }}

    {{ ctx.output_format }}
  "#
}
