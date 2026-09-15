enum AdaptiveQueryRoute {
  Direct
  MultiHop
}

class AdaptiveQueryPlanOutput {
  route AdaptiveQueryRoute
  subqueries string[]
}

function PlanAdaptiveQuery(query_vi: string, max_subqueries: int) -> AdaptiveQueryPlanOutput {
  client OpenAICompatibleProvider
  prompt #"
    {{ _.role("user") }}

    Bạn là chuyên gia lập kế hoạch truy xuất văn bản pháp luật giao thông Việt Nam.

    Chỉ lập kế hoạch tìm kiếm, KHÔNG trả lời câu hỏi và KHÔNG tự suy đoán số điều,
    mức phạt hoặc căn cứ pháp luật không xuất hiện trong câu hỏi.

    Chọn route:
    - Direct: một truy vấn tìm kiếm là đủ để tìm căn cứ trả lời. Khi chọn Direct,
      subqueries bắt buộc là danh sách rỗng.
    - MultiHop: cần kết hợp ít nhất hai nhóm căn cứ hoặc sự kiện pháp lý độc lập.
      Khi chọn MultiHop, tạo từ 2 đến {{ max_subqueries }} subqueries.

    Mỗi subquery phải:
    - là một câu hỏi tìm kiếm độc lập bằng tiếng Việt;
    - giữ đúng ý định của câu hỏi gốc;
    - dùng thuật ngữ pháp lý chính thức nếu có thể;
    - chỉ tìm một phần căn cứ cần thiết;
    - không lặp lại nguyên văn toàn bộ câu hỏi gốc;
    - không chứa câu trả lời hoặc căn cứ do bạn tự đoán.

    Câu hỏi: {{ query_vi }}

    {{ ctx.output_format }}
  "#
}
