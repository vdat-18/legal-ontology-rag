class FollowUpQuestions {
  questions string[] @description(#"
    Đúng 3 câu hỏi tiếp theo, mỗi câu tối đa 12 từ, viết như người dân tự hỏi.
  "#)
}

function SuggestFollowUpQuestions(query_vi: string, answer_vi: string) -> FollowUpQuestions {
  client OpenAICompatibleProvider
  prompt #"
    {{ _.role("user") }}

    Bạn đang hỗ trợ một người dân vừa được tư vấn về luật giao thông Việt Nam.

    Đề xuất ĐÚNG 3 câu hỏi tiếp theo mà người đó nhiều khả năng muốn hỏi.

    Yêu cầu:
    - Bám vào nội dung câu trả lời vừa rồi, đào sâu thêm một khía cạnh cụ thể:
      mức phạt cho phương tiện/đối tượng khác, hình phạt bổ sung (trừ điểm, tước
      giấy phép lái xe, tạm giữ phương tiện), tình tiết tăng nặng, hoặc thủ tục
      nộp phạt và khiếu nại.
    - Mỗi câu tối đa 12 từ, viết tự nhiên như người dân nói.
    - Ba câu phải khác nhau rõ rệt, không diễn đạt lại cùng một ý.
    - Không hỏi lại điều mà câu trả lời đã nêu rõ.
    - Chỉ hỏi về luật giao thông đường bộ Việt Nam.

    Ví dụ:
    Câu hỏi: "Ô tô vượt đèn đỏ phạt bao nhiêu?"
    Trả lời: "Phạt từ 18.000.000 đến 20.000.000 đồng theo Nghị định 168/2024/NĐ-CP."
    Gợi ý: ["Xe máy vượt đèn đỏ phạt bao nhiêu?", "Vượt đèn đỏ bị trừ mấy điểm bằng lái?",
    "Vượt đèn đỏ khi nhường xe cứu thương có bị phạt không?"]

    Câu hỏi vừa rồi: {{ query_vi }}
    Câu trả lời vừa rồi: {{ answer_vi }}

    {{ ctx.output_format }}
  "#
}
