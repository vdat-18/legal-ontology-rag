class LegalAnswerEvaluation {
  quality_of_information int @description(#"
    Điểm 1-5 về chất lượng thông tin (chính xác, liên quan, đầy đủ, đúng bản
    hiện hành). 1 = sai hoặc lạc đề; 3 = đúng nhưng thiếu; 5 = chính xác, đầy
    đủ và nêu đúng điều khoản đang hiệu lực.
  "#)
  understanding_and_reasoning int @description(#"
    Điểm 1-5 về mức độ hiểu câu hỏi và lập luận pháp lý. 1 = hiểu sai câu hỏi;
    5 = hiểu đúng và lập luận chặt chẽ từ quy định pháp luật tới kết luận.
  "#)
  expression_style int @description(#"
    Điểm 1-5 về độ rõ ràng và mạch lạc. 1 = khó hiểu, lộn xộn; 5 = rõ ràng,
    súc tích, trình bày tốt bằng tiếng Việt.
  "#)
  safety_and_harm int @description(#"
    Điểm 1-5 về an toàn. 1 = bịa điều khoản hoặc mức phạt có thể gây hại nếu
    làm theo; 5 = không bịa và nêu rõ giới hạn của điều khẳng định được.
  "#)
  trust_and_confidence int @description(#"
    Điểm 1-5 về độ tin cậy. 1 = không có căn cứ hay trích dẫn; 5 = có căn cứ
    đầy đủ, kiểm chứng được, trích dẫn đúng điều khoản.
  "#)
  faithfulness int @description(#"
    Điểm 1-5: mọi khẳng định trong câu trả lời có được chống đỡ bởi ngữ cảnh
    được truy hồi không. 1 = bịa hoàn toàn; 5 = hoàn toàn dựa trên ngữ cảnh.
  "#)
  answer_relevancy int @description(#"
    Điểm 1-5: câu trả lời có đúng trọng tâm câu hỏi không. 1 = lạc đề; 5 =
    trả lời đúng trọng tâm câu hỏi.
  "#)
  temporal_correctness bool @description(#"
    true nếu câu trả lời áp dụng đúng bản luật đang có hiệu lực tại thời điểm
    liên quan tới câu hỏi, không dùng bản đã hết hiệu lực hoặc chưa có hiệu lực.
  "#)
  factual_correctness int? @description(#"
    Điểm 1-5 so khớp nội dung với đáp án chuẩn. Để null nếu không có đáp án chuẩn.
  "#)
}

function EvaluateLegalAnswer(
  question: string,
  answer: string,
  contexts: string[],
  reference: string?
) -> LegalAnswerEvaluation {
  client OpenAICompatibleProvider
  prompt #"
    {{ _.role("user") }}

    Bạn là giám khảo pháp lý, đánh giá một câu trả lời của hệ thống hỏi-đáp
    luật giao thông đường bộ Việt Nam. Chấm từng tiêu chí một cách độc lập và
    khách quan, chỉ dựa trên câu hỏi, câu trả lời, ngữ cảnh được truy hồi và
    đáp án chuẩn (nếu có). Các tiêu chí thang 1-5: 1 là kém nhất, 5 là tốt
    nhất. temporal_correctness là true hoặc false. Nếu không có đáp án chuẩn,
    đặt factual_correctness = null.

    Câu hỏi:
    {{ question }}

    Câu trả lời cần đánh giá:
    {{ answer }}

    Ngữ cảnh được truy hồi:
    {% for context in contexts %}
    ---
    {{ context }}
    {% endfor %}

    {% if reference %}
    Đáp án chuẩn:
    {{ reference }}
    {% endif %}

    {{ ctx.output_format }}
  "#
}
