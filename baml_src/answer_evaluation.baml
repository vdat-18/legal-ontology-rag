class LegalAnswerToEvaluate {
  sample_id string
  question string
  answer string
  contexts string[]
  reference string?
}

class LegalAnswerEvaluation {
  sample_id string @description(#"Phải trùng với một sample_id trong danh sách đầu vào."#)
  faithfulness float @description(#"
    Tỷ lệ 0.0-1.0: phần trăm khẳng định trong câu trả lời có được hỗ trợ bởi ngữ cảnh.
    
    Phương pháp:
    1. Trích xuất TẤT CẢ các khẳng định (claims) từ câu trả lời
    2. Với mỗi khẳng định, kiểm tra xem ngữ cảnh có thông tin hỗ trợ không
    3. Tính score = (số khẳng định có bằng chứng) / (tổng số khẳng định)
    
    Ví dụ:
    - Câu trả lời có 4 khẳng định: "Phạt 800k-1tr", "tước bằng 1-3 tháng", "theo ND168/2024", "áp dụng từ 2025"
    - 3 khẳng định có evidence trong contexts
    - Score = 3/4 = 0.75
    
    Lưu ý: Một khẳng định được coi là có bằng chứng nếu có thể suy ra từ ngữ cảnh.
  "#)
  correctness int @description(#"
    Điểm 1-5 so sánh NỘI DUNG câu trả lời với đáp án chuẩn (không so cách diễn đạt).
    
    Chỉ đánh giá khi có đáp án chuẩn (reference). Nếu không có reference, đánh giá dựa trên tính chính xác pháp lý.
    
    Tiêu chí:
    1 = Sai hoàn toàn hoặc không liên quan đến đáp án chuẩn
    2 = Đúng một phần nhỏ (<30% nội dung đúng)
    3 = Đúng khoảng một nửa (30-70% nội dung đúng)
    4 = Đúng hầu hết, chỉ thừa/thiếu chi tiết nhỏ (70-95% đúng)
    5 = Đúng hoàn toàn, đầy đủ và chính xác (95-100% đúng)
    
    Lưu ý: Không trừ điểm vì khác wording, chỉ so sánh nội dung pháp lý.
  "#)
  answer_relevancy int @description(#"
    Điểm 1-5: câu trả lời có đúng trọng tâm câu hỏi không.
    1 = lạc đề hoàn toàn;
    2 = trả lời không đúng trọng điểm;
    3 = trả lời được một phần trọng tâm;
    4 = trả lời đúng trọng tâm nhưng thừa thông tin;
    5 = trả lời đúng trọng tâm, súc tích.
  "#)
  temporal_correctness bool? @description(#"
    true nếu câu trả lời áp dụng đúng bản luật đang có hiệu lực tại thời điểm
    liên quan tới câu hỏi, không dùng bản đã hết hiệu lực hoặc chưa có hiệu lực.
    null nếu câu trả lời từ chối trả lời hoặc không khẳng định bất kỳ quy định
    pháp luật nào (không có version luật nào để đánh giá tính temporal).
  "#)
}

function EvaluateAnswers(items: LegalAnswerToEvaluate[]) -> LegalAnswerEvaluation[] {
  client OpenAICompatibleProvider
  prompt #"
    {{ _.role("user") }}

    Bạn là giám khảo pháp lý, đánh giá câu trả lời của hệ thống hỏi-đáp luật
    giao thông đường bộ Việt Nam. Với MỖI mục trong danh sách dưới đây, chấm
    từng tiêu chí một cách độc lập và khách quan, chỉ dựa trên câu hỏi, câu
    trả lời, ngữ cảnh được truy hồi và đáp án chuẩn (nếu có). Đánh giá từng
    mục độc lập, không để mục này ảnh hưởng tới mục khác.

    Tiêu chí đánh giá:
    1. faithfulness (0.0-1.0): Tỷ lệ khẳng định trong câu trả lời có bằng chứng trong ngữ cảnh.
       - Trích ra từng khẳng định chính từ câu trả lời
       - Với mỗi khẳng định, kiểm tra xem ngữ cảnh có thông tin hỗ trợ không
       - Tính tỷ lệ = (số khẳng định có bằng chứng) / (tổng số khẳng định)
    
    2. correctness (1-5): So sánh nội dung với đáp án chuẩn
       - Chỉ so sánh nội dung, không so cách diễn đạt
       - Nếu không có đáp án chuẩn, đánh giá dựa trên tính chính xác pháp lý
    
    3. answer_relevancy (1-5): Đánh giá xem câu trả lời có đúng trọng tâm câu hỏi không
       - Không đánh giá đúng/sai, chỉ đánh giá relevance
    
    4. temporal_correctness (true/false/null): Áp dụng đúng luật đang hiệu lực không?
        - Kiểm tra năm hiệu lực, số hiệu văn bản
        - false nếu dùng luật đã hết hiệu lực hoặc chưa có hiệu lực
        - null nếu câu trả lời từ chối trả lời hoặc không trích dẫn/khẳng định bất kỳ quy định pháp luật nào

    Trả về đúng một kết quả cho mỗi mục, giữ nguyên sample_id để đối chiếu.

    {% for item in items %}
    ---
    sample_id: {{ item.sample_id }}
    Câu hỏi: {{ item.question }}
    Câu trả lời: {{ item.answer }}
    Ngữ cảnh được truy hồi:
    {% for context in item.contexts %}
    - {{ context }}
    {% endfor %}
    {% if item.reference %}Đáp án chuẩn: {{ item.reference }}{% endif %}
    ---
    {% endfor %}

    {{ ctx.output_format }}
  "#
}
