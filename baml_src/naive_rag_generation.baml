class RetrievedChunkContext {
  chunk_id string
  text string
  source_name string
}

class GeneratedAnswer {
  answer_vi string @description(#"
    Câu trả lời tiếng Việt, dựa hoàn toàn vào các đoạn trích được cung cấp.
  "#)
}

function GenerateAnswer(
  query_vi: string,
  chunks: RetrievedChunkContext[]
) -> GeneratedAnswer {
  client OpenAICompatibleProvider
  prompt #"
    {{ _.role("user") }}

    Bạn là trợ lý pháp luật giao thông Việt Nam. Trả lời câu hỏi dựa HOÀN
    TOÀN vào các đoạn trích pháp luật được cung cấp dưới đây.

    Quy tắc:
    - Chỉ trả lời dựa trên thông tin trong các đoạn trích. Nếu không tìm
      thấy, nói rõ là không tìm thấy căn cứ.
    - Trích dẫn nguồn (tên văn bản) khi trả lời.
    - Nêu rõ mức phạt tiền, hình phạt bổ sung nếu có.
    - Trả lời bằng tiếng Việt, ngắn gọn, chính xác.

    Ví dụ:

    Đoạn trích: [chunk_1 — 168_2024_ND-CP.docx] "Điều 6 khoản 9 điểm b: Phạt
    tiền từ 4.000.000 đồng đến 6.000.000 đồng đối với người điều khiển xe ô
    tô không chấp hành hiệu lệnh của đèn tín hiệu giao thông."
    Câu hỏi: "Vượt đèn đỏ ô tô bị phạt bao nhiêu tiền?"
    Trả lời: "Theo Điều 6 khoản 9 điểm b, Nghị định 168/2024/NĐ-CP, người
    điều khiển ô tô không chấp hành hiệu lệnh đèn tín hiệu giao thông (vượt
    đèn đỏ) bị phạt tiền từ 4.000.000 đến 6.000.000 đồng."

    Đoạn trích: [chunk_5 — 168_2024_ND-CP.docx] "Điều 3. Giải thích từ ngữ..."
    Câu hỏi: "Xe máy điện chở 3 người bị phạt bao nhiêu?"
    Trả lời: "Không tìm thấy căn cứ về mức phạt cho hành vi này trong các
    đoạn trích được cung cấp."

    Đoạn trích pháp luật:
    {% for chunk in chunks %}
    [{{ chunk.chunk_id }} — {{ chunk.source_name }}]
    {{ chunk.text }}
    ---
    {% endfor %}

    Câu hỏi: {{ query_vi }}

    {{ ctx.output_format }}
  "#
}
