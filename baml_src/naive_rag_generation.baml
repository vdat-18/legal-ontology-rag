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
