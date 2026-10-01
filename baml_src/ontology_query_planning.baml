enum OntologyTemporalMode {
  Current
  AtDate
  CompareDates
}

class OntologyQueryPlanOutput {
  query_vi string @description("Câu hỏi độc lập, dùng thuật ngữ pháp lý và giữ nguyên ý định của người hỏi.")
  temporal_mode OntologyTemporalMode
  query_dates string[] @description("Các ngày áp dụng pháp luật ở định dạng YYYY-MM-DD, theo thứ tự xuất hiện.")
}

function PlanOntologyQuery(
  query_vi: string,
  history: ConversationTurn[],
  reference_date: string
) -> OntologyQueryPlanOutput {
  client OpenAICompatibleProvider
  prompt #"
    {{ _.role("user") }}

    Bạn lập kế hoạch tra cứu ontology pháp luật giao thông Việt Nam.

    Hãy làm đồng thời hai việc:
    1. Viết lại câu hỏi cuối thành một câu độc lập, giữ nguyên mọi đối tượng, hành vi,
       văn bản và mốc thời gian; chuẩn hoá thuật ngữ pháp lý như RewriteQueryToLegalTerms.
    2. Xác định các ngày mà người hỏi muốn áp dụng hoặc so sánh quy định.

    Quy tắc thời gian:
    - Current: hỏi hiện tại hoặc không nêu ngày áp dụng; query_dates phải rỗng.
    - AtDate: hỏi quy định áp dụng tại đúng một ngày; query_dates có đúng một ngày.
    - CompareDates: yêu cầu so sánh quy định tại ít nhất hai ngày hoặc hai giai đoạn có
      ngày bắt đầu rõ; query_dates chứa các ngày đó theo thứ tự xuất hiện.
    - Chỉ lấy ngày dùng để xác định hiệu lực. Không coi năm trong số hiệu văn bản, ngày
      ban hành, ngày của sự kiện thực tế không liên quan, hay ngày tham chiếu bên dưới là
      ngày áp dụng.
    - Không tự bịa ngày. Nếu câu hỏi nói “hiện nay” thì dùng Current, không chép ngày tham chiếu.
    - Ngày đầu ra luôn là YYYY-MM-DD.

    Ngày tham chiếu hiện tại: {{ reference_date }}

    {% if history %}
    Hội thoại trước:
    {% for turn in history %}
    Người hỏi: {{ turn.question }}
    Trả lời: {{ turn.answer }}
    {% endfor %}
    {% endif %}

    Câu hỏi cuối: {{ query_vi }}

    {{ ctx.output_format }}
  "#
}
