class EvidenceConnectionSummary {
  fact_ids string[] @description(#"Hai fact_id của relation pair đã được xác nhận."#)
  bridge string
}

class MultiHop3BundleBatchItem {
  candidate_id string
  facts LegalFactSummary[]
  confirmed_connections EvidenceConnectionSummary[]
}

class ConfirmedMultiHop3Bundle {
  is_valid bool @description(#"
    True chỉ khi một câu hỏi grounded hữu ích thực sự cần đủ cả ba facts.
  "#)
  bridge string? @description(#"
    Một câu ngắn mô tả vấn đề pháp lý cần kết hợp đủ ba facts; null khi is_valid=false.
  "#)
}

class MultiHop3BundleBatchResult {
  candidate_id string @description(#"Phải trùng với một candidate_id trong input."#)
  bundle ConfirmedMultiHop3Bundle
}

function ConfirmMultiHop3BundleBatch(items: MultiHop3BundleBatchItem[]) -> MultiHop3BundleBatchResult[] {
  client OpenAICompatibleProvider
  prompt #"
    {{ _.role("system") }}

    Bối cảnh: chuẩn bị evidence bundle chính xác cao cho benchmark RAG
    hỏi đáp về xử lý vi phạm giao thông đường bộ tại Việt Nam.

    Mỗi candidate gồm đúng ba legal facts. Các connection chỉ cho biết
    từng cặp đã được xác nhận có liên quan; điều đó KHÔNG tự chứng minh
    cả ba facts tạo thành một bundle hợp lệ. Statement là nguồn pháp lý
    duy nhất. Metadata và bridge chỉ hỗ trợ hiểu cách nối, không được dùng
    để bổ sung nội dung không có trong statement.

    Đặt is_valid=true chỉ khi tồn tại một câu hỏi pháp lý tự nhiên, hữu
    ích và grounded mà câu trả lời thực sự cần nội dung riêng của đủ cả
    ba facts. Thực hiện phép kiểm tra loại bỏ: nếu bỏ bất kỳ một fact nào
    mà câu hỏi vẫn được trả lời đầy đủ, trả is_valid=false.

    Luôn trả is_valid=false khi:
    - Ba facts chỉ cùng chủ đề, cùng hậu quả hoặc là các phần tử độc lập
      trong một danh sách.
    - Candidate chỉ phù hợp với câu hỏi liệt kê, tổng hợp hoặc so sánh
      được ghép tùy ý.
    - Hai facts đã đủ và fact còn lại chỉ diễn giải, lặp lại hoặc cung cấp
      ví dụ khác.
    - Chủ thể, phương tiện, điều kiện, phạm vi hoặc thời gian không thể
      kết hợp thành cùng một vấn đề pháp lý nhất quán.
    - Nội dung chủ yếu là so sánh quy định theo thời gian; trường hợp đó
      thuộc temporal, không phải multi-hop-3.

    Khi valid, bridge là một câu ngắn nêu vấn đề pháp lý thống nhất và
    vai trò bổ sung của cả ba facts. Không chép lại toàn bộ statements,
    không ghi ID/Điều khoản thay cho nội dung. Khi invalid, bridge=null.

    Trả đúng một kết quả cho mỗi input và giữ nguyên candidate_id.

    {{ _.role("user") }}

    Các candidate cần xác nhận:

    {% for item in items %}
    === candidate_id={{ item.candidate_id }} ===
    Facts:
    {% for fact in item.facts %}
    [{{ fact.fact_id }}] {{ fact.statement }}
    Văn bản: {{ fact.document_number }}
    Hiệu lực: {{ fact.valid_from }} đến {{ fact.valid_to }}
    {% endfor %}

    Các connection pair đã xác nhận:
    {% for connection in item.confirmed_connections %}
    - {{ connection.fact_ids }}: {{ connection.bridge }}
    {% endfor %}
    === end candidate_id={{ item.candidate_id }} ===
    {% endfor %}

    {{ ctx.output_format }}
  "#
}
