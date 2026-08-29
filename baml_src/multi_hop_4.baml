class MultiHop4BundleBatchItem {
  candidate_id string
  source_fact_ids string[]
  source_bridge string
  facts LegalFactSummary[]
  confirmed_connections EvidenceConnectionSummary[]
}

class ConfirmedMultiHop4Bundle {
  is_valid bool @description(#"
    True chỉ khi một câu hỏi grounded hữu ích thực sự cần đủ cả bốn facts.
  "#)
  bridge string? @description(#"
    Một câu ngắn mô tả vấn đề pháp lý cần kết hợp đủ bốn facts; null khi is_valid=false.
  "#)
}

class MultiHop4BundleBatchResult {
  candidate_id string @description(#"Phải trùng với một candidate_id trong input."#)
  bundle ConfirmedMultiHop4Bundle
}

function ConfirmMultiHop4BundleBatch(items: MultiHop4BundleBatchItem[]) -> MultiHop4BundleBatchResult[] {
  client OpenAICompatibleProvider
  prompt #"
    {{ _.role("system") }}

    Bối cảnh: chuẩn bị evidence bundle chính xác cao cho benchmark RAG
    hỏi đáp về xử lý vi phạm giao thông đường bộ tại Việt Nam.

    Mỗi candidate mở rộng một bundle ba facts đã được xác nhận bằng đúng
    một fact mới. Bundle nguồn và các connection chỉ là gợi ý cấu trúc;
    chúng KHÔNG tự chứng minh cả bốn facts hợp thành một bundle. Statement
    là nguồn pháp lý duy nhất. Không dùng kiến thức ngoài input.

    Mục tiêu là MỘT tình huống và MỘT câu hỏi tự nhiên mà câu trả lời cần
    đủ cả bốn facts. Ưu tiên precision hơn recall; nếu fact mới chỉ cùng
    chủ đề hoặc câu hỏi phải mở rộng để chứa nó, trả is_valid=false.

    Đặt is_valid=true chỉ khi:
    - Cả bốn facts đồng thời áp dụng cho cùng chủ thể, phương tiện, sự kiện,
      thời gian và phạm vi pháp lý cụ thể.
    - Fact mới tạo thêm một bước phụ thuộc nội dung cần thiết từ bundle ba
      facts nguồn; không chỉ thêm ví dụ, định nghĩa nền hoặc nhánh độc lập.
    - Có ít nhất ba bước nối có phụ thuộc nội dung trong toàn bundle.
    - Bỏ bất kỳ một fact nào cũng khiến câu trả lời không còn đầy đủ hoặc
      không thể đi đến cùng kết luận.

    Trước khi kết luận, tự kiểm tra:
    1. Viết thầm một tình huống và một câu hỏi cụ thể trước khi ghép facts.
    2. Nêu đóng góp riêng của từng fact và bước phụ thuộc mà nó tạo ra.
    3. Lần lượt bỏ từng fact; nếu chỉ mất một nhánh liệt kê hoặc chi tiết
       không ảnh hưởng kết luận, trả is_valid=false.
    4. Kiểm tra danh tính và vai trò pháp lý của chủ thể không bị đánh tráo
       qua các nhãn chung như cơ quan, lực lượng hoặc người có thẩm quyền.

    Luôn trả is_valid=false khi:
    - Bốn facts chỉ cùng chủ đề, cùng hậu quả hoặc ghép thành câu hỏi tổng
      hợp, liệt kê hay so sánh tùy ý.
    - Ba facts đã đủ và fact thứ tư chỉ diễn giải, lặp lại, cung cấp ví dụ,
      trạng thái nền hoặc tính hợp pháp chung.
    - Các facts thuộc những nhánh điều kiện loại trừ nhau hoặc không thể
      đồng thời đúng trong một tình huống.
    - Một bước nối cần tự tạo tiền đề, hệ quả, quan hệ chủ thể hoặc phạm vi
      mà statements không trực tiếp hỗ trợ.
    - Nội dung chủ yếu là so sánh quy định theo thời gian; trường hợp đó
      thuộc temporal, không phải multi-hop-4.

    Khi valid, bridge là một câu tiếng Việt ngắn nêu đúng vấn đề pháp lý
    thống nhất và thể hiện chuỗi phụ thuộc bốn facts. Chỉ dùng tiếng Việt;
    không chèn từ thuộc ngôn ngữ khác, nhãn Fact, ID hoặc vị trí input.
    Không chép lại toàn bộ statements. Khi invalid, bridge=null.

    Trả đúng một kết quả cho mỗi input và giữ nguyên candidate_id.

    {{ _.role("user") }}

    Các candidate cần xác nhận:

    {% for item in items %}
    === candidate_id={{ item.candidate_id }} ===
    Bundle ba facts nguồn {{ item.source_fact_ids }}:
    {{ item.source_bridge }}

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
