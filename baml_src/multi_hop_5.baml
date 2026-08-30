class MultiHop5BundleBatchItem {
  candidate_id string
  source_fact_ids string[]
  source_bridge string
  facts LegalFactSummary[]
  confirmed_connections EvidenceConnectionSummary[]
}

class ConfirmedMultiHop5Bundle {
  is_valid bool @description(#"
    True chỉ khi một câu hỏi grounded hữu ích thực sự cần đủ cả năm facts.
  "#)
  bridge string? @description(#"
    Một câu ngắn mô tả vấn đề pháp lý cần kết hợp đủ năm facts; null khi is_valid=false.
  "#)
}

class MultiHop5BundleBatchResult {
  candidate_id string @description(#"Phải trùng với một candidate_id trong input."#)
  bundle ConfirmedMultiHop5Bundle
}

function ConfirmMultiHop5BundleBatch(items: MultiHop5BundleBatchItem[]) -> MultiHop5BundleBatchResult[] {
  client OpenAICompatibleProvider
  prompt #"
    {{ _.role("system") }}

    Bối cảnh: chuẩn bị evidence bundle chính xác cao cho benchmark RAG
    hỏi đáp về xử lý vi phạm giao thông đường bộ tại Việt Nam.

    Mỗi candidate mở rộng một bundle bốn facts đã được xác nhận bằng đúng
    một fact mới. Bundle nguồn và các connection chỉ là gợi ý cấu trúc;
    chúng KHÔNG tự chứng minh cả năm facts hợp thành một bundle. Statement
    là nguồn pháp lý duy nhất. Không dùng kiến thức ngoài input.

    Mục tiêu là MỘT tình huống và MỘT câu hỏi tự nhiên mà câu trả lời cần
    đủ cả năm facts. Ưu tiên precision hơn recall; nếu fact mới chỉ cùng
    chủ đề hoặc câu hỏi phải mở rộng tùy ý để chứa nó, trả is_valid=false.

    Đặt is_valid=true chỉ khi:
    - Cả năm facts đồng thời áp dụng cho cùng chủ thể, phương tiện, sự kiện,
      thời gian và phạm vi pháp lý cụ thể.
    - Fact mới tạo thêm một bước phụ thuộc nội dung cần thiết từ bundle bốn
      facts nguồn; không chỉ thêm ví dụ, định nghĩa nền hoặc nhánh độc lập.
    - Có ít nhất bốn bước nối có phụ thuộc nội dung trong toàn bundle.
    - Bỏ bất kỳ một fact nào cũng khiến câu trả lời không còn đầy đủ hoặc
      không thể đi đến cùng kết luận.
    - Đóng góp cụ thể của fact mới xuất hiện rõ trong bridge cuối. Nếu bỏ
      fact mới mà bridge về bản chất không đổi, trả is_valid=false.

    Trước khi kết luận, tự kiểm tra:
    1. Viết thầm một tình huống và một câu hỏi cụ thể trước khi ghép facts.
    2. Nêu đóng góp riêng của từng fact và bước phụ thuộc mà nó tạo ra.
    3. Lần lượt bỏ từng fact; nếu chỉ mất một nhánh liệt kê hoặc chi tiết
       không ảnh hưởng kết luận, trả is_valid=false.
    4. Kiểm tra danh tính và vai trò pháp lý của chủ thể không bị đánh tráo
       qua các nhãn chung như cơ quan, lực lượng hoặc người có thẩm quyền.
    5. Kiểm tra fact thứ năm nối vào chuỗi lập luận, không biến bundle thành
       hai nhóm câu hỏi độc lập được ghép chung.
    6. Kiểm tra bridge cuối thể hiện rõ đóng góp của từng fact, đặc biệt là
       fact mới, thay vì chỉ mô tả chủ đề chung của bundle.
    7. Sắp xếp thầm thành chuỗi A -> B -> C -> D -> E, trong đó kết quả
       pháp lý của mỗi bước là tiền đề cần thiết để áp dụng bước sau. Nếu
       chỉ có thể nối các yêu cầu bằng "và", "khi" hoặc "trong đó", trả false.

    Luôn trả is_valid=false khi:
    - Năm facts chỉ cùng chủ đề, cùng hậu quả hoặc ghép thành câu hỏi tổng
      hợp, liệt kê hay so sánh tùy ý.
    - Bốn facts đã đủ và fact thứ năm chỉ diễn giải, lặp lại, cung cấp ví dụ,
      trạng thái nền hoặc tính hợp pháp chung.
    - Các facts thuộc những nhánh điều kiện loại trừ nhau hoặc không thể
      đồng thời đúng trong một tình huống, như bản giấy/điện tử, được phép/
      bị cấm hoặc có mặt/vắng mặt.
    - Một fact chỉ áp dụng cho nhóm chủ thể, loại phương tiện hoặc vai trò
      hẹp hơn nhưng tình huống không có statement nào xác lập đúng phạm vi đó.
    - Một định nghĩa, khoảng thời gian, ngoại lệ hoặc phạm vi từ văn bản hay
      bối cảnh này bị chuyển sang văn bản hay bối cảnh khác mà không có
      statement trực tiếp xác lập phạm vi dùng chung.
    - Các facts chỉ ghép thành một quy trình hoặc vòng đời rộng; output của
      mỗi bước không phải tiền đề cần thiết cho bước kế tiếp.
    - Các facts chỉ là checklist nghĩa vụ thủ tục cùng áp dụng trong một quy
      trình; thực hiện một nghĩa vụ không xác lập tiền đề pháp lý cho nghĩa
      vụ kế tiếp, dù tất cả có thể xuất hiện trong cùng một tình huống.
    - Một bước nối cần tự tạo tiền đề, hệ quả, quan hệ chủ thể hoặc phạm vi
      mà statements không trực tiếp hỗ trợ.
    - Nội dung chủ yếu là so sánh quy định theo thời gian; trường hợp đó
      thuộc temporal, không phải multi-hop-5.

    Khi valid, bridge là một câu tiếng Việt ngắn nêu đúng vấn đề pháp lý
    thống nhất và thể hiện chuỗi phụ thuộc năm facts. Chỉ dùng tiếng Việt;
    không chèn từ thuộc ngôn ngữ khác, nhãn Fact, ID hoặc vị trí input.
    Không chép lại toàn bộ statements. Khi invalid, bridge=null.

    Trả đúng một kết quả cho mỗi input và giữ nguyên candidate_id.
    Không coi source_bridge hoặc connection là bằng chứng về phạm vi, điều
    kiện áp dụng hay danh tính chủ thể; chúng chỉ là gợi ý để kiểm tra facts.

    {{ _.role("user") }}

    Các candidate cần xác nhận:

    {% for item in items %}
    === candidate_id={{ item.candidate_id }} ===
    Bundle bốn facts nguồn {{ item.source_fact_ids }}:
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
