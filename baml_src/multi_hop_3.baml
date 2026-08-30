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

    Mục tiêu là tìm quan hệ hợp thành cho MỘT tình huống cụ thể, không
    phải tạo câu hỏi đủ rộng để bao phủ ba facts. Ưu tiên precision hơn
    recall; nếu chưa chắc cả ba facts cùng cần thiết thì trả is_valid=false.

    Đặt is_valid=true chỉ khi có thể đặt một câu hỏi tự nhiên về cùng một
    chủ thể, phương tiện, sự kiện và phạm vi áp dụng, trong đó mỗi fact
    cung cấp một phần riêng cần thiết để giải quyết đúng cùng intent đó.
    Ít nhất hai bước nối phải có phụ thuộc nội dung: điều kiện, kết quả
    hoặc kết luận từ một fact cần để hiểu hay áp dụng fact tiếp theo.

    Trước khi kết luận, tự kiểm tra:
    1. Viết thầm một tình huống và một câu hỏi cụ thể, không mở rộng câu
       hỏi sau khi thấy các facts.
    2. Xác định đóng góp riêng của từng fact.
    3. Lần lượt bỏ từng fact. Nếu câu hỏi vẫn trả lời đầy đủ, hoặc chỉ mất
       một nhánh so sánh/liệt kê độc lập, trả is_valid=false.
    4. Kiểm tra cả ba facts có thể đồng thời áp dụng cho chính tình huống
       đó hay không.

    Luôn trả is_valid=false khi:
    - Ba facts chỉ cùng chủ đề, cùng hậu quả hoặc là các phần tử độc lập
      trong một danh sách.
    - Candidate chỉ phù hợp với câu hỏi liệt kê, tổng hợp hoặc so sánh
      được ghép tùy ý.
    - Các facts là những nhánh điều kiện loại trừ nhau, như có/không có
      một dấu hiệu, các khoảng giá trị không giao nhau, hoặc các loại chủ
      thể/phương tiện không thể đồng thời đúng trong một tình huống.
    - Hai facts đã đủ và fact còn lại chỉ diễn giải, lặp lại hoặc cung cấp
      ví dụ khác.
    - Một fact chỉ xác nhận trạng thái nền hoặc tính hợp pháp chung nhưng
      không làm thay đổi điều kiện, chế tài hay kết luận khi áp dụng hai
      facts còn lại. Việc ba facts nói về cùng một phương tiện hoặc giấy
      tờ không đủ tạo thành quan hệ multi-hop-3.
    - Chủ thể, phương tiện, điều kiện, phạm vi hoặc thời gian không thể
      kết hợp thành cùng một vấn đề pháp lý nhất quán.
    - Chuỗi nối phải giữ đúng danh tính và vai trò pháp lý của chủ thể.
      Không đồng nhất một lực lượng, cơ quan hoặc nhóm người có thẩm quyền
      với một chức danh cụ thể chỉ vì cùng thuộc phạm vi xử phạt. Khi có
      giao quyền hoặc ủy quyền, statements phải trực tiếp xác lập đúng
      chức danh giao quyền, người nhận quyền và thẩm quyền được chuyển.
    - Nội dung chủ yếu là so sánh quy định theo thời gian; trường hợp đó
      thuộc temporal, không phải multi-hop-3.
    - Các facts chỉ tạo thành checklist thủ tục hoặc chuỗi thao tác theo
      thứ tự thời gian. Việc bước này thường diễn ra trước bước khác không
      đủ; kết quả pháp lý của bước trước phải là điều kiện cần của bước sau.

    Khi valid, bridge là một câu trần thuật tiếng Việt ngắn nêu đúng một
    vấn đề pháp lý thống nhất và thể hiện chuỗi phụ thuộc. Bridge không
    được là câu hỏi, không kết thúc bằng dấu hỏi, không dùng từ hoặc liên
    từ thuộc ngôn ngữ khác. Chỉ dùng tiếng Việt. Không chép lại toàn bộ
    statements, không ghi ID/Điều khoản thay cho nội dung. Khi invalid,
    bridge=null.

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
