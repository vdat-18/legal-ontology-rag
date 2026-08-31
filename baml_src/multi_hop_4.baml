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
    - Bốn facts thuộc cùng một nhánh điều kiện và cùng một chuỗi nhân quả;
      không phải các căn cứ độc lập chỉ tình cờ dẫn đến cùng một quyết định.
    - Fact mới tạo thêm một bước phụ thuộc nội dung cần thiết từ bundle ba
      facts nguồn; không chỉ thêm ví dụ, định nghĩa nền hoặc nhánh độc lập.
    - Có ít nhất ba bước nối có phụ thuộc nội dung trong toàn bundle.
    - Bỏ bất kỳ một fact nào cũng khiến câu trả lời không còn đầy đủ hoặc
      không thể đi đến cùng kết luận.
    - Đóng góp cụ thể của fact mới xuất hiện rõ trong bridge cuối. Nếu bỏ
      fact mới mà bridge về bản chất không đổi, trả is_valid=false.

    Trước khi kết luận, tự kiểm tra:
    1. Viết thầm một tình huống và một câu hỏi cụ thể trước khi ghép facts.
    2. Nêu đóng góp riêng của từng fact và bước phụ thuộc mà nó tạo ra.
    3. Lần lượt bỏ từng fact; nếu chỉ mất một nhánh liệt kê hoặc chi tiết
       thủ tục mà kết luận pháp lý cuối vẫn giữ nguyên, trả is_valid=false.
    4. Kiểm tra danh tính và vai trò pháp lý của chủ thể không bị đánh tráo
       qua các nhãn chung như cơ quan, lực lượng hoặc người có thẩm quyền.
    5. Kiểm tra bridge cuối thể hiện rõ đóng góp của từng fact, đặc biệt là
       fact mới, thay vì chỉ mô tả chủ đề chung của bundle.
    6. Sắp xếp thầm thành chuỗi A -> B -> C -> D, trong đó kết quả pháp lý
       của mỗi bước là tiền đề cần thiết để áp dụng bước sau. Nếu chỉ có thể
       nối các kết quả bằng "và", "đồng thời" hoặc "cần xác định", trả false.
    7. Phân biệt suy luận pháp lý với mô tả thao tác: dừng, kiểm tra, tiếp
       nhận, chuyển, thông báo, lập biên bản, lưu, cập nhật hoặc giao trả
       không tạo thành hop nếu chỉ là các nghĩa vụ độc lập theo trình tự.
    8. Dựng đúng dòng thời gian. Điều kiện làm phát sinh trách nhiệm phải
       tồn tại trước hành vi được xét; không dùng chính hậu quả xử phạt của
       hành vi hiện tại để chứng minh điều kiện đó đã tồn tại từ trước.
    9. Kiểm tra từng fact thuộc quan hệ phụ thuộc hay chỉ là một căn cứ thay
       thế. Hai lý do độc lập cùng dẫn đến thu hồi, xử phạt hoặc tịch thu
       không tạo thành hai hop của cùng một tình huống.

    Luôn trả is_valid=false khi:
    - Bốn facts chỉ cùng chủ đề, cùng hậu quả hoặc ghép thành câu hỏi tổng
      hợp, liệt kê hay so sánh tùy ý.
    - Ba facts đã đủ và fact thứ tư chỉ diễn giải, lặp lại, cung cấp ví dụ,
      trạng thái nền hoặc tính hợp pháp chung.
    - Các facts thuộc những nhánh điều kiện loại trừ nhau hoặc không thể
      đồng thời đúng trong một tình huống, như bản giấy/điện tử, được phép/
      bị cấm hoặc có mặt/vắng mặt.
    - Các facts mô tả các căn cứ thay thế cho cùng một hậu quả, như hai lý
      do thu hồi giấy phép độc lập. Bridge chọn một căn cứ sẽ làm fact còn
      lại dư; bridge chứa cả hai chỉ trở thành câu hỏi liệt kê.
    - Candidate đánh tráo điều kiện có trước với hậu quả có sau, như dùng
      hình thức tước giấy phép phát sinh từ hành vi hiện tại để suy ra rằng
      giấy phép đã bị tước khi một chủ thể khác giao phương tiện.
    - Candidate ghép các cách xử lý tài sản ở những nhánh khác nhau, như
      trả lại cho chủ thể hợp pháp với xử lý tài sản bị tịch thu hoặc tài
      sản bảo đảm, khi statements không trực tiếp xác lập chúng cùng áp dụng.
    - Candidate cần ghép nhánh chấp thuận với từ chối, đạt với không đạt,
      hoặc thực hiện với không thực hiện để dùng đủ facts; một tình huống
      cụ thể chỉ cần một nhánh kết quả.
    - Một fact chỉ áp dụng cho nhóm chủ thể, loại phương tiện hoặc vai trò
      hẹp hơn nhưng tình huống không có statement nào xác lập đúng phạm vi đó.
    - Một định nghĩa, khoảng thời gian, ngoại lệ hoặc phạm vi từ văn bản hay
      bối cảnh này bị chuyển sang văn bản hay bối cảnh khác mà không có
      statement trực tiếp xác lập phạm vi dùng chung.
    - Các facts chỉ ghép thành một quy trình hoặc vòng đời rộng; output của
      mỗi bước không phải tiền đề cần thiết cho bước kế tiếp.
    - Candidate chỉ mô tả đầy đủ vòng đời nghiệp vụ như phát hiện, tiếp nhận,
      xác định, chuyển hồ sơ, ban hành, thông báo, lưu trữ hay cập nhật dữ
      liệu mà không suy ra một kết luận pháp lý duy nhất qua đủ bốn facts.
    - Các facts chỉ là checklist nghĩa vụ, hồ sơ, thời hạn hoặc thao tác
      hành chính. Thứ tự thực hiện hoặc cùng thuộc một thủ tục không phải
      quan hệ multi-hop nếu kết quả pháp lý của bước trước không quyết định
      khả năng áp dụng bước sau.
    - Các facts nêu những hậu quả pháp lý song song của cùng một hành vi,
      như mức phạt cá nhân, trách nhiệm của tổ chức và bồi thường dân sự.
      Cùng chủ thể hoặc cùng sự kiện không biến các nhánh này thành multi-hop.
    - Một bước nối cần tự tạo tiền đề, hệ quả, quan hệ chủ thể hoặc phạm vi
      mà statements không trực tiếp hỗ trợ.
    - Nội dung chủ yếu là so sánh quy định theo thời gian; trường hợp đó
      thuộc temporal, không phải multi-hop-4.

    Khi valid, bridge là một câu trần thuật tiếng Việt ngắn nêu đúng vấn
    đề pháp lý thống nhất và thể hiện chuỗi phụ thuộc bốn facts. Bridge
    không được là câu hỏi, không kết thúc bằng dấu hỏi, không chèn từ thuộc
    ngôn ngữ khác, nhãn Fact, ID hoặc vị trí input. Chỉ dùng tiếng Việt,
    gồm cả từ và từ nối; nếu không thể diễn đạt hoàn toàn bằng tiếng Việt,
    trả false.
    Tự kiểm tra bridge không chứa từ hoặc ký tự thuộc hệ chữ nước ngoài.
    Không chép lại toàn bộ statements. Khi invalid, bridge=null.

    Trả đúng một kết quả cho mỗi input và giữ nguyên candidate_id.
    Không coi source_bridge hoặc connection là bằng chứng về phạm vi, điều
    kiện áp dụng hay danh tính chủ thể; chúng chỉ là gợi ý để kiểm tra facts.

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
