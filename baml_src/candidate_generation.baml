class FactContext {
  fact_id string
  statement string
  valid_from string?
  valid_to string?
  document_number string
}

class ClaimReference {
  claim string @description(#"Một ý bắt buộc trong reference_answer, viết ngắn gọn."#)
  fact_id string @description(#"Phải trùng với một fact_id trong facts."#)
}

class GeneratedSingleTurnCandidate {
  query string @description(#"Câu hỏi tự nhiên, đúng phong cách được yêu cầu."#)
  reference_answer string @description(#"Đáp án tham chiếu chỉ dựa trên facts."#)
  reasoning string @description(#"Giải thích ngắn cách facts dẫn tới đáp án."#)
  required_claims ClaimReference[] @description(#"Các ý bắt buộc trong đáp án và fact hỗ trợ."#)
}

class CandidateGenerationBatchInput {
  task_id string
  facts FactContext[]
  persona_description string
  query_style string
  query_length string
  query_form string
  answerability string
  bundle_kind string
  bridge string?
  reasoning_hops int
}

class GeneratedSingleTurnCandidateBatchItem {
  task_id string
  candidate GeneratedSingleTurnCandidate
}

function GenerateSingleTurnCandidateBatch(
  items: CandidateGenerationBatchInput[]
) -> GeneratedSingleTurnCandidateBatchItem[] {
  client OpenAICompatibleProvider
  prompt #"
    {{ _.role("system") }}

    Bối cảnh: tạo ground truth single-turn để đánh giá RAG trong domain
    xử lý vi phạm giao thông đường bộ tại Việt Nam. Xử lý độc lập
    từng task.

    Nguồn và grounding:
    - statement và metadata của facts là nguồn pháp lý duy nhất. Không
      thêm kiến thức, con số, điều kiện, ngoại lệ, mốc thời gian,
      tình tiết làm đổi kết luận hay số Điều/khoản/điểm không có trong facts.
      Chỉ gán tên hoặc số hiệu văn bản cho nội dung của đúng fact mang metadata
      đó; không chuyển citation giữa các facts trong cùng bundle.
    - Giữ nguyên mọi từ hạn định làm đổi hành vi hoặc kết luận, như mục đích,
      cách thức, chủ thể, đối tượng và điều kiện đi kèm. Không được bỏ một từ
      hạn định để biến hành vi hẹp trong fact thành hành vi rộng hơn trong query.
      Các điều kiện như phương tiện đang bị tạm giữ, bị tịch thu hoặc thuộc
      một thủ tục cụ thể phải xuất hiện trong query nếu đáp án phụ thuộc vào đó.
      Điều kiện phải được giữ theo cụm nghĩa đầy đủ; không rút "đã đăng ký biện
      pháp bảo đảm thế chấp" thành "đã thế chấp", hoặc rút một chủ thể, địa điểm,
      loại thiết bị, loại thủ tục hay trạng thái pháp lý thành khái niệm rộng hơn.
    - Persona, form, style và length chỉ điều khiển cách hỏi, không
      bổ sung sự kiện, vai trò hoặc quan hệ như con, trẻ em, chủ sở hữu,
      người cho mượn, người trực tiếp vi phạm hay hình thức phát hiện như
      phạt nguội. bundle_kind và reasoning_hops mô tả hình dạng
      evidence. bridge chỉ gợi cách nối facts, không phải nguồn pháp lý.
    - Tự chọn một câu hỏi hữu ích và tự nhiên được facts hỗ trợ;
      không ghép facts chỉ để làm câu hỏi phức tạp.
    - Trọng tâm query bắt buộc là xác định hành vi vi phạm giao thông đường
      bộ, chế tài/hậu quả của vi phạm, hoặc phát hiện, chứng minh, thẩm quyền,
      trình tự và thi hành trong một vụ vi phạm. Không chuyển facts thành câu
      hỏi về cấp/đổi giấy phép, đào tạo, sát hạch, khám sức khỏe, bảo hiểm,
      đăng kiểm, vận hành hoặc quản lý thông thường. Bundle đã qua scope gate;
      vẫn phải tự kiểm tra query và reference_answer không rời domain.
    - Với quy định xử lý vi phạm hành chính có tính áp dụng chung, query phải
      đặt rõ quy định đó trong một vụ vi phạm giao thông đường bộ. Không tạo
      query chung có thể thuộc bất kỳ lĩnh vực nào. Không sử dụng fact chỉ nói
      về vi phạm xuất nhập cảnh, xuất nhập khẩu hoặc quản lý cửa khẩu để tạo
      câu hỏi về vi phạm giao thông đường bộ.
    - Query và reference_answer phải tự đứng độc lập. Không dùng các cụm thiếu
      tiền đề như "hành vi này", "trường hợp này", "tổ chức này", "quy định
      trên" hoặc chỉ hỏi một Điều/khoản/điểm mà không nêu nội dung pháp lý cần
      giải quyết. Mọi chủ thể, hành vi và điều kiện cần hiểu câu hỏi phải xuất
      hiện ngay trong query.
    - Nếu query viện dẫn Điều, khoản hoặc điểm, phải nêu văn bản hoặc mô tả đủ
      hành vi/quy tắc để xác định duy nhất nội dung được hỏi. Không dùng một dẫn
      chiếu trần như "khoản 12 Điều 12" làm thay cho hành vi pháp lý.
    - Không hỏi đơn thuần ý nghĩa biển báo, quy tắc lưu thông, điều kiện kỹ
      thuật, thủ tục cấp phép hoặc nghiệp vụ nội bộ của cơ quan. Chỉ sử dụng
      các nội dung đó khi facts trực tiếp nối chúng với việc xác định, xử lý
      hoặc thi hành đối với một vi phạm giao thông đường bộ. Nếu draft chỉ hỏi
      người tham gia giao thông "phải làm gì" trong hoạt động bình thường,
      phải viết lại theo đúng trọng tâm xử lý vi phạm mà facts thực sự hỗ trợ;
      không tự suy diễn rằng không tuân thủ sẽ bị phạt.

    Scenario contract:
    - answerable: facts trả lời trọn câu hỏi. Chỉ hỏi kết luận áp dụng
      khi facts chứa đủ mọi điều kiện cần thiết; một ngưỡng hoặc một
      điều kiện riêng lẻ không đủ để khẳng định thủ tục được áp dụng. Nếu
      query hỏi kết quả như "phạt thế nào", đáp án phải xác định được kết
      quả đó; chỉ loại trừ một mức hoặc một khả năng chưa phải là trả lời đủ.
      Query phải xác định đủ từng điều kiện kích hoạt được facts nêu, kể cả
      hành vi thuộc đúng điểm/khoản, loại phương tiện, chủ thể, thời điểm,
      ngoại lệ và quan hệ như "vi phạm nhiều lần và bị xử phạt từng hành vi".
    - partially_answerable: câu hỏi thiếu đúng một thông tin ảnh hưởng
      việc áp dụng mà facts không xác định; đáp án tách phần có căn
      cứ và phần chưa thể kết luận, không tự điền thông tin thiếu. Thông tin
      thiếu phải là tình tiết của vụ việc như thời điểm, chủ thể hoặc vị trí;
      không dùng partially_answerable để hỏi mức phạt, danh sách hành vi,
      biện pháp hay trách nhiệm pháp lý mà facts hoàn toàn không cung cấp.
      Query phải là một tình huống cụ thể có đúng một chỗ chưa rõ; không biến
      một câu hỏi quy tắc chung đã được facts trả lời đầy đủ thành partially_answerable.
      Tình tiết chưa rõ phải thực sự tạo ra ít nhất hai kết luận pháp lý khác
      nhau. Không nêu các khả năng đều cùng thỏa một điều kiện rồi kết luận rằng
      chưa thể trả lời, và không mô tả sẵn đầy đủ mọi khả năng đều cho cùng đáp án.
    - reasoning_hops=1: một fact đủ trả lời. reasoning_hops>1: mọi
      fact phải đóng góp một ý cần thiết để trả lời đúng trọng tâm câu hỏi,
      không chỉ được chèn thêm vào đáp án cho đủ số fact.
      Không tạo multi-hop bằng cách ghép nhiều câu hỏi độc lập, hỏi thêm một
      phân loại chung cho kết luận cụ thể, hoặc nối một nghĩa vụ nền với chế
      tài đã được fact khác quy định trực tiếp. Các facts phải cùng tham gia
      vào một kết luận pháp lý thống nhất và không fact nào là bối cảnh thừa.
    - single_hop dùng một fact; multi_hop kết hợp các fact bổ sung;
      temporal phân biệt quy định theo thay đổi hoặc hiệu lực thời gian.
      Với temporal, không suy ra thay thế hoặc hết hiệu lực chỉ từ valid_from
      của fact mới. Khi valid_to của fact cũ là null và statements không nêu
      quan hệ thay đổi, không được khẳng định hai fact loại trừ nhau theo thời
      điểm. Mọi kết luận phụ thuộc ngày hiệu lực phải nêu đúng mốc áp dụng trong
      query hoặc reference_answer.
    - valid_from và valid_to là điều kiện pháp lý của fact, không phải metadata
      để bỏ qua. Nếu query không tự giới hạn thời điểm, reference_answer phải
      nói rõ kết luận áp dụng từ ngày nào hoặc trong khoảng nào. Không áp dụng
      vô điều kiện một fact có valid_from cho vụ việc chưa rõ thời điểm.

    Query contract:
    - Viết như người hỏi thật theo persona, không lộ đáp án, không dùng
      thuật ngữ nội bộ. specific nêu tình huống; abstract hỏi quy tắc
      tự nhiên, không sao chép statement. Query phải hỏi đúng kết luận mà
      reference_answer trả lời; không hỏi điều kiện nếu facts chỉ nêu thủ tục.
    - Mức độ khái quát của query không được rộng hơn facts. Không hỏi "những
      ai", "những hành vi nào", "biện pháp nào" hoặc một danh sách đầy đủ
      nếu facts chỉ chứng minh một phần. Nếu fact chỉ nêu hành vi bị cấm hoặc
      nghĩa vụ, chỉ hỏi hành vi có bị cấm hoặc phải thực hiện gì; không hỏi
      mức phạt khi fact không có chế tài.
    - length tính theo các từ ngăn cách bằng khoảng trắng: short 5-10
      từ; medium 10-20 từ; long trên 20 từ nhưng gọn. Đếm lại và viết
      lại query nếu nằm ngoài khoảng. Short có giới hạn cứng tối đa 10
      từ, kể cả khi phải bỏ bớt từ lịch sự hoặc chi tiết phụ. Với short,
      chủ động viết 5-8 từ để còn biên an toàn; tuyệt đối không trả query
      11 từ rồi gắn nhãn short.
    - style: standard=hỏi trực tiếp, đủ dấu và có dấu hỏi;
      conversational=hội thoại;
      polite=lịch sự; formal=trang trọng, trung tính, không dùng từ viết tắt
      kiểu tin nhắn; no_diacritics=hoàn toàn
      không dấu; misspelled=có vài lỗi gõ hoặc viết tắt tự nhiên;
      keyword_style=chỉ từ khóa cốt lõi, không thành câu, không dấu hỏi.
    - Với no_diacritics, mọi ký tự trong query đều không dấu, kể cả chữ đ/Đ
      trong từ viết tắt. Với keyword_style, không dùng đại từ chỉ xuất thiếu
      ngữ cảnh như "này", "đó", "trên".

    Output contract:
    - query tuân thủ scenario.
    - reference_answer trả lời trực tiếp, đủ điều kiện, ngoại lệ và
      hệ quả cần thiết; luôn dùng tiếng Việt chuẩn. Mọi thời hạn phải giữ cả
      mốc bắt đầu tính, như "kể từ ngày phát hiện", "kể từ ngày hết thời hạn
      tạm giữ" hoặc mốc tương ứng trong fact; không chỉ giữ con số ngày.
      Nếu fact chỉ cho biết một cụm từ được bổ sung hoặc bãi bỏ, chỉ kết luận
      đúng thay đổi câu chữ và phạm vi trực tiếp được chứng minh; không suy ra
      toàn bộ chế tài, thẩm quyền hoặc biện pháp đã phát sinh hay chấm dứt.
    - reasoning giải thích ngắn cách facts dẫn tới đáp án; không bịa
      citation hoặc nói "dữ liệu được cung cấp".
    - required_claims có ít nhất một phần tử, liệt kê các ý bắt buộc
      trong reference_answer, không lặp nội dung và trỏ đúng fact_id.
      Với reasoning_hops>1,
      claims phải bao phủ mọi fact. Không tạo claim cho phần chưa thể kết
      luận của partially_answerable. Mỗi claim phải tự giữ đủ chủ thể, hành vi,
      điều kiện, ngoại lệ, thời điểm và phạm vi làm kết luận đúng; không dùng
      câu rút gọn rộng hơn fact dù reference_answer đã nêu điều kiện ở câu khác.
      Nếu fact dùng nhiều điều kiện liên hợp, từng claim dựa vào fact đó phải
      giữ trọn các điều kiện; không tách một vế thành quy tắc áp dụng độc lập.
      Giữ đúng đại lượng dùng để so sánh hoặc kích hoạt quy tắc: không đổi
      "mức phạt khác nhau" thành "khung thời hạn tước khác nhau", hay thay một
      tiêu chí pháp lý bằng tiêu chí gần nghĩa nhưng không tương đương.
      Điều kiện dẫn chiếu như "theo quy định tại Điều ..." cũng là một phần của
      phạm vi áp dụng; phải giữ nguyên trong query, answer và claim khi kết luận
      phụ thuộc vào nó, không rút thành mô tả chung rộng hơn.
    - Nếu statement có cụm thiếu đơn vị hoặc câu chưa hoàn chỉnh, không
      suy đoán phần bị thiếu và không lặp nguyên cụm lỗi; chỉ dùng phần
      mệnh đề hoàn chỉnh vẫn được fact hỗ trợ.

    Trước khi trả kết quả, kiểm tra riêng từng task theo đúng thứ tự:
    1. Query tự đứng độc lập và trọng tâm thuộc xử lý vi phạm giao thông
       đường bộ, không phải vi phạm hành chính chung, vi phạm cửa khẩu,
       quy tắc hoặc quản lý thông thường. Xóa mọi quan hệ gia đình, tư cách
       chủ thể và cách thức phát hiện không xuất hiện trong facts.
       Không tự đổi "giấy phép" chung thành giấy phép lái xe hoặc một loại
       giấy phép cụ thể khi facts không nêu loại đó.
    2. Đếm lại số từ theo query_length và viết lại cho tới khi đúng khoảng.
    3. Kiểm tra query_style trên toàn bộ query, đặc biệt no_diacritics và
       keyword_style.
    4. Kiểm tra reference_answer, reasoning, required_claims, answerability
       và reasoning_hops nhất quán, không vượt facts. Với answerable, đáp án
       không được nói thiếu điều kiện để kết luận hoặc chỉ loại trừ một khả
       năng mà chưa xác định kết quả được hỏi. Với partially_answerable,
       facts phải trả lời được phần pháp lý đã biết và chỉ thiếu đúng một tình
       tiết vụ việc; nếu query hiện là câu hỏi quy tắc chung, phải viết lại thành
       một vụ việc cụ thể chỉ thiếu tình tiết đó. Với multi-hop, thử bỏ từng fact; nếu vẫn trả lời được đúng
       trọng tâm query thì phải viết lại query để fact đó trở thành cần thiết.
       Cuối cùng đối chiếu từng điều kiện trong reference_answer với query;
       không được để đáp án tự thêm điều kiện quyết định mà query không nêu.
       Đối chiếu lại từng required_claim độc lập với fact được trỏ tới; claim
       chỉ đạt khi vẫn đúng nguyên nghĩa mà không cần mượn điều kiện từ claim khác.
       Với mỗi fact có valid_from hoặc valid_to, kiểm tra query và answer đã nêu
       giới hạn thời gian cần thiết chưa. Với câu hỏi ghép nhiều kết luận, nếu
       facts không trả lời trọn tất cả các vế thì phải bỏ vế không được hỗ trợ,
       không giữ câu hỏi rộng rồi trả lời một phần dưới nhãn answerable.
       Với answerable, lập danh sách từng điều kiện kích hoạt và ngoại lệ trong
       facts rồi xác nhận query đã khóa hết; không được để đáp án tự giả định
       điều kiện còn thiếu. Với partially_answerable, xác nhận ngoài đúng một
       tình tiết được nêu là chưa rõ, không còn chiều áp dụng nào khác mơ hồ.
    Nếu bất kỳ mục nào chưa đạt, tự viết lại candidate trước khi trả. Trả đúng
    một kết quả cho mỗi task, giữ nguyên task_id và thứ tự.

    {{ _.role("user") }}

    Các task cần sinh:
    {% for item in items %}
    ===== TASK {{ item.task_id }} =====
    Scenario:
    - Persona: {{ item.persona_description }}
    - Answerability: {{ item.answerability }}
    - Evidence: bundle_kind={{ item.bundle_kind }}, reasoning_hops={{ item.reasoning_hops }}
    {% if item.bridge %}- Bridge: {{ item.bridge }}{% endif %}
    - Query: form={{ item.query_form }}, style={{ item.query_style }}, length={{ item.query_length }}

    Facts:
    {% for fact in item.facts %}
    [{{ fact.fact_id }}] {{ fact.statement }}
    Văn bản: {{ fact.document_number }}
    Hiệu lực: {{ fact.valid_from }} đến {{ fact.valid_to }}
    ---
    {% endfor %}
    ===== END TASK {{ item.task_id }} =====
    {% endfor %}

    {{ ctx.output_format }}
  "#
}
