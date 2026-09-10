class PriorTurn {
  question string
  answer string
}

class GeneratedMultiTurnTurn {
  query string @description(#"Câu hỏi của lượt này, đúng dialogue_act và context_dependency được yêu cầu."#)
  reference_answer string @description(#"Đáp án tham chiếu của lượt này, chỉ dựa trên facts của lượt."#)
  reasoning string @description(#"Giải thích ngắn cách facts của lượt dẫn tới đáp án."#)
  required_claims ClaimReference[] @description(#"Các ý bắt buộc trong đáp án của lượt, trỏ đúng fact_id trong facts của lượt."#)
}

class MultiTurnGenerationBatchInput {
  task_id string
  facts FactContext[]
  persona_description string
  dialogue_act string
  context_dependency string
  answerability string
  history PriorTurn[]
}

class GeneratedMultiTurnTurnBatchItem {
  task_id string
  turn GeneratedMultiTurnTurn
}

function GenerateMultiTurnTurnBatch(
  items: MultiTurnGenerationBatchInput[]
) -> GeneratedMultiTurnTurnBatchItem[] {
  client OpenAICompatibleProvider
  prompt #"
    {{ _.role("system") }}

    Bối cảnh: tạo ground truth multi-turn để đánh giá RAG hội thoại trong domain
    xử lý vi phạm giao thông đường bộ tại Việt Nam. Mỗi task là MỘT lượt của
    một hội thoại; xử lý độc lập từng task theo đúng history được cung cấp.

    Nguồn và grounding:
    - statement và metadata của facts là nguồn pháp lý duy nhất của lượt này.
      Không thêm kiến thức, con số, điều kiện, ngoại lệ, mốc thời gian hay số
      Điều/khoản/điểm không có trong facts. Chỉ gán tên hoặc số hiệu văn bản
      cho nội dung của đúng fact mang metadata đó.
    - Giữ nguyên mọi từ hạn định làm đổi hành vi hoặc kết luận, như chủ thể,
      đối tượng, mục đích, cách thức và điều kiện đi kèm. Không rút một cụm
      điều kiện thành khái niệm rộng hơn, và không bỏ một từ hạn định để biến
      hành vi hẹp trong fact thành hành vi rộng hơn trong query.
    - valid_from và valid_to là điều kiện pháp lý của fact. Nếu query của lượt
      không giới hạn thời điểm, reference_answer phải nêu rõ mốc áp dụng. Không
      tự đặt ngày kết thúc hay quan hệ thay thế khi valid_to là null và
      statements không nêu quan hệ đó.
    - history chỉ là ngữ cảnh để hiểu lượt này; không dùng nội dung của các
      lượt trước làm căn cứ pháp lý cho reference_answer, và không trả lời lại
      các lượt trước.

    Vai trò của lượt (dialogue_act):
    - inquiry: lượt mở đầu, chưa có history. Đặt một câu hỏi độc lập, tự chứa,
      cùng tính chất với single-turn: nêu đủ chủ thể, hành vi và điều kiện.
    - continuity: hỏi khía cạnh pháp lý khác nhưng cùng chủ đề với lượt liền
      trước; facts của lượt này là căn cứ trả lời.
    - deepening: hỏi chi tiết hơn, sâu hơn về cùng khía cạnh pháp lý với lượt
      liền trước. Ví dụ: lượt trước hỏi "Điều 5 quy định gì?", deepening hỏi
      "Giải thích rõ hơn khoản 2", "Khoản 2 áp dụng khi nào?", "Tại sao có
      ngoại lệ đó?". Query phải rõ là theo chiều sâu, không phải chuyển khía cạnh.
    - switch: chuyển sang chủ đề mới. Câu hỏi gần như tự chứa; chỉ được nối
      nhẹ với hội thoại nếu tự nhiên, không dùng đại từ thiếu ngữ cảnh.
    - return: quay lại một chủ đề đã hỏi ở các lượt trước (xem history). Phải
      có dấu hiệu quay lại rõ ràng, ví dụ "quay lại vấn đề lúc đầu về ...",
      rồi nêu lại đủ nội dung pháp lý của chủ đề đó.
    - clarification: hỏi làm rõ một điểm cụ thể ngay trong câu trả lời liền
      trước mà facts của lượt này trả lời được; không hỏi lại toàn bộ vấn đề.
    - underspecified: hỏi về một tình huống thiếu đúng một tình tiết mà facts
      không xác định. Query phải là vụ việc cụ thể, nêu rõ đúng một dữ kiện
      còn thiếu bằng cụm như "chưa rõ thời điểm/chủ thể/vị trí", và dữ kiện
      thiếu đó phải tạo ra ít nhất hai kết luận pháp lý khác nhau.

    Phụ thuộc ngữ cảnh (context_dependency, áp cho mọi lượt đã có history):
    - anaphora: dùng từ chỉ đại gắn với hội thoại ("vậy", "như vậy",
      "trường hợp đó", "mức phạt này") nhưng khi đọc cùng history phải xác định
      duy nhất chủ thể, hành vi được hỏi; không để một câu hiểu được hai cách.
    - ellipsis: lược bỏ chủ thể hoặc hành vi đã rõ từ hội thoại; câu rất ngắn
      như "Còn xe máy thì sao?", "Bị tước bằng không?".
    - Mọi lượt sau lượt đầu phải KHÔNG tự đứng độc lập — đó là mục đích của
      benchmark — nhưng khi ghép với history phải chỉ còn đúng một cách hiểu
      và trỏ về nội dung mà facts của lượt này trả lời.
    - Không lặp nguyên văn câu hỏi cũ và không tự trả lời trong câu hỏi.

    Answerability:
    - answerable: facts của lượt trả lời trọn trọng tâm câu hỏi. Query phải
      khóa đủ mọi điều kiện kích hoạt được facts nêu, kể cả loại phương tiện,
      chủ thể, thời điểm, ngoại lệ và quan hệ pháp lý; đáp án không được kết
      thúc bằng phần "chưa thể xác định".
    - partially_answerable: đúng một tình tiết còn thiếu như mô tả ở
      underspecified; đáp án tách phần có căn cứ và phần chưa thể kết luận,
      không tự điền thông tin thiếu.

    Query contract:
    - Viết như người hỏi thật theo persona, tự nhiên tiếng Việt, không lộ đáp
      án, không thuật ngữ nội bộ, không dùng dấu gạch nối để nối từ.
    - Khoảng độ dài mềm, ưu tiên tự nhiên: inquiry/switch 10-25 từ;
      continuity/return/deepening 8-18 từ; clarification 6-15 từ; underspecified 10-20 từ
      và phải nêu tình huống cộng cụm "chưa rõ ...".
    - Trọng tâm lượt phải thuộc xác định vi phạm giao thông đường bộ, chế tài
      hoặc hậu quả của vi phạm, phát hiện/chứng minh, thẩm quyền, trình tự
      hoặc thi hành trong một vụ vi phạm; không hỏi về cấp/đổi giấy phép,
      đào tạo, đăng kiểm hay quản lý thông thường.
    - Nếu viện dẫn Điều, khoản hoặc điểm, phải nêu văn bản hoặc mô tả đủ hành
      vi để xác định duy nhất nội dung được hỏi; không dùng dẫn chiếu trần.

    Output contract:
    - query tuân thủ dialogue_act, context_dependency và answerability.
    - reference_answer trả lời trực tiếp trọng tâm lượt này, đủ điều kiện,
      ngoại lệ, hệ quả và mốc thời gian cần thiết; luôn giữ cả mốc bắt đầu
      tính của mọi thời hạn; không viết "như đã trả lời ở trên" thay cho nội
      dung, mà diễn đạt lại phần cần thiết một cách trọn vẹn.
    - reasoning giải thích ngắn cách facts của lượt dẫn tới đáp án; không bịa
      citation và không nói "dữ liệu được cung cấp".
    - required_claims có ít nhất một phần tử; claim trỏ đúng fact_id trong
      facts của lượt và tập fact_id phải phủ đúng toàn bộ facts của lượt; mỗi
      claim phải tự giữ đủ chủ thể, hành vi, điều kiện, thời điểm và phạm vi
      làm kết luận đúng. Không tạo claim cho phần chưa thể kết luận của
      partially_answerable.

    Trả đúng một kết quả cho mỗi task, giữ nguyên task_id và thứ tự.

    {{ _.role("user") }}

    Các task cần sinh:
    {% for item in items %}
    ===== TASK {{ item.task_id }} =====
    Lượt này:
    - Persona: {{ item.persona_description }}
    - Dialogue act: {{ item.dialogue_act }}
    - Context dependency: {{ item.context_dependency }}
    - Answerability: {{ item.answerability }}

    Facts của lượt:
    {% for fact in item.facts %}
    [{{ fact.fact_id }}] {{ fact.statement }}
    Văn bản: {{ fact.document_number }}
    Hiệu lực: {{ fact.valid_from }} đến {{ fact.valid_to }}
    ---
    {% endfor %}
    {% if item.history %}
    Hội thoại trước:
    {% for turn in item.history %}
    Người hỏi: {{ turn.question }}
    Trả lời: {{ turn.answer }}
    {% endfor %}
    {% else %}
    Hội thoại trước: không có, đây là lượt mở đầu.
    {% endif %}
    ===== END TASK {{ item.task_id }} =====
    {% endfor %}

    {{ ctx.output_format }}
  "#
}
