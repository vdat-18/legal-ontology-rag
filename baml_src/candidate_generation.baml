class FactContext {
  fact_id string
  conduct string
  subject string
  conditions string[]
  monetary_penalty string?
  license_points_deducted string?
  additional_penalties string[]
  remedial_measures string[]
  valid_from string?
  valid_to string?
  document_number string
  provision_path string
  evidence_chunk_ids string[] @description(#"
    chunk_id đã có sẵn cho fact này. Mọi evidence_chunk_ids trong
    required_claims phải là tập con của danh sách này cho đúng fact_id.
  "#)
}

class ClaimReference {
  claim string @description(#"Diễn đạt ngắn gọn nội dung claim, bằng tiếng Việt."#)
  fact_id string @description(#"Phải trùng với một fact_id trong facts."#)
  evidence_chunk_ids string[] @description(#"
    Tập con của evidence_chunk_ids thuộc đúng fact_id này. Không tự thêm
    chunk_id không có trong danh sách đã cho.
  "#)
}

class GeneratedSingleTurnCandidate {
  query_vi string @description(#"Câu hỏi tiếng Việt tự nhiên, đúng phong cách được yêu cầu."#)
  reference_answer_vi string
  reasoning_vi string @description(#"Chuỗi lập luận nối evidence tới đáp án, bằng tiếng Việt."#)
  required_claims ClaimReference[] @description(#"Ít nhất một phần tử. Mỗi ý bắt buộc trong đáp án phải có một claim."#)
}

function GenerateSingleTurnCandidate(
  facts: FactContext[],
  persona_description: string,
  query_style: string,
  query_length: string,
  query_form: string,
  query_type_description: string,
  answerability: string,
  bundle_kind: string,
  bridge: string[],
  reasoning_hops: int,
  as_of_date: string?
) -> GeneratedSingleTurnCandidate {
  client OpenAICompatibleProvider
  prompt #"
    {{ _.role("user") }}

    Đóng vai {{ persona_description }} và viết một câu hỏi tiếng Việt về xử
    phạt vi phạm giao thông đường bộ, dựa đúng trên các fact pháp lý đã duyệt
    dưới đây. Sau đó sinh đáp án tham chiếu, lập luận, và các claim bắt buộc.

    Yêu cầu về câu hỏi:
    - Dạng câu hỏi: {{ query_type_description }}
    - Khả năng trả lời yêu cầu: {{ answerability }}
    - Loại evidence bundle: {{ bundle_kind }}
    - Số fact bắt buộc phải dùng: {{ reasoning_hops }}
    {% if bridge %}- Cầu nối bắt buộc giữa các fact: {{ bridge }}{% endif %}
    - Phong cách: {{ query_form }}
    - Độ dài: {{ query_length }}{% if query_length == "short" %} (khoảng 5-10 từ){% elif query_length == "medium" %} (khoảng 10-20 từ){% elif query_length == "long" %} (trên 20 từ, có thể kèm chi tiết tình huống cụ thể){% endif %}
    {% if as_of_date %}- Thời điểm áp dụng: {{ as_of_date }}{% endif %}

    query_vi phải đọc đúng như một người Việt Nam bình thường — không phải
    luật sư, không phải cán bộ soạn văn bản — thực sự gõ câu hỏi này trên
    điện thoại khi cần hỏi gấp ngoài đường. Tuyệt đối không dùng văn phong
    hành chính, không viết như đơn từ hay báo cáo pháp lý. Trừ khi
    {{ persona_description }} là luật sư hoặc cán bộ xử phạt, KHÔNG trích số
    điều/khoản/điểm cụ thể trong query_vi — người bình thường không biết và
    không gõ số hiệu điều luật khi hỏi, chỉ luật sư/cán bộ mới quen cách
    trích dẫn này.

    reference_answer_vi và reasoning_vi thì luôn viết tiếng Việt chuẩn mực,
    đầy đủ dấu, rõ ràng, dễ hiểu, đúng văn phong tư vấn pháp lý thật — mọi
    quy tắc "cách gõ" dưới đây CHỈ áp dụng cho query_vi, không áp dụng cho
    phần đáp án hay lập luận. Tuyệt đối không dùng từ tiếng Anh "fact" hay
    bất kỳ thuật ngữ nội bộ nào, và không viết kiểu "theo dữ liệu/thông tin
    được cung cấp" — viết thẳng vào nội dung quy định như một câu trả lời
    pháp lý thật, không nhắc đến việc đang được cấp dữ liệu.

    Cách gõ bắt buộc cho query_vi — "{{ query_style }}" (các câu ví dụ dưới
    đây chỉ để minh hoạ văn phong/cách gõ, KHÔNG copy nội dung — nội dung câu
    hỏi phải dựa đúng vào fact pháp lý được cung cấp ở phần dưới):
    {% if query_style == "standard" %}
    Gõ đúng chính tả, đầy đủ dấu, câu hoàn chỉnh, giống văn viết bình thường
    khi soạn tin nhắn cẩn thận, có dấu chấm hỏi ở cuối.
    Ví dụ minh hoạ: "Xe ô tô vượt đèn đỏ thì bị phạt bao nhiêu tiền?"
    {% elif query_style == "no_diacritics" %}
    Gõ HOÀN TOÀN không dấu — kiểu gõ vội trên điện thoại chưa bật bàn phím
    tiếng Việt có dấu. Bỏ hết dấu thanh và dấu phụ, giữ chữ cái Latin gần
    nhất (ư→u, ơ→o, ă→a, đ→d, ê→e, ô→o). Không thêm lỗi chính tả nào khác
    ngoài việc bỏ dấu — từ ngữ vẫn đúng, chỉ là không dấu.
    Ví dụ minh hoạ: "xe oto vuot den do thi bi phat bao nhieu tien"
    {% elif query_style == "misspelled" %}
    Vẫn giữ dấu tiếng Việt nhưng gõ sai/tắt tự nhiên như gõ vội bằng điện
    thoại: dùng viết tắt kiểu chat phổ biến (ko/k = không, đc/dc = được,
    vs = với, ntn = như thế nào, j = gì, b = bạn, mng = mọi người,
    v = vậy), gõ nhịu các âm dễ lẫn (ch/tr, d/gi/r, s/x), thiếu dấu câu,
    thiếu hoặc lặp một vài chữ. KHÔNG bỏ hết dấu như no_diacritics — dấu
    tiếng Việt vẫn còn, chỉ sai/tắt rải rác một cách tự nhiên.
    Ví dụ minh hoạ: "oto vượt đèn đỏ thì phạt mấy tiền v ạ, có bị trừ điểm k"
    {% elif query_style == "keyword_style" %}
    Gõ kiểu tìm trên Google — chỉ liệt kê từ khóa cốt lõi, KHÔNG viết thành
    câu hoàn chỉnh, không chủ ngữ/vị ngữ đầy đủ, không dấu chấm hỏi.
    Ví dụ minh hoạ: "oto vuot den do muc phat bao nhieu"
    {% else %}
    Gõ đúng theo mô tả phong cách: {{ query_style }}.
    {% endif %}

    Nguyên tắc bắt buộc:
    - Không tự tìm hoặc thay đổi evidence. Chỉ dùng đúng nội dung các fact
      dưới đây.
    - Mỗi required_claims phải khai báo đúng fact_id nó phái sinh từ đó, và
      evidence_chunk_ids chỉ được lấy từ evidence_chunk_ids của đúng fact đó.
    - Không dùng kiến thức pháp luật sẵn có của bạn để bổ sung nội dung ngoài
      các fact dưới đây.
    - Nếu reasoning_hops lớn hơn 1, câu hỏi và đáp án phải thực sự cần đủ số
      fact đã cung cấp; required_claims phải bao phủ mọi fact.
    - Với partially_answerable, chỉ trả lời phần có căn cứ, rồi nêu rõ CẦN
      HỎI THÊM THÔNG TIN GÌ từ người hỏi để trả lời phần còn lại — không nói
      "dữ liệu/thông tin được cung cấp không đủ", chỉ nói tự nhiên như đang
      thiếu dữ kiện thật từ tình huống. Không biến nó thành câu answerable
      hoàn toàn.
    - Nếu có bridge, câu hỏi và lập luận phải sử dụng đúng cầu nối đó.

    Fact pháp lý:
    {% for fact in facts %}
    [{{ fact.fact_id }}] {{ fact.conduct }} — {{ fact.subject }}
    Điều khoản: {{ fact.provision_path }} ({{ fact.document_number }})
    Phạt tiền: {{ fact.monetary_penalty }}
    Trừ điểm: {{ fact.license_points_deducted }}
    Hình phạt bổ sung: {{ fact.additional_penalties }}
    Biện pháp khắc phục: {{ fact.remedial_measures }}
    Hiệu lực: {{ fact.valid_from }} đến {{ fact.valid_to }}
    Chunk căn cứ: {{ fact.evidence_chunk_ids }}
    ---
    {% endfor %}

    {{ ctx.output_format }}
  "#
}

class SurfaceCritique {
  passes bool @description(#"True nếu câu hỏi đã đúng phong cách yêu cầu, không cần viết lại."#)
  feedback string @description(#"Lý do chưa đạt. Rỗng nếu passes=true."#)
  rewritten_query_vi string @description(#"
    Câu hỏi viết lại theo đúng phong cách yêu cầu, giữ nguyên ý nghĩa và nội
    dung. Giữ nguyên văn bản gốc nếu passes=true.
  "#)
}

function CritiqueSurface(
  query_vi: string,
  query_style: string,
  query_length: string,
  query_form: string
) -> SurfaceCritique {
  client OpenAICompatibleProvider
  prompt #"
    {{ _.role("user") }}

    Chấm câu hỏi dưới đây có đúng phong cách yêu cầu không: cách gõ
    "{{ query_style }}", độ dài "{{ query_length }}", dạng "{{ query_form }}".
    Chỉ đánh giá bề mặt câu chữ — cách gõ, độ dài, mức độ cụ thể/trừu tượng —
    không đánh giá nội dung pháp lý bên trong.

    Nếu chưa đạt, viết lại đúng phong cách yêu cầu mà không đổi ý nghĩa hay
    nội dung câu hỏi.

    Câu hỏi:
    {{ query_vi }}

    {{ ctx.output_format }}
  "#
}

enum EvolutionOperation {
  Concretizing
  Reasoning
  Hypothetical
}

class EvolvedCandidate {
  query_vi string
  reference_answer_vi string
  reasoning_vi string
}

function EvolveCandidate(
  query_vi: string,
  reference_answer_vi: string,
  reasoning_vi: string,
  operation: EvolutionOperation,
  query_length: string
) -> EvolvedCandidate {
  client OpenAICompatibleProvider
  prompt #"
    {{ _.role("user") }}

    Biến đổi câu hỏi và đáp án dưới đây theo đúng một phép, không đổi căn cứ
    pháp lý hay kết luận cuối cùng:

    - Concretizing: thêm tình tiết cụ thể vào tình huống
    - Reasoning: buộc suy luận từ quy định thay vì tra cứu trực tiếp
    - Hypothetical: đặt giả định "nếu ... thì"

    Phép áp dụng: {{ operation }}

    Câu hỏi gốc:
    {{ query_vi }}

    Đáp án gốc:
    {{ reference_answer_vi }}

    Lập luận gốc:
    {{ reasoning_vi }}

    Giữ nguyên chính xác cách gõ/văn phong của câu hỏi gốc ở trên — có dấu
    hay không dấu, viết tắt kiểu chat hay đầy đủ, viết thành câu hay chỉ liệt
    kê từ khóa. Chỉ thêm/đổi nội dung tình huống theo đúng phép biến đổi ở
    trên; không "sửa lại" câu hỏi về chính tả hay văn phong chuẩn mực.

    Độ dài query_vi sau biến đổi vẫn phải giữ đúng mức "{{ query_length }}"
    {% if query_length == "short" %}(khoảng 5-10 từ){% elif query_length == "medium" %}(khoảng 10-20 từ){% elif query_length == "long" %}(trên 20 từ){% endif %}
    như câu gốc — KHÔNG được viết thành cả đoạn văn dài. Concretizing chỉ
    thêm ĐÚNG MỘT chi tiết cụ thể (ví dụ một con số, một mốc thời gian, một
    loại phương tiện), không kể lể thêm nhiều tình tiết cùng lúc.

    reference_answer_vi và reasoning_vi thì luôn viết tiếng Việt chuẩn mực,
    đầy đủ dấu, rõ ràng, đúng văn phong tư vấn pháp lý thật. Tuyệt đối không
    dùng từ tiếng Anh "fact" hay thuật ngữ nội bộ, không viết kiểu "theo dữ
    liệu/thông tin được cung cấp".

    Trả về cả lập luận đã cập nhật để luôn nhất quán với câu hỏi và đáp án sau
    biến đổi.

    {{ ctx.output_format }}
  "#
}
