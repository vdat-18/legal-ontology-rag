class FactContext {
  fact_id string
  statement string
  valid_from string?
  valid_to string?
  document_number string
}

class ClaimReference {
  claim string @description(#"Diễn đạt ngắn gọn nội dung claim, bằng tiếng Việt."#)
  fact_id string @description(#"Phải trùng với một fact_id trong facts."#)
}

class GeneratedSingleTurnCandidate {
  query_vi string @description(#"Câu hỏi tiếng Việt tự nhiên, đúng phong cách được yêu cầu."#)
  reference_answer_vi string
  reasoning_vi string @description(#"Chuỗi lập luận nối evidence tới đáp án, bằng tiếng Việt."#)
  required_claims ClaimReference[] @description(#"Ít nhất một phần tử. Mỗi ý bắt buộc trong đáp án phải có một claim."#)
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
    {{ _.role("user") }}

    Nhiệm vụ: với mỗi task, tạo một mẫu ground truth single-turn để đánh giá
    RAG trong domain xử lý vi phạm giao thông đường bộ tại Việt Nam. Chỉ dùng
    facts của task; trả đúng một kết quả, giữ nguyên task_id và thứ tự.

    {% for item in items %}
    ===== TASK {{ item.task_id }} =====
    Yêu cầu scenario:
    - Người hỏi: {{ item.persona_description }}
    - Answerability: {{ item.answerability }}
    - Evidence shape: {{ item.bundle_kind }}, reasoning_hops={{ item.reasoning_hops }}
    {% if item.bridge %}- Nội dung nối các fact: {{ item.bridge }}{% endif %}
    - Form={{ item.query_form }}, style={{ item.query_style }}, length={{ item.query_length }}

    Grounding và coverage:
    - Tự chọn một câu hỏi pháp lý hữu ích, tự nhiên và đủ cụ thể được facts
      trực tiếp hỗ trợ; không ép facts vào một loại câu hỏi định trước.
    - Mọi kết luận, con số, điều kiện, ngoại lệ và mốc thời gian phải có trong
      facts. Không thêm kiến thức pháp luật, tình tiết làm đổi kết luận, số
      Điều/khoản/điểm không được cung cấp.
    - Bridge chỉ mô tả cách các fact được nối, không phải nguồn pháp lý. Không
      đưa nội dung chỉ xuất hiện trong bridge vào query, answer hoặc claims.
    - Với reasoning_hops=1, một fact phải đủ để trả lời. Với reasoning_hops>1,
      câu hỏi và đáp án phải thực sự cần mọi fact, required_claims bao phủ mọi
      fact và thể hiện đúng mối nối có căn cứ trong facts; không ghép nhiều
      fact chỉ để làm dài.
    - Nếu answerable, facts phải trả lời trọn câu hỏi. Nếu partially_answerable,
      câu hỏi phải thiếu đúng một thông tin ảnh hưởng việc áp dụng mà facts
      không xác định; đáp án nêu phần kết luận có căn cứ và phần chưa thể kết
      luận, không tự điền dữ kiện thiếu.

    query_vi:
    - Viết như người hỏi thật theo persona, không dùng thuật ngữ nội bộ, không
      lộ đáp án. Chỉ luật sư hoặc cán bộ xử phạt mới nên hỏi bằng số điều khoản
      khi số đó có trong facts.
    - Form=specific: nêu tình huống cụ thể. Form=abstract: hỏi trực tiếp về quy
      tắc nhưng vẫn tự nhiên, không sao chép nguyên statement.
    - Tuân thủ length: short khoảng 5-10 từ; medium 10-20 từ; long trên 20 từ
      nhưng vẫn là một câu hỏi gọn.
    - Tuân thủ chính xác style:
    {% if item.query_style == "standard" %}
      standard: hỏi trực tiếp, tiếng Việt đủ dấu, đúng chính tả và có dấu hỏi.
    {% elif item.query_style == "conversational" %}
      conversational: giọng hội thoại tự nhiên, có thể xưng tôi/em/mình và kể
      tình huống ngắn như khi hỏi trên diễn đàn hoặc ứng dụng nhắn tin.
    {% elif item.query_style == "polite" %}
      polite: lịch sự, tự nhiên, có thể mở đầu bằng cho hỏi, xin hỏi hoặc kết
      thúc bằng mong được giải đáp; không viết thành công văn.
    {% elif item.query_style == "formal" %}
      formal: trang trọng, chính xác, trung tính như câu hỏi nghiệp vụ hoặc tư
      vấn pháp lý; chỉ dùng thuật ngữ và căn cứ có trong facts.
    {% elif item.query_style == "no_diacritics" %}
      no_diacritics: query hoàn toàn không dấu, không thêm lỗi chính tả.
    {% elif item.query_style == "misspelled" %}
      misspelled: vẫn có dấu, chỉ thêm vài lỗi hoặc viết tắt chat tự nhiên.
    {% elif item.query_style == "keyword_style" %}
      keyword_style: chỉ các từ khóa cốt lõi, không thành câu, không dấu hỏi.
    {% else %}
      Thực hiện đúng style={{ item.query_style }}.
    {% endif %}

    Đáp án, lập luận và claims:
    - reference_answer_vi trả lời trực tiếp, đủ điều kiện/ngoại lệ và hệ quả
      cần thiết; dùng tiếng Việt chuẩn dù query có lỗi hoặc không dấu.
    - reasoning_vi giải thích ngắn cách facts dẫn tới đáp án, theo đúng thứ tự
      suy luận; không bịa citation và không nói "dữ liệu được cung cấp".
    - Mỗi required claim là một ý bắt buộc trong đáp án, trỏ đúng fact_id và
      được code gắn provenance từ evidence của fact đó. Không tạo claim cho
      phần chưa thể kết luận của partially_answerable và không lặp claim cùng
      nội dung.

    Trước khi trả lời, tự kiểm tra query, answer, reasoning, claims, số hop,
    answerability và evidence nhất quán với nhau.

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
