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
    - Persona, form, style và length chỉ điều khiển cách hỏi, không
      bổ sung sự kiện. bundle_kind và reasoning_hops mô tả hình dạng
      evidence. bridge chỉ gợi cách nối facts, không phải nguồn pháp lý.
    - Tự chọn một câu hỏi hữu ích và tự nhiên được facts hỗ trợ;
      không ghép facts chỉ để làm câu hỏi phức tạp.

    Scenario contract:
    - answerable: facts trả lời trọn câu hỏi.
    - partially_answerable: câu hỏi thiếu đúng một thông tin ảnh hưởng
      việc áp dụng mà facts không xác định; đáp án tách phần có căn
      cứ và phần chưa thể kết luận, không tự điền thông tin thiếu.
    - reasoning_hops=1: một fact đủ trả lời. reasoning_hops>1: mọi
      fact phải đóng góp một ý cần thiết cho câu hỏi và đáp án.
    - single_hop dùng một fact; multi_hop kết hợp các fact bổ sung;
      temporal phân biệt quy định theo thay đổi hoặc hiệu lực thời gian.

    Query contract:
    - Viết như người hỏi thật theo persona, không lộ đáp án, không dùng
      thuật ngữ nội bộ. specific nêu tình huống; abstract hỏi quy tắc
      tự nhiên, không sao chép statement.
    - length: short 5-10 từ; medium 10-20 từ; long trên 20 từ nhưng gọn.
    - style: standard=hỏi trực tiếp, đủ dấu và có dấu hỏi;
      conversational=hội thoại;
      polite=lịch sự; formal=trang trọng, trung tính; no_diacritics=hoàn toàn
      không dấu; misspelled=có vài lỗi gõ hoặc viết tắt tự nhiên;
      keyword_style=chỉ từ khóa cốt lõi, không thành câu, không dấu hỏi.

    Output contract:
    - query tuân thủ scenario.
    - reference_answer trả lời trực tiếp, đủ điều kiện, ngoại lệ và
      hệ quả cần thiết; luôn dùng tiếng Việt chuẩn.
    - reasoning giải thích ngắn cách facts dẫn tới đáp án; không bịa
      citation hoặc nói "dữ liệu được cung cấp".
    - required_claims có ít nhất một phần tử, liệt kê các ý bắt buộc
      trong reference_answer, không lặp nội dung và trỏ đúng fact_id.
      Với reasoning_hops>1,
      claims phải bao phủ mọi fact. Không tạo claim cho phần chưa thể kết
      luận của partially_answerable.

    Tự kiểm tra query, reference_answer, reasoning, required_claims, answerability
    và reasoning_hops nhất quán. Trả đúng một kết quả cho mỗi task,
    giữ nguyên task_id và thứ tự.

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
