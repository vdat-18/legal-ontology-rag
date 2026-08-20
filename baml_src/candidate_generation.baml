class FactContext {
  fact_id string
  statement string
  valid_from string?
  valid_to string?
  document_number string
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

class CandidateGenerationBatchInput {
  task_id string
  facts FactContext[]
  persona_description string
  query_style string
  query_length string
  query_form string
  query_type_description string
  answerability string
  bundle_kind string
  bridge string[]
  reasoning_hops int
  as_of_date string?
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
    - Mục tiêu câu hỏi: {{ item.query_type_description }}
    - Answerability: {{ item.answerability }}
    - Evidence shape: {{ item.bundle_kind }}, reasoning_hops={{ item.reasoning_hops }}
    {% if item.bridge %}- Nội dung nối các fact: {{ item.bridge }}{% endif %}
    - Form={{ item.query_form }}, style={{ item.query_style }}, length={{ item.query_length }}
    {% if item.as_of_date %}- Thời điểm phải áp dụng: {{ item.as_of_date }}{% endif %}

    Grounding và coverage:
    - Mọi kết luận, con số, điều kiện, ngoại lệ và mốc thời gian phải có trong
      facts. Không thêm kiến thức pháp luật, tình tiết làm đổi kết luận, số
      Điều/khoản/điểm hoặc chunk_id không được cung cấp.
    - Mục tiêu câu hỏi phải được diễn giải trong giới hạn facts; không ép đúng
      loại bằng cách bịa nội dung mà facts không hỗ trợ.
    - Với reasoning_hops=1, một fact phải đủ để trả lời. Với reasoning_hops>1,
      câu hỏi và đáp án phải thực sự cần mọi fact, required_claims bao phủ mọi
      fact và thể hiện nội dung bridge; không ghép nhiều fact chỉ để làm dài.
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
      standard: tiếng Việt đủ dấu, đúng chính tả, câu hoàn chỉnh có dấu hỏi.
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
      chỉ dùng evidence_chunk_ids thuộc fact đó. Không tạo claim cho phần chưa
      thể kết luận của partially_answerable và không lặp claim cùng nội dung.

    Trước khi trả lời, tự kiểm tra query, answer, reasoning, claims, số hop,
    answerability và evidence nhất quán với nhau.

    Facts:
    {% for fact in item.facts %}
    [{{ fact.fact_id }}] {{ fact.statement }}
    Văn bản: {{ fact.document_number }}
    Hiệu lực: {{ fact.valid_from }} đến {{ fact.valid_to }}
    Chunk căn cứ: {{ fact.evidence_chunk_ids }}
    ---
    {% endfor %}
    ===== END TASK {{ item.task_id }} =====
    {% endfor %}

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

class CandidateEvolutionBatchInput {
  task_id string
  query_vi string
  reference_answer_vi string
  reasoning_vi string
  operation EvolutionOperation
  query_length string
}

class EvolvedCandidateBatchItem {
  task_id string
  candidate EvolvedCandidate
}

function EvolveCandidateBatch(
  items: CandidateEvolutionBatchInput[]
) -> EvolvedCandidateBatchItem[] {
  client OpenAICompatibleProvider
  prompt #"
    {{ _.role("user") }}

    Nhiệm vụ: biến đổi bề mặt một mẫu hỏi đáp về xử lý vi phạm giao thông
    đường bộ mà không thay đổi căn cứ, answerability hoặc kết luận pháp lý.
    Giữ nguyên task_id, thứ tự và trả đúng một kết quả mỗi task.

    - Concretizing: thêm đúng một chi tiết tình huống không làm thay đổi phạm
      vi áp dụng hoặc kết luận.
    - Reasoning: viết lại câu hỏi để người trả lời phải nối các dữ kiện/quy tắc
      đã có, không thêm kiến thức hay hop mới.
    - Hypothetical: chuyển cùng nội dung sang giả định "nếu... thì", giữ nguyên
      điều kiện và hệ quả.

    {% for item in items %}
    ===== TASK {{ item.task_id }} =====
    Phép áp dụng: {{ item.operation }}

    Câu hỏi gốc:
    {{ item.query_vi }}

    Đáp án gốc:
    {{ item.reference_answer_vi }}

    Lập luận gốc:
    {{ item.reasoning_vi }}

    Giữ kiểu gõ của query gốc: có/không dấu, lỗi/viết tắt, câu đầy đủ hay từ
    khóa. Giữ length={{ item.query_length }}
    {% if item.query_length == "short" %}(khoảng 5-10 từ){% elif item.query_length == "medium" %}(khoảng 10-20 từ){% elif item.query_length == "long" %}(trên 20 từ){% endif %}
    và không kéo dài thành đoạn văn.
    ===== END TASK {{ item.task_id }} =====
    {% endfor %}

    Cập nhật answer và reasoning cho nhất quán với query mới. Giữ nguyên mọi
    con số, điều kiện, ngoại lệ, mức độ chắc chắn và phần thông tin còn thiếu.
    Không thêm căn cứ, citation, hành vi, chế tài hoặc kết luận mới. Answer và
    reasoning luôn dùng tiếng Việt chuẩn, không dùng thuật ngữ nội bộ hay cụm
    "dữ liệu được cung cấp".

    {{ ctx.output_format }}
  "#
}
