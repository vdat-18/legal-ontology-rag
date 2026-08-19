class FactContext {
  fact_id string
  statement string
  conduct string?
  subject string?
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

    Với mỗi task, tạo một câu hỏi tiếng Việt về xử phạt giao thông, đáp án
    tham chiếu, lập luận và các claim bắt buộc. Chỉ dùng facts của chính task;
    trả đúng một kết quả cho mỗi task, giữ nguyên task_id và thứ tự.

    {% for item in items %}
    ===== TASK {{ item.task_id }} =====
    Cấu hình:
    - Persona: {{ item.persona_description }}
    - Loại câu hỏi: {{ item.query_type_description }}
    - Answerability: {{ item.answerability }}
    - Bundle: {{ item.bundle_kind }}; reasoning_hops={{ item.reasoning_hops }}
    {% if item.bridge %}- Bridge bắt buộc: {{ item.bridge }}{% endif %}
    - Form: {{ item.query_form }}; style={{ item.query_style }};
      length={{ item.query_length }}{% if item.query_length == "short" %} (5-10 từ){% elif item.query_length == "medium" %} (10-20 từ){% elif item.query_length == "long" %} (>20 từ){% endif %}
    {% if item.as_of_date %}- Thời điểm áp dụng: {{ item.as_of_date }}{% endif %}

    query_vi phải tự nhiên theo persona và form, không mang văn phong hành
    chính. Trừ persona luật sư/cán bộ xử phạt, không trích số Điều/khoản/điểm.
    Áp dụng chính xác query_style sau; ví dụ chỉ minh họa cách gõ, không được
    sao chép nội dung:
    {% if item.query_style == "standard" %}
    - standard: đủ dấu, đúng chính tả, câu hoàn chỉnh và có dấu hỏi.
      Ví dụ: "Xe ô tô vượt đèn đỏ thì bị phạt bao nhiêu tiền?"
    {% elif item.query_style == "no_diacritics" %}
    - no_diacritics: hoàn toàn không dấu, không tạo thêm lỗi chính tả.
      Ví dụ: "xe oto vuot den do thi bi phat bao nhieu tien"
    {% elif item.query_style == "misspelled" %}
    - misspelled: vẫn có dấu nhưng dùng vài lỗi/viết tắt chat tự nhiên; không
      biến thành no_diacritics. Ví dụ: "oto vượt đèn đỏ phạt mấy tiền v ạ"
    {% elif item.query_style == "keyword_style" %}
    - keyword_style: chỉ từ khóa cốt lõi, không thành câu và không dấu hỏi.
      Ví dụ: "oto vuot den do muc phat"
    {% else %}
    - Thực hiện đúng mô tả style={{ item.query_style }}.
    {% endif %}

    Ràng buộc nội dung:
    - Chỉ dùng facts bên dưới; không bổ sung kiến thức, đổi evidence hoặc tạo
      chunk_id. Mỗi claim phải trỏ đúng fact_id và tập con evidence_chunk_ids
      của fact đó.
    - Câu hỏi, đáp án và lập luận phải thực sự cần đủ reasoning_hops; nếu >1,
      required_claims phải bao phủ mọi fact và sử dụng bridge nếu có.
    - Với partially_answerable, trả lời phần có căn cứ rồi hỏi thêm dữ kiện cụ
      thể còn thiếu; không nói về "dữ liệu được cung cấp" và không biến thành
      answerable hoàn toàn.
    - reference_answer_vi và reasoning_vi dùng tiếng Việt chuẩn, rõ ràng như
      tư vấn pháp lý; không dùng thuật ngữ nội bộ hoặc từ tiếng Anh "fact".
    - Tự kiểm tra tính nhất quán giữa query, answer, reasoning và claims trước
      khi trả lời.

    Fact pháp lý:
    {% for fact in item.facts %}
    [{{ fact.fact_id }}] {{ fact.statement }}
    {% if fact.conduct %}Hành vi: {{ fact.conduct }}{% endif %}
    {% if fact.subject %}Đối tượng: {{ fact.subject }}{% endif %}
    Điều khoản: {{ fact.provision_path }} ({{ fact.document_number }})
    Phạt tiền: {{ fact.monetary_penalty }}
    Trừ điểm: {{ fact.license_points_deducted }}
    Hình phạt bổ sung: {{ fact.additional_penalties }}
    Biện pháp khắc phục: {{ fact.remedial_measures }}
    Hiệu lực: {{ fact.valid_from }} đến {{ fact.valid_to }}
    Chunk căn cứ: {{ fact.evidence_chunk_ids }}
    ---
    {% endfor %}
    ===== END TASK {{ item.task_id }} =====
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

    Chỉ đánh giá bề mặt câu hỏi theo style={{ query_style }},
    length={{ query_length }}, form={{ query_form }}; không chấm nội dung pháp
    lý. Nếu chưa đạt, viết lại mà không đổi ý nghĩa. Nếu đạt, giữ nguyên.

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

    Với mỗi task, áp dụng đúng một phép biến đổi mà không đổi căn cứ pháp lý
    hoặc kết luận. Giữ nguyên task_id, thứ tự và trả đúng một kết quả mỗi task.

    - Concretizing: thêm tình tiết cụ thể vào tình huống
    - Reasoning: buộc suy luận từ quy định thay vì tra cứu trực tiếp
    - Hypothetical: đặt giả định "nếu ... thì"

    {% for item in items %}
    ===== TASK {{ item.task_id }} =====
    Phép áp dụng: {{ item.operation }}

    Câu hỏi gốc:
    {{ item.query_vi }}

    Đáp án gốc:
    {{ item.reference_answer_vi }}

    Lập luận gốc:
    {{ item.reasoning_vi }}

    Giữ nguyên cách gõ của query gốc (dấu, lỗi/viết tắt, câu hay từ khóa); chỉ
    đổi tình huống theo operation. Giữ length={{ item.query_length }}
    {% if item.query_length == "short" %}(khoảng 5-10 từ){% elif item.query_length == "medium" %}(khoảng 10-20 từ){% elif item.query_length == "long" %}(trên 20 từ){% endif %}
    và không kéo dài thành đoạn văn. Concretizing chỉ thêm đúng một chi tiết.
    ===== END TASK {{ item.task_id }} =====
    {% endfor %}

    Cập nhật answer và reasoning cho nhất quán với query mới, bằng tiếng Việt
    chuẩn; không thêm căn cứ, thuật ngữ nội bộ hoặc cụm "dữ liệu được cung cấp".

    {{ ctx.output_format }}
  "#
}
