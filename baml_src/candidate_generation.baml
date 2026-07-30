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
    - Cách gõ: {{ query_style }}
    - Độ dài: {{ query_length }}
    {% if as_of_date %}- Thời điểm áp dụng: {{ as_of_date }}{% endif %}

    Nguyên tắc bắt buộc:
    - Không tự tìm hoặc thay đổi evidence. Chỉ dùng đúng nội dung các fact
      dưới đây.
    - Mỗi required_claims phải khai báo đúng fact_id nó phái sinh từ đó, và
      evidence_chunk_ids chỉ được lấy từ evidence_chunk_ids của đúng fact đó.
    - Không dùng kiến thức pháp luật sẵn có của bạn để bổ sung nội dung ngoài
      các fact dưới đây.
    - Nếu reasoning_hops lớn hơn 1, câu hỏi và đáp án phải thực sự cần đủ số
      fact đã cung cấp; required_claims phải bao phủ mọi fact.
    - Với partially_answerable, chỉ trả lời phần evidence hỗ trợ và nêu rõ phần
      còn thiếu. Không biến nó thành câu answerable hoàn toàn.
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
  operation: EvolutionOperation
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

    Trả về cả lập luận đã cập nhật để luôn nhất quán với câu hỏi và đáp án sau
    biến đổi.

    {{ ctx.output_format }}
  "#
}
