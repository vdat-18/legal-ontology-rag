class ScenarioPlanningFact {
  fact_id string
  statement string
  document_number string
  valid_from string?
  valid_to string?
}

class ScenarioQueryTypeCandidate {
  query_type_id string
  description_vi string
}

class ScenarioPlanningBatchItem {
  bundle_id string
  bundle_kind string
  bridge string[]
  facts ScenarioPlanningFact[]
  query_type_candidates ScenarioQueryTypeCandidate[]
  candidate_as_of_dates string[]
}

class PlannedScenarioQueryType {
  query_type_id string @description(#"Phải trùng với một query_type_id được cung cấp cho bundle."#)
  as_of_date string? @description(#"
    Null nếu loại câu hỏi không cần một thời điểm cụ thể. Nếu có, phải sao
    chép chính xác một ngày trong candidate_as_of_dates.
  "#)
}

class ScenarioPlanningBatchResult {
  bundle_id string @description(#"Phải trùng với một bundle_id trong input."#)
  supported_query_types PlannedScenarioQueryType[]
}

function PlanScenarioBundles(items: ScenarioPlanningBatchItem[]) -> ScenarioPlanningBatchResult[] {
  client OpenAICompatibleProvider
  prompt #"
    {{ _.role("user") }}

    Nhiệm vụ: xác định loại câu hỏi nào thực sự được mỗi evidence bundle hỗ
    trợ trong benchmark xử lý vi phạm giao thông đường bộ. Chưa viết câu hỏi
    hoặc đáp án. Xử lý độc lập từng bundle và giữ nguyên bundle_id.

    Chỉ chọn query type khi các statement đủ căn cứ để tạo câu hỏi đúng mô tả
    mà không bổ sung kiến thức. Với multi_hop hoặc temporal, câu hỏi phải thực
    sự cần mọi fact và bridge; nếu một fact đã đủ thì không chọn loại yêu cầu
    kết hợp. Không chọn chỉ vì trùng từ, cùng văn bản hoặc cùng chủ đề chung.

    as_of_date:
    - Null nếu câu hỏi không cần một ngày cụ thể hoặc là so sánh trước/sau.
    - Nếu cần hỏi quy định áp dụng tại một thời điểm, sao chép đúng một ngày
      trong candidate_as_of_dates và bảo đảm facts có thể trả lời tại ngày đó.
    - Không tự tạo ngày mới.

    Mỗi query_type_id chỉ xuất hiện tối đa một lần. Không có loại nào được hỗ
    trợ thì trả supported_query_types=[].

    {% for item in items %}
    ===== BUNDLE {{ item.bundle_id }} =====
    Kind: {{ item.bundle_kind }}
    {% if item.bridge %}Bridge: {{ item.bridge }}{% endif %}
    Candidate dates: {{ item.candidate_as_of_dates }}

    Query types được phép:
    {% for query_type in item.query_type_candidates %}
    - {{ query_type.query_type_id }}: {{ query_type.description_vi }}
    {% endfor %}

    Facts:
    {% for fact in item.facts %}
    [{{ fact.fact_id }}] {{ fact.statement }}
    Văn bản: {{ fact.document_number }}
    Hiệu lực: {{ fact.valid_from }} đến {{ fact.valid_to }}
    ---
    {% endfor %}
    ===== END BUNDLE {{ item.bundle_id }} =====
    {% endfor %}

    {{ ctx.output_format }}
  "#
}
