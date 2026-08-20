# Ground-Truth Synthesis Pipeline

## Mục tiêu

Pipeline sinh bộ câu hỏi và đáp án chuẩn cho RAG trong domain xử lý vi phạm
giao thông đường bộ Việt Nam. Output cuối hỗ trợ:

- retrieval: Precision@k, Recall@k và các metric dựa trên gold chunk;
- generation: faithfulness, correctness và các metric dựa trên reference answer;
- end-to-end: đánh giá toàn bộ luồng retrieve rồi generate.

Luồng hiện tại ưu tiên dữ liệu đơn giản và rõ provenance. LLM chịu trách nhiệm
hiểu ngữ nghĩa pháp lý; code chỉ điều phối, gắn metadata, tạo định danh và duy
trì quan hệ giữa fact với chunk nguồn. Không dùng regex, keyword matching,
handcrafted score hoặc exact-span matching để quyết định ngữ nghĩa.

## Tổng quan

```text
Document chunks
    |
    +-- 1A. Screen target chunks
    +-- 1B. Resolve document metadata
    |
    `-- 2. Extract legal facts
            |
            `-- 3. Assemble corpus facts
                    |
                    `-- 4. Build evidence graph and bundles
                            |
                            `-- 5. Sample scenarios
                                    |
                                    `-- 6. Generate/evolve candidates
                                            |
                                            `-- 7. Export evaluation ground truth
```

## 1A. Sàng lọc target chunk

### Input

Mỗi chunk đã được parser/chunker tạo, gồm `chunk_id`, source text và structural
context.

### Xử lý

LLM quyết định chunk có nội dung pháp lý đáng trích xuất cho domain hay không.
Screening chạy theo batch và trả kết quả theo `chunk_id`. Đây là quyết định
ngữ nghĩa của LLM, không phải keyword rule.

Chỉ chunk được chấp nhận mới trở thành `Target` ở bước trích fact. Tuy nhiên,
mọi chunk gốc vẫn có thể được dùng làm context liền kề cho một target.

### Output

`*.chunk_screening.jsonl`, với quyết định và lý do của model cho từng chunk.

## 1B. Lấy metadata cấp văn bản

### Input

Các chunk có `record_scope == "root"`, được sắp theo:

```text
(chunk_index, chunk_id)
```

Sau đó code lấy đúng thứ tự:

- 5 chunk đầu của thân văn bản;
- 5 chunk cuối của thân văn bản.

Các số lượng này nằm trong typed config, không hardcode trong logic:
`stages.document_metadata.opening_chunk_count` và
`stages.document_metadata.closing_chunk_count`.

### Xử lý

LLM đọc hai nhóm chunk để xác định metadata của chính văn bản đang xử lý,
không lấy metadata của văn bản được dẫn chiếu:

- `document_number`;
- `valid_from`;
- `valid_to`.

Code tin typed output của LLM và chỉ parse ngày. Không kiểm tra `quoted_span`,
không so khớp câu chữ với chunk.

### Output

`*.document_metadata.jsonl`. Metadata này được dùng chung cho mọi fact của
văn bản, tránh mỗi target chunk tự đoán một giá trị khác nhau.

## 2. Trích xuất legal fact

### Input của mỗi LLM call

Một reading window theo đúng thứ tự tài liệu:

```text
Preceding: tối đa 2 chunk liền trước
Target:    1 chunk đã qua screening
Following: tối đa 1 chunk liền sau
```

Các neighbor được đi theo `previous_chunk_id` và `next_chunk_id` đã persist.
Context chỉ giúp hoàn chỉnh quy định bị chia chunk; fact vẫn phải có nội dung
pháp lý trực tiếp trong `Target`.

### Output của LLM

LLM chỉ trả phần ngữ nghĩa cần thiết:

```json
{
  "statement": "Mệnh đề pháp lý nguyên tử, đầy đủ và tự hiểu được.",
  "evidence": [
    {"chunk_id": "chunk-1"}
  ]
}
```

`statement` là nguồn sự thật duy nhất. Nó phải tự chứa đầy đủ các thành phần
có căn cứ và cần thiết, ví dụ chủ thể, hành vi hoặc nghĩa vụ, điều kiện, ngoại
lệ, mức phạt, hình thức xử phạt, biện pháp, thẩm quyền hoặc thủ tục.

Không còn các field tách riêng như `conduct`, `subject`, `conditions`,
`monetary_penalty`, `license_points_deducted`, `additional_penalties`,
`remedial_measures`, `provision_path` hoặc `quoted_span`. Các field này trùng
lặp với `statement`, làm schema dài và khiến downstream phụ thuộc vào nhiều
biểu diễn của cùng một nội dung.

### Phần code bổ sung

Code gắn metadata từ bước 1B và tạo persisted `LegalFact`:

```json
{
  "schema_version": "3.0.0",
  "fact_id": "...",
  "source_chunk_id": "target-chunk-id",
  "statement": "Mệnh đề pháp lý nguyên tử, đầy đủ và tự hiểu được.",
  "valid_from": "2025-01-01",
  "valid_to": null,
  "document_number": "168/2024/NĐ-CP",
  "evidence": [
    {"chunk_id": "chunk-1"}
  ]
}
```

`fact_id` là content-derived identity được hash từ:

- `statement`;
- `document_number`;
- `valid_from` và `valid_to`;
- danh sách `evidence` theo thứ tự.

Không còn `fact_content_sha256`. Một định danh duy nhất đã đủ biểu diễn nội
dung và provenance của fact.

Code chỉ yêu cầu evidence không rỗng và `chunk_id` thuộc reading window. Code
không xác minh lại ý nghĩa của statement, không yêu cầu trích dẫn nguyên văn
và không dùng exact-span gate.

### Output

Theo từng văn bản:

```text
data/processed/groundtruth/<document>.legal_facts.jsonl
data/processed/groundtruth/<document>.extraction_progress.jsonl
data/processed/groundtruth/<document>.legal_facts.run.json
```

## 3. Đóng gói fact toàn corpus

`legal-assemble-groundtruth` đọc `legal_facts.jsonl` của từng văn bản theo thứ
tự corpus config, nối nguyên trạng và ghi:

```text
data/benchmark_synthesis/legal_facts.jsonl
data/benchmark_synthesis/run_manifest.json
```

Bước này không validate, sửa nội dung hoặc deduplicate fact. Corpus fingerprint
được tính từ danh sách `fact_id` theo đúng thứ tự đóng gói.

## 4. Evidence graph và evidence bundle

Candidate retrieval đề xuất các cặp fact có khả năng liên quan bằng dense và
lexical retrieval trên duy nhất `statement`, rồi gộp thứ hạng bằng RRF. Đây
chỉ là bước đề xuất ứng viên.

LLM đọc `fact_id`, `statement`, `document_number`, `valid_from` và `valid_to`
để quyết định hai fact có thực sự liên quan hay không, loại quan hệ, hướng
quan hệ và `bridge`. Evidence chunk IDs không được gửi vì ID không mang nội
dung ngữ nghĩa; chúng vẫn được giữ trong fact để tạo gold chunks. Retrieval
score hoặc keyword không phải quyết định cuối.

Relation taxonomy dành cho các statement tự đầy đủ:

- `complementary_rule`: quy tắc nền và quy tắc bổ sung cho cùng tình huống;
- `rule_to_exception`: quy định chung và ngoại lệ;
- `rule_to_authority`: quy định và thẩm quyền xử phạt;
- `amended_by`: quy định cũ và quy định sửa đổi/thay thế;
- `procedure_sequence`: hai bước liên tiếp trong một thủ tục.

Các relation đã xác nhận được chuyển thành:

- `single_hop` bundle: một fact;
- `multi_hop` bundle: hai fact có quan hệ và bridge;
- `temporal` bundle: hai fact nối bằng quan hệ sửa đổi.

Không còn bundle kind `synthesis`. Mọi relation hai fact không phải sửa đổi
đều là `multi_hop`, không suy luận loại bundle từ việc hai fact cùng hay khác
văn bản.

Bundle giữ `fact_ids`; gold chunks được suy ra trực tiếp từ `evidence` của các
fact trong bundle.

## 5. Lấy mẫu scenario

Scenario sampling kết hợp evidence bundle với taxonomy và các trục phân bố:

- query type và query form;
- persona;
- query style và query length;
- answerability;
- single-hop hoặc multi-hop;
- `as_of_date` nếu các fact có khoảng hiệu lực chung.

Eligibility hiện chỉ dựa trên cấu trúc bundle và số lượng fact tối thiểu của
query type. Nó không còn kiểm tra các field chuyên biệt đã bị xóa.

Output `scenarios.jsonl` chứa `fact_ids`, `bridge` và `gold_chunk_ids`, chưa
chứa câu hỏi hay đáp án hoàn chỉnh.

## 6. Sinh candidate

### Input

LLM nhận scenario và mỗi fact dưới dạng tối giản:

```text
fact_id
statement
document_number
valid_from / valid_to
evidence_chunk_ids
```

### Output

LLM sinh:

- `query_vi`;
- `reference_answer_vi`;
- `reasoning_vi`;
- `required_claims`, mỗi claim trỏ về `fact_id` và evidence chunk IDs.

Nếu có evolution, LLM biến đổi query, answer và reasoning theo operation được
cấu hình. Claims và provenance gốc vẫn được giữ.

Không có critique gate hoặc verification gate sau generation. Pipeline tin
typed output của LLM và ghi candidate trực tiếp. Code chỉ giữ evidence IDs
thực sự thuộc fact tương ứng để provenance không trỏ sang chunk ngoài fact.

Output:

```text
data/benchmark_synthesis/candidates/single_turn.jsonl
data/benchmark_synthesis/candidates/single_turn.progress.jsonl
```

## 7. Export ground truth để evaluate

`legal-export-groundtruth` map candidate sang schema evaluator:

```json
{
  "question_id": "...",
  "question": "Câu hỏi cho RAG",
  "expected_answer": "Đáp án tham chiếu",
  "expected_chunk_ids": ["chunk-1", "chunk-2"]
}
```

Các metadata như scenario, query type, persona, answerability, reasoning,
claims, evolutions và source fact IDs vẫn được giữ để phân tích theo lát cắt.

Ý nghĩa khi evaluate:

- `question` là input cho retriever hoặc toàn bộ RAG;
- `expected_chunk_ids` là gold evidence cho Precision@k, Recall@k và các
  retrieval metric khác;
- `expected_answer` là reference cho correctness;
- retrieved context, generated answer, expected answer và required claims có
  thể được dùng cho faithfulness và các generation metric.

## Contract vận hành

- Mọi LLM work unit có retry hữu hạn và resume key ổn định.
- Các artifact thành công được ghi atomically; task thất bại không bị đánh dấu
  hoàn thành.
- Sidecar `*.run.json` bảo vệ resume khỏi trộn chunks, config, prompt, model
  hoặc upstream run khác nhau.
- Cấu hình synthesis nằm tại
  `data/config/groundtruth/synthesis_v1.json`.
- Các JSONL synthesis cũ theo schema trước `LegalFact 3.0.0` phải được sinh
  lại; không sửa tay hoặc tự động migrate chúng.

## Giới hạn hiện tại

Pipeline tạo ground truth tốt để chạy metric, nhưng chất lượng metric vẫn phụ
thuộc chất lượng semantic output của LLM vì các semantic verification gate đã
được chủ động loại bỏ. Cần audit mẫu thủ công hoặc một quy trình đánh giá độc
lập nếu muốn đo độ chính xác tuyệt đối của chính ground truth.

Phạm vi hiện tại là single-turn. Multi-turn, hard negatives, leakage scoring,
deduplication và dataset split không nằm trong pipeline này.
