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

## Contract chung của prompt

Mọi prompt dùng cùng phạm vi: quy định về hành vi/nghĩa vụ, chủ thể và phương
tiện, điều kiện/ngoại lệ, chế tài, trừ điểm hoặc tước quyền sử dụng, biện pháp
khắc phục, thẩm quyền, thủ tục và hiệu lực liên quan trực tiếp đến xử lý vi
phạm giao thông đường bộ tại Việt Nam.

Mỗi stage chỉ làm đúng một nhiệm vụ: screening chọn target; metadata giải
thông tin toàn văn bản; extraction tạo fact; relation nối các fact hoàn chỉnh;
scenario sampling cân bằng hình thức câu hỏi; generation tạo mẫu hỏi đáp. Tất
cả stage chỉ dùng input được cung cấp, không bổ sung kiến thức pháp luật bên
ngoài và giữ nguyên ID của từng batch item. Chỉ dẫn ổn định về
bài toán, domain và contract được tách khỏi source text của từng request.

## Tổng quan

```text
Document chunks
    |
    v
1A. Screen target chunks
    |
    v
1B. Resolve document metadata
    |
    v
2. Extract legal facts
    |
    v
3. Assemble corpus facts
    |
    v
4. Build evidence graph and bundles
    |
    v
5. Plan and sample scenarios
    |
    v
6. Generate candidates
    |
    v
7. Export evaluation ground truth
```

## 1A. Sàng lọc target chunk

### Input

Mỗi chunk đã được parser/chunker tạo, gồm `chunk_id`, source text và structural
context.

### Xử lý

LLM quyết định chunk có nội dung pháp lý đáng trích xuất cho domain hay không.
Screening chạy theo batch và trả kết quả theo `chunk_id`. Đây là quyết định
ngữ nghĩa của LLM, không phải keyword rule. Model chỉ giữ nội dung trực tiếp
giúp xác định hành vi/tình huống vi phạm, phạm vi áp dụng hoặc cách phát hiện,
ngăn chặn và xử lý. Nội dung quản lý giao thông nói chung hoặc thủ tục hành
chính độc lập không đủ; recall chỉ được ưu tiên sau khi vượt qua boundary này.
Danh mục/tên biểu mẫu, trường điền, dòng giao nhận và ô ký không phải legal
fact nếu target không chứa nghĩa vụ hoặc hướng dẫn pháp lý cụ thể.
Runtime log hiển thị tổng batch, `batch_size`, `max_concurrency`, thời điểm
mỗi batch bắt đầu và tiến độ sau khi batch đã checkpoint thành công.
Mỗi response checkpoint ngay các verdict có `chunk_id` khớp duy nhất, theo
thứ tự chunk gốc; retry tiếp theo chỉ gửi lại các chunk còn thiếu hoặc sai ID.

Chỉ chunk được chấp nhận mới trở thành `Target` ở bước trích fact. Tuy nhiên,
mọi chunk gốc vẫn có thể được dùng làm context liền kề cho một target.

### Output

`*.chunk_screening.jsonl`, với quyết định domain, cờ metadata cấp văn bản và
lý do của model cho từng chunk. Mỗi verdict lưu riêng prompt version, model và
screening run để có thể chạy lại chọn lọc mà vẫn giữ provenance. Cờ metadata
chỉ định source text cho Step 1B; code không tìm ngày hoặc số hiệu bằng
regex/keyword.

Khi chỉ một số verdict sai, truyền từng `--rescreen-chunk-id` hoặc một JSON
array qua `--rescreen-chunk-ids`. LLM chỉ nhận các chunk được chọn; code thay
verdict tại đúng vị trí cũ và giữ nguyên các chunk khác. Step 1B được dùng lại
nếu tập metadata candidate không đổi, và chỉ chạy lại khi selection metadata
thực sự thay đổi.

## 1B. Lấy metadata cấp văn bản

### Input

Các chunk có `record_scope == "root"`, được sắp theo:

```text
(chunk_index, chunk_id)
```

Sau đó code chỉ lấy các chunk được Step 1A đánh dấu
`has_document_metadata=true`, giữ đúng thứ tự tài liệu. Không có cửa sổ
đầu/cuối hoặc fallback semantic trong code.

### Xử lý

LLM chỉ đọc các đoạn metadata đã được chọn trên toàn văn để xác định metadata
của chính văn bản đang xử lý, không lấy metadata của văn bản được dẫn chiếu.
Cách này vẫn tìm được điều khoản hiệu lực nằm trước một phụ lục dài:

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

Resume được checkpoint theo batch target. Nếu tiến trình dừng sau khi facts đã
được ghi nhưng trước progress, lần replay thay thế toàn bộ facts có cùng
`source_chunk_id` trước khi ghi kết quả mới, không tích lũy hai phiên bản LLM.

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

Dense retrieval embedding toàn bộ statement theo batch 100, tối đa 10 batch
đồng thời. Mỗi batch retry độc lập theo cấu hình embedding và checkpoint ngay
thành file vector riêng. Khi resume, chỉ batch còn thiếu được gọi lại; vectors
được ghép về đúng thứ tự corpus trước khi dense search bắt đầu. Candidate-pair
cache hoàn tất vẫn bỏ qua toàn bộ embedding như trước.

Sau khi chuẩn bị corpus, dense search và BM25L search chạy song song trên toàn
bộ fact. RRF chỉ bắt đầu khi cả hai nhánh đã hoàn tất, sau đó giữ top candidate
cho từng fact và loại cặp đối xứng trùng nhau.

LLM đọc `fact_id`, `statement`, `document_number`, `valid_from` và `valid_to`
để quyết định hai fact có thực sự cần nhau cho một câu hỏi grounded hay không.
Nếu có, LLM chọn trực tiếp `bundle_kind` là `multi_hop` hoặc `temporal` và viết
một `bridge` ngắn giải thích nội dung nối hai fact. Không còn taxonomy loại
quan hệ hoặc hướng `from_fact_id`/`to_fact_id`.

Evidence chunk IDs không được gửi vì ID không mang nội dung ngữ nghĩa; chúng
vẫn được giữ trong fact để tạo gold chunks. Retrieval score hoặc keyword
không phải quyết định cuối.

Relation confirmation checkpoint một batch hoàn chỉnh mỗi lần. Khi replay,
kết quả cũ của các pair trong batch được thay thế trước khi progress được đánh
dấu. Candidate-pair cache và relation progress đều gắn run ID.

Các relation đã xác nhận được chuyển thành:

- `single_hop` bundle: một fact;
- `multi_hop` bundle: hai fact bổ sung trực tiếp cho cùng câu hỏi;
- `temporal` bundle: hai fact thể hiện thay đổi hoặc chế độ hiệu lực khác nhau.

Để tạo multi-hop sâu hơn mà không làm pair confirmation phát sinh thêm chi
phí mặc định, chạy Step 4B riêng:

```powershell
uv run legal-build-multi-hop-3-bundles `
  data/benchmark_synthesis/legal_facts.jsonl `
  data/benchmark_synthesis/evidence_relations.jsonl `
  data/benchmark_synthesis/multi_hop_3_bundles.jsonl `
  data/benchmark_synthesis/multi_hop_3_progress.jsonl `
  data/benchmark_synthesis/scenarios.jsonl `
  --candidate-cache data/benchmark_synthesis/multi_hop_3_candidates.json `
  --personas data/config/groundtruth/personas.json `
  --scenario-distribution data/config/groundtruth/scenario_distribution.json `
  --total-samples 1000 `
  --synthesis-config data/config/groundtruth/synthesis_v1.json
```

Step 4B tạo các đường nối ba fact `A-B-C` chỉ từ relation `multi_hop` đã
xác nhận. Candidate được lấy round-robin theo fact trung tâm và giới hạn bởi
config để node có bậc lớn không chiếm toàn bộ ngân sách. LLM sau đó kiểm tra
phép loại bỏ: bundle chỉ được giữ khi bỏ bất kỳ fact nào cũng khiến câu trả
lời không còn đầy đủ. Progress checkpoint theo candidate và hỗ trợ selective
rerun bằng `--reconfirm-candidate-id` hoặc `--reconfirm-candidate-ids`.

Bundle được xác nhận vẫn có `bundle_kind=multi_hop`, nhưng chứa ba `fact_ids`
và tạo scenario có `reasoning_hops=3`. Scenario sampling cân bằng bundle
multi-hop hai và ba facts trong phạm vi bundle chưa dùng; nếu bundle ba facts
ít, chúng không bị lặp chỉ để ép tỷ lệ bằng nhau.

Để mở rộng tiếp thành bốn facts, chạy Step 4C sau khi Step 4B đã hoàn tất:

```powershell
uv run legal-build-multi-hop-4-bundles `
  data/benchmark_synthesis/legal_facts.jsonl `
  data/benchmark_synthesis/evidence_relations.jsonl `
  data/benchmark_synthesis/multi_hop_3_bundles.jsonl `
  data/benchmark_synthesis/multi_hop_4_bundles.jsonl `
  data/benchmark_synthesis/multi_hop_4_progress.jsonl `
  data/benchmark_synthesis/scenarios.jsonl `
  --candidate-cache data/benchmark_synthesis/multi_hop_4_candidates.json `
  --personas data/config/groundtruth/personas.json `
  --scenario-distribution data/config/groundtruth/scenario_distribution.json `
  --total-samples 1000 `
  --synthesis-config data/config/groundtruth/synthesis_v1.json
```

Step 4C không chạy lại embedding, BM25L, dense search hoặc RRF. Mỗi bundle
ba facts đã xác nhận được nối thêm một fact qua relation `multi_hop` hiện có;
candidate được lấy round-robin theo bundle nguồn và loại trùng theo tập bốn
fact. LLM nhận đủ bốn statements, bundle ba facts nguồn và mọi pair connection
đã xác nhận giữa bốn facts. Bundle chỉ được giữ khi fact thứ tư tạo một bước
phụ thuộc thực sự và bỏ bất kỳ fact nào cũng làm câu trả lời không còn đầy đủ.

Candidate, progress và bundle bốn facts được lưu riêng. Lệnh resume chỉ xử lý
candidate chưa có progress; selective rerun dùng `--reconfirm-candidate-id`
hoặc `--reconfirm-candidate-ids`. Scenario sampling nhận cả bundle hai, ba và
bốn facts, rồi cân bằng theo `reasoning_hops` trong phạm vi bundle chưa dùng.

Để mở rộng thành năm facts, chạy Step 4D sau khi Step 4C đã hoàn tất:

```powershell
uv run legal-build-multi-hop-5-bundles `
  data/benchmark_synthesis/legal_facts.jsonl `
  data/benchmark_synthesis/evidence_relations.jsonl `
  data/benchmark_synthesis/multi_hop_3_bundles.jsonl `
  data/benchmark_synthesis/multi_hop_4_bundles.jsonl `
  data/benchmark_synthesis/multi_hop_5_bundles.jsonl `
  data/benchmark_synthesis/multi_hop_5_progress.jsonl `
  data/benchmark_synthesis/scenarios.jsonl `
  --candidate-cache data/benchmark_synthesis/multi_hop_5_candidates.json `
  --personas data/config/groundtruth/personas.json `
  --scenario-distribution data/config/groundtruth/scenario_distribution.json `
  --total-samples 1000 `
  --synthesis-config data/config/groundtruth/synthesis_v1.json
```

Step 4D cũng không chạy lại retrieval. Mỗi bundle bốn facts đã xác nhận được
nối thêm đúng một fact qua relation `multi_hop`; candidate được lấy
round-robin và loại trùng theo tập năm facts. LLM chỉ giữ candidate khi fact
thứ năm tạo bước phụ thuộc thứ tư cho cùng một tình huống, không tạo nhánh
câu hỏi độc lập, và phép loại bỏ lần lượt cả năm facts đều thất bại.

Artifacts năm facts tách riêng và có cùng cơ chế fingerprint, retry,
checkpoint, resume và selective rerun như Step 4C. Khi ghi lại scenarios,
Step 4D đưa bundle multi-hop hai, ba, bốn và năm facts vào cùng pool; sampler
cân bằng theo số `fact_ids` mà không cần tỷ lệ hop hard-code.

`bridge` là một string do LLM tạo, không phải danh sách nhãn hoặc số Điều.
Không còn bundle kind `synthesis` và không suy luận loại bundle từ việc hai
fact cùng hay khác văn bản.

Bundle giữ `fact_ids`; gold chunks được suy ra trực tiếp từ `evidence` của các
fact trong bundle.

## 5. Lấy mẫu scenario

Mọi evidence bundle đã được Step 4 xác nhận đều có thể đi trực tiếp vào
sampling. Không còn query taxonomy, query-type candidate, LLM planner hoặc
`scenario_plans.jsonl`; vì vậy không có danh sách loại câu hỏi định trước làm
giới hạn độ bao phủ.

Code chỉ cân bằng các trục cấu hình không mang quyết định pháp lý:

- query form;
- persona;
- query style và query length;
- answerability;
- single-hop, multi-hop hoặc temporal.

Persona bao phủ người lái xe cá nhân và chuyên nghiệp, tài xế công nghệ,
chủ xe, doanh nghiệp vận tải, phụ huynh, người học lái, người xử lý phạt
nguội, người dân phản ánh vi phạm, cán bộ và luật sư. Query style bao gồm hỏi
trực tiếp, hội thoại, lịch sự, trang trọng, không dấu, lỗi gõ và từ khóa. Các
giá trị trong từng trục được luân phiên cân bằng; ba bundle kind có trọng số
bằng nhau khi đều có dữ liệu.

Code không quyết định statement phải sinh loại câu hỏi nào. Mục tiêu câu hỏi
được LLM chọn trực tiếp trong Step 6 từ các facts của bundle.

Output của bước này:

```text
data/benchmark_synthesis/scenarios.jsonl
```

`scenarios.jsonl` chứa `fact_ids`, `bridge` và `gold_chunk_ids`, chưa chứa câu
hỏi hay đáp án hoàn chỉnh.

## 6. Sinh candidate

### Input

LLM nhận các thuộc tính bề mặt của scenario và mỗi fact dưới dạng tối giản:

```text
Scenario:
- persona_description
- query_style, query_length, query_form
- answerability
- bundle_kind, bridge, reasoning_hops

Mỗi fact:
- fact_id
- statement
- document_number
- valid_from / valid_to
```

`gold_chunk_ids` và evidence của fact không được gửi cho LLM. Chunk ID chỉ là
provenance kỹ thuật, không giúp model hiểu nội dung pháp lý.

### Output

LLM sinh:

- `query`;
- `reference_answer`;
- `reasoning`;
- `required_claims`, mỗi claim chỉ gồm nội dung `claim` và `fact_id` hỗ trợ.

LLM tự chọn một câu hỏi pháp lý hữu ích được facts hỗ trợ, không nhận taxonomy
hoặc question intent trung gian. Prompt buộc mọi con số, điều kiện, ngoại lệ
và kết luận phải có trong facts.
Với multi-hop, mọi fact phải cần thiết cho câu hỏi và đáp án. Với
`partially_answerable`, đáp án chỉ nêu phần có căn cứ và phần chưa thể kết
luận; không tự điền thông tin còn thiếu.

Không có critique gate hoặc verification gate sau generation. Pipeline tin
typed output của LLM và ghi candidate trực tiếp. Với mỗi claim, code lấy toàn
bộ `LegalFact.evidence` của `fact_id` tương ứng và gắn thành provenance; LLM
không tự chọn hoặc tạo chunk ID. Nếu model trả `fact_id` không thuộc scenario,
native lookup error được giữ nguyên thay vì thêm rule xác minh tùy chỉnh.

Step 6 chỉ có một lần generation. Không còn evolution vì việc viết lại query,
answer và reasoning sau khi claims đã cố định có thể làm chúng lệch nhau; độ
đa dạng bề mặt đã được Step 5 điều khiển bằng persona, style, form và length.

Output:

```text
data/benchmark_synthesis/candidates/single_turn.jsonl
data/benchmark_synthesis/candidates/single_turn.progress.jsonl
```

Candidate progress được checkpoint một lần cho mỗi LLM batch. Sidecar
`single_turn.run.json` fingerprint facts, scenarios, mô tả persona, số
attempt, candidate config, prompt, endpoint và model; progress của run khác
không được resume.

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

Các metadata như scenario, persona, answerability, reasoning, claims và source
fact IDs vẫn được giữ để phân tích theo lát cắt.

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
- Sidecar `*.run.json` bảo vệ Steps 1A, 1B, 2 và 6 khỏi trộn input, config,
  prompt, model hoặc upstream run khác nhau. Step 4 dùng candidate/relation
  run ID trực tiếp trong cache và progress.
- Cấu hình synthesis nằm tại
  `data/config/groundtruth/synthesis_v1.json`.
- Synthesis config hiện dùng schema `5.2.0`; mỗi LLM stage khai báo rõ
  `reasoning_effort`, hiện đặt là `medium`.
- `LegalFact 3.0.0`, `EvidenceRelation 3.1.0` và `Scenario 4.0.0` vẫn dùng lại
  được nếu đã được sinh bởi Steps 1-5 hiện tại.
- `Candidate 6.0.0`, `CandidateProgress 5.0.0` và `GroundTruth 5.0.0` không
  tương thích với artifact Step 6-7 cũ; cần chạy lại Step 6 và Step 7, không
  sửa tay hoặc tự động migrate.
- Candidate artifacts được tạo trước khi Step 6 có fingerprint sidecar không
  được resume; chạy lại Step 6 để tạo một run nhất quán.

## Giới hạn hiện tại

Pipeline tạo ground truth tốt để chạy metric, nhưng chất lượng metric vẫn phụ
thuộc chất lượng semantic output của LLM vì các semantic verification gate đã
được chủ động loại bỏ. Cần audit mẫu thủ công hoặc một quy trình đánh giá độc
lập nếu muốn đo độ chính xác tuyệt đối của chính ground truth.

Phạm vi hiện tại là single-turn. Multi-turn, hard negatives, leakage scoring,
deduplication và dataset split không nằm trong pipeline này.
