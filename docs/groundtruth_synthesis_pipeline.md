# Ground-Truth Synthesis Pipeline

## Current hardened run contract

This section is authoritative where older implementation notes below still
describe retry-then-skip behavior.

- Every LLM work unit has bounded retry and a stable resume key: metadata,
  screening batch, extraction batch, relation pair, generation attempt, and
  evolution operation.
- `legal-prepare-groundtruth-document` runs chunk screening and document
  metadata resolution concurrently because the two stages read the same chunks
  but do not depend on each other.
- Chunk screening and fact extraction use bounded concurrency from
  `stages.chunk_screening.max_concurrency` and
  `stages.fact_extraction.max_concurrency`.
- Successful units are written atomically after completion. An exhausted unit
  is never marked complete; the CLI exits non-zero after preserving prior
  successful checkpoints.
- Document-stage resume artifacts are guarded by `<artifact>.run.json`
  fingerprints. A CLI refuses to resume legacy artifacts without sidecars, or
  artifacts produced from different chunks, config, prompt version, model, or
  upstream stage run ids.
- `legal-assemble-groundtruth` đọc `legal_facts.jsonl` của từng văn bản theo
  thứ tự corpus config, nối nguyên trạng và tạo corpus-level `legal_facts.jsonl`
  cùng `run_manifest.json`. Bước này không validate, deduplicate hoặc sửa fact.
  Evidence graph tiêu thụ file fact đã đóng gói này.
- Fact extraction requires every evidence reference to point to a chunk in the
  reading window, but trusts the model's supporting span semantically instead
  of requiring character-for-character source matching.
- Candidate generation receives answerability, bundle kind, bridge, and hops,
  and trusts the generator's output directly. Gate 1 (deterministic binding
  check) and gate 2 (independent LLM verification) were both removed by
  deliberate decision (see "Nguyên tắc kiến trúc" below) — no candidate is
  rejected, relabeled, or checked after generation. Declared `query_type`,
  `answerability`, and `reasoning_hops` are the final labels as generated;
  there is no `requested_*`/observed split.
- Candidate identity is recomputed after every evolution from the final query,
  answer, reasoning, and claims.
- Candidate generation and evolution use true LLM batches. The checked-in
  config sends 5 tasks per request and permits 10 concurrent requests. Every
  returned item carries its stable task id and is checkpointed separately, so
  an interrupted run resumes only unfinished generation/evolution work.
- Candidate review is outside this pipeline. Generated candidates proceed
  directly to downstream synthesis stages; any external-agent review does not
  block or change pipeline state.
- `legal-export-groundtruth` maps completed single-turn candidates to the
  `question`/`expected_answer`/`expected_chunk_ids` schema consumed by
  `legal-evaluate`, while retaining synthesis metadata for later analysis.
- The current deliverable stops at single-turn RAG evaluation. Multi-turn,
  hard negatives, difficulty/leakage scoring, deduplication, and dataset splits
  are outside the current scope.
- The MVP taxonomy contains penalty-fact capabilities only. Authority,
  procedure, and exception queries wait for dedicated fact schemas.
- `data/config/groundtruth/synthesis_v1.json` is the single source of truth
  for embedding selection, retry bounds and delay, quality thresholds,
  screening and fact-extraction batch sizes, and every synthesis LLM
  operation's token/temperature bounds.
- Fact extraction sends one reading window per LLM call. Each window contains
  up to two preceding chunks, the target, and one following chunk, resolved
  only through persisted neighbor links.
  Screening receives parser-derived structural context alongside target text.
  The batch-capable
  interface remains in place, with `batch_size=1`, so each work unit is one target.
  A response must contain exactly one result per target id. Successful batches
  write facts once and then mark every target in that batch in
  `extraction_progress.jsonl`; exhausted batches remain resumable.
- Every LLM-backed synthesis CLI requires `--synthesis-config`; synthesis
  tunables are not read from `.env`. Existing environment settings for other
  subsystems remain unchanged. Synthesis LLM calls keep using the shared
  `LLM_*` variables; evidence-candidate embedding alone uses the isolated
  `SYNTHESIS_EMBEDDING_*` provider variables. Candidate lexical retrieval uses
  VnCoreNLP 1.2 from `SYNTHESIS_VNCORENLP_DIR`; Java defaults to `java` on
  `PATH` and can be overridden by `SYNTHESIS_VNCORENLP_JAVA_PATH`.
- `legal-assemble-groundtruth` always hashes the required synthesis config in
  the run manifest, alongside any taxonomy/persona/distribution configs passed
  through additional `--config` arguments. Manifest fingerprint băm danh sách
  `(fact_id, fact_content_sha256)` theo đúng thứ tự đóng gói và giữ cả record
  trùng; đây chỉ là provenance, không phải deduplication hay quality gate.
- `groundtruth_corpus_v1.json` contains all 35 documents in
  `road_traffic_penalty_scope.json`'s broad `in_scope` list; its corpus snapshot
  freezes that complete selection for reproducible synthesis runs.

```text
data/processed/groundtruth/
|-- <document>.source.jsonl
|-- <document>.chunks.jsonl
|-- <document>.chunk_screening.jsonl
|-- <document>.chunk_screening.run.json
|-- <document>.document_metadata.jsonl
|-- <document>.document_metadata.run.json
|-- <document>.legal_facts.jsonl
|-- <document>.legal_facts.run.json
`-- <document>.extraction_progress.jsonl

data/benchmark_synthesis/
|-- legal_facts.jsonl
|-- run_manifest.json
|-- evidence_relation_candidates.json
|-- evidence_relations.jsonl
|-- evidence_relation_progress.jsonl
|-- scenarios.jsonl
|-- ground_truth.jsonl
`-- candidates/
    |-- single_turn.jsonl
    `-- single_turn.progress.jsonl
```

## Mục tiêu và phạm vi

Pipeline này sinh dữ liệu ground truth single-turn để đánh giá hệ thống RAG
trong phạm vi **xử phạt vi phạm giao thông đường bộ tại Việt Nam**, gồm
retrieval, generation và toàn bộ luồng RAG.

Dữ liệu do LLM sinh ra được dùng thẳng làm candidate — không còn cổng kiểm
tra tự động nào ở tầng candidate (xem "Nguyên tắc kiến trúc" ngay dưới đây).

## Nguyên tắc kiến trúc: tin tưởng hoàn toàn LLM

**Đã đổi so với thiết kế ban đầu, hai lần.**

Lần 1 — tầng fact: bản đầu tài liệu này đặt điểm phê duyệt con người tại
từng fact pháp lý qua một fact ledger (`extracted → span_verified →
needs_expert_review → approved | rejected`, cộng `stale`). Quyết định: tin
tưởng hoàn toàn output của LLM ở tầng trích xuất fact — bỏ hẳn fact ledger và
bước chuyên gia duyệt tay. Không còn khái niệm fact `approved`/`stale`; mọi
fact trong `legal_facts.jsonl` được Phase 2/3 dùng thẳng, không lọc theo
trạng thái nào.

Lần 2 — tầng candidate: bản đầu tài liệu này còn đặt hai cổng chất lượng tất
định ở tầng candidate:

```text
Cổng 1  Ràng buộc căn cứ tất định   không dùng LLM, không dùng người
Cổng 2  Kiểm chứng LLM độc lập      dùng LLM khác generator
```

**Cả hai cổng này đã bị bỏ** (mục 7 và 10 dưới đây giữ lại mô tả gốc, đánh
dấu đã bỏ, để tham khảo lịch sử thiết kế). Quyết định hiện tại mở rộng
nguyên tắc tin tưởng LLM sang toàn bộ tầng candidate: không còn bước nào
kiểm tra, loại bỏ, hay gán nhãn lại candidate sau khi sinh — mọi candidate
sinh ra được ghi thẳng vào output, kể cả khi trích dẫn sai hoặc câu hỏi
không đúng phong cách yêu cầu.

```text
.chunks.jsonl
    -> Legal facts (typed output, kèm quoted span)
    -> Evidence graph trên mọi fact đã trích xuất
    -> Scenario sampling (persona x style x length)
    -> Single-turn candidates                     (không còn cổng kiểm tra nào sau bước này)
    -> Evaluation ground_truth.jsonl
    -> Đánh giá retrieval / generator / end-to-end
```

## 1. Đóng băng corpus

Nguồn đầu vào hiện tại:

- `data/config/groundtruth_corpus_v1.json`
- `data/benchmark_synthesis/groundtruth-corpus-v1-snapshot.json`
- Các file `data/processed/groundtruth/*.chunks.jsonl`

Mỗi lần sinh benchmark phải lưu:

- `corpus_snapshot_id`
- Phiên bản chunk schema, tokenizer và chunker
- Model và phiên bản prompt sinh dữ liệu
- Thời gian sinh
- Cấu hình phân bố câu hỏi

Ground truth phải tham chiếu tới output chunking mà RAG thực sự index, không
đọc lại DOCX để tạo một biểu diễn khác với retrieval corpus.

Không sinh câu hỏi theo thời điểm cho văn bản chưa xác định được ngày hoặc
khoảng thời gian hiệu lực.

## 2. Trích xuất legal facts

LLM đọc từng chunk hoặc nhóm chunk liên quan để trích xuất các fact pháp lý
nguyên tử. Việc trích xuất dùng typed structured output, không dùng regex,
keyword matching hoặc các luật suy diễn thủ công.

```json
{
  "fact_id": "...",
  "fact_content_sha256": "...",
  "source_chunk_id": "...",
  "statement": "Người điều khiển ô tô không chấp hành đèn tín hiệu bị xử phạt theo quy định này.",
  "conduct": "Không chấp hành hiệu lệnh của đèn tín hiệu giao thông",
  "subject": "Người điều khiển ô tô",
  "conditions": [],
  "monetary_penalty": "...",
  "license_points_deducted": "...",
  "additional_penalties": [],
  "remedial_measures": [],
  "valid_from": "2025-01-01",
  "valid_to": null,
  "document_number": "168/2024/NĐ-CP",
  "provision_path": "Điều 6 khoản 9 điểm b",
  "evidence": [
    {
      "chunk_id": "..."
    }
  ]
}
```

Nguyên tắc:

- Một fact chỉ chứa nội dung có bằng chứng trực tiếp.
- `statement` luôn chứa một mệnh đề pháp lý nguyên tử đầy đủ. `conduct`,
  `subject` và các field chế tài là phần cấu trúc bổ sung, có thể null khi
  không áp dụng; vì vậy thẩm quyền, thủ tục, ngoại lệ và nghĩa vụ vẫn có chỗ lưu.
- Không sử dụng kiến thức sẵn có của model làm căn cứ.
- Nếu một fact cần nhiều chunk thì phải lưu tất cả chunk cần thiết.
- Mỗi `evidence` chỉ cần `chunk_id` thuộc reading window. Pipeline đánh giá
  retrieval ở mức chunk nên không yêu cầu model tạo hoặc sao chép một
  `quoted_span` bên trong chunk.
- `chunk_id` phục vụ retrieval evaluation, lấy trực tiếp từ trích dẫn của LLM,
  không tra cứu lại `unit_ids`/`source_refs` từ chunk gốc.
- Query taxonomy có loại `legal_rule` dùng `statement` cho các fact tổng quát;
  các loại câu hỏi mức phạt hoặc đối tượng chỉ được chọn khi field chuyên biệt
  tương ứng thực sự có dữ liệu.

### Sàng lọc chunk trước khi trích xuất

Trước khi trích xuất fact, một critic model rẻ trả lời một câu hỏi nhị phân
cho mỗi chunk: đoạn này có chứa nội dung chế tài cụ thể (hành vi, mức phạt,
thẩm quyền, thủ tục, đối tượng áp dụng) hay không. Chunk bị đánh giá không
có nội dung chế tài bị bỏ qua và ghi vào danh sách loại trừ kèm lý do do
chính model đưa ra — thường là mục lục, tiêu đề chương, đoạn dẫn nhập không
mang chế tài.

Các chunk được gộp theo batch (mặc định 50 chunk/lần gọi) thay vì gọi LLM
riêng cho từng chunk, để giảm số lần round-trip. Mỗi chunk trong batch mang
theo `chunk_id`, và model phải trả về đúng `chunk_id` đó trong từng kết quả —
kết quả được đối chiếu ngược lại theo `chunk_id`, không dựa vào thứ tự trả về,
để tránh lệch kết quả giữa các chunk khi batch lớn.

Screening mặc định cho phép tối đa 16.384 output token. Transport dùng chung
ghi log `input_tokens`, `output_tokens` và `total_tokens` của từng LLM response
nếu OpenAI-compatible provider trả trường `usage`.

Cổng rẻ này đứng trước để giảm số LLM call trích xuất fact đắt tiền phía sau.
Câu hỏi nhị phân này thay cho cách
chấm điểm liên tục rồi so ngưỡng: LLM đáng tin hơn nhiều với quyết định
nhị phân rõ ràng so với việc tự hiệu chỉnh điểm số liên tục 0.0-1.0 (dễ bị
dồn cụm, khiến một ngưỡng cố định gần như vô nghĩa trong thực tế). Danh sách
chunk bị loại phải audit được — lý do lấy trực tiếp từ model, không phải câu
chữ tính toán từ điểm số.

### Metadata cấp văn bản không hỏi qua cửa sổ đọc

`document_number` và ngày hiệu lực là thuộc tính của **văn bản**, không phải của
chunk: số hiệu in ở phần đầu tài liệu, ngày hiệu lực nằm ở điều khoản thi hành
gần cuối. Cả hai đều nằm ngoài cửa sổ đọc của một chunk chế tài, nên hỏi model
trích fact về chúng là hỏi một thông tin nó không thể thấy — thực tế cho ra
trường rỗng, chuỗi kiểu "Nghị định này", hoặc số hiệu của văn bản bị sửa đổi mà
văn bản đang đọc nhắc tới.

Vì vậy hai trường này được giải một lần cho mỗi văn bản, bằng một LLM call
riêng đọc 5 chunk đầu và 5 chunk cuối của **thân văn bản**
(`record_scope == "root"`). Các chunk được sắp theo `chunk_index` trước khi
lấy mẫu để giữ đúng thứ tự tài liệu và tránh phần cuối rơi vào phụ lục, rồi ghi ra artifact
`*.document_metadata.jsonl`. Bước trích fact đọc artifact này và dập giá trị vào
mọi fact của văn bản bằng code — một giá trị duy nhất cho cả văn bản, không thể
lệch nhau giữa các fact.

`valid_from`/`valid_to` vẫn để model trích fact trả về được: một điều khoản có
thể tự nêu ngày hiệu lực riêng, muộn hơn ngày của toàn văn bản. Khi model không
nêu thì lấy ngày của văn bản. Đây là xấp xỉ có ý thức — điều khoản có ngày hiệu
lực riêng nhưng chỉ nêu ở điều khoản thi hành cuối văn bản sẽ nhận ngày của
toàn văn bản, vì thông tin đó không nằm trong cửa sổ đọc của nó.

### Gộp chunk khi một fact cần nhiều nguồn

Trong fact extraction, cửa sổ đọc mở rộng bằng liên kết liền kề và structural context:

- **Chunk liền kề bị cắt bởi giới hạn token.** Chunker cắt một điều khoản dài
  thành hai chunk theo `previous_chunk_id`/`next_chunk_id` đã có sẵn trong
  schema chunk. Gộp bằng cách kéo thêm tối đa 2 chunk liền trước và 1 chunk
  liền sau vào cùng cửa sổ đọc, thuần cơ học, không cần LLM đoán.
Chunk liền kề được tra trên **toàn bộ chunk gốc của văn bản**,
không giới hạn trong tập đã qua được cổng sàng lọc — chỉ tập đã lọc mới quyết
định chunk nào được chọn làm `Target` (tốn LLM call). Một chunk bị cổng sàng
lọc loại (ví dụ đoạn định nghĩa không nêu mức phạt cụ thể) vẫn có thể được
kéo vào làm ngữ cảnh `Preceding` hoặc `Following` cho một `Target` liền kề,
vì việc tra cứu này thuần cơ học, không tốn thêm LLM call nào.

### Định danh fact tách khỏi nội dung fact

`fact_id` là hash trên các trường **định danh**: `statement`,
`document_number`, `provision_path`. `fact_content_sha256` là hash trên **toàn
bộ** nội dung fact — giữ lại để phát hiện nội dung fact đổi giữa các lần trích
xuất, dù không còn gắn với trạng thái duyệt nào (xem "Nguyên tắc kiến trúc" ở
đầu tài liệu).

Cặp này lặp lại đúng kỷ luật mà chunker đang dùng cho `chunk_id` và
`content_sha256`: thay đổi field chuyên biệt không làm đổi `fact_id` nếu mệnh
đề pháp lý trung tâm, văn bản và điều khoản không đổi.

Output dự kiến: `data/benchmark_synthesis/legal_facts.jsonl`.

## 3. Phê duyệt fact ledger — đã bỏ

Bản đầu tài liệu này đặt một cổng chất lượng ở đây, dùng chuyên gia duyệt tay
từng fact qua một máy trạng thái ledger
(`extracted → span_verified → needs_expert_review → approved | rejected`,
cộng `stale` khi nội dung đổi). Đã bỏ theo quyết định "tin tưởng LLM có kiểm
tra tất định" ở đầu tài liệu — mọi fact trích xuất được Phase 2/3 dùng thẳng,
không qua cổng người nào ở tầng fact.

## 4. Xây evidence graph

Evidence graph liên kết mọi fact đã trích xuất.

Các loại quan hệ:

- Hành vi và mức phạt
- Hành vi và đối tượng hoặc loại phương tiện
- Hành vi và trừ điểm hoặc tước giấy phép
- Hành vi và biện pháp khắc phục hậu quả
- Quy định và ngoại lệ
- Quy định và thẩm quyền xử phạt
- Quy định cũ và quy định sửa đổi hoặc thay thế
- Các bước liên tiếp trong thủ tục xử lý

Embedding dùng để tìm quan hệ ứng viên; LLM xác nhận quan hệ dựa trên nội dung
nguồn. Không xác định quan hệ chỉ bằng độ tương đồng hoặc từ khóa.

### Cạnh phải lưu cầu nối

Mỗi cạnh lưu `bridge` — thực thể hoặc thuộc tính chung khiến hai fact liên
quan tới nhau:

```json
{
  "relation_id": "...",
  "relation_type": "amended_by",
  "from_fact_id": "...",
  "to_fact_id": "...",
  "bridge": ["Điều 6 khoản 9 điểm b", "vượt đèn đỏ"],
  "confirmed_by_model": "...",
  "run_id": "..."
}
```

`bridge` được đưa vào prompt sinh câu hỏi multi-hop làm chủ đề bắc cầu. Không
có nó, câu multi-hop thường bị rút gọn thành single-hop vì một chunk đã đủ trả
lời. Với quan hệ temporal, `bridge` chính là điều khoản bị sửa đổi.

LLM phải trả rõ hướng `from_fact_id → to_fact_id` theo ngữ nghĩa của từng loại
quan hệ. Prompt nhận `statement`, các field chuyên biệt, `conditions` và
evidence `chunk_id`. Mỗi relation và progress record lưu `run_id` được
băm từ fact content, embedding profile, cấu hình đề cử/xác nhận, endpoint,
model và prompt version. CLI từ chối resume nếu fingerprint không khớp để
không trộn kết quả cũ với một lần chạy mới.

Candidate retrieval chỉ dùng `LegalFact.statement`, vì Step 3 bắt buộc field này
là một mệnh đề pháp lý đầy đủ và độc lập. Dense embedding và BM25L chạy song
song trên cùng statement. Dense leg mặc định gọi `deepinfra/BAAI/bge-m3` qua
endpoint riêng trong `SYNTHESIS_EMBEDDING_*`; lexical leg dùng VnCoreNLP 1.2,
giữ token âm tiết có dấu, từ ghép và bigram, không bỏ stopword pháp lý như
`không`, `chưa`, `trừ`.

Mỗi leg lấy top 30, sau đó standard RRF cộng `1 / (rrf_k + rank)` với đóng góp
ngang nhau và lấy top 10 candidate cho mỗi fact. Không cộng trực tiếp cosine
với BM25 score và không dùng retrieval weight hoặc similarity threshold.
BM25L `k1`/`b`/`delta`, pool size, RRF `k` và candidate top-k đều nằm trong
synthesis config versioned. Full `LegalFact` chỉ được đưa vào LLM ở bước xác
nhận quan hệ; retrieval không tự đưa ra quyết định ngữ nghĩa.

VnCoreNLP chạy toàn bộ fact text trong một JVM batch, không khởi động Java cho
từng fact. Runtime cần Java 8+ và ba artifact từ release VnCoreNLP `v1.2`:
`VnCoreNLP-1.2.jar`, `models/wordsegmenter/vi-vocab` và
`models/wordsegmenter/wordsegmenter.rdr`. Do giới hạn path resolution của JAR
1.2 trên Windows, `SYNTHESIS_VNCORENLP_DIR` phải là đường dẫn không có khoảng
trắng; ví dụ `C:\Users\Public\legal-rag\vncorenlp-1.2`.

Thiết lập runtime trên Windows PowerShell:

```powershell
winget install --id EclipseAdoptium.Temurin.21.JRE --exact

$vnCoreDir = "C:\Users\Public\legal-rag\vncorenlp-1.2"
$assets = @(
    @{
        RelativePath = "VnCoreNLP-1.2.jar"
        Url = "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/v1.2/VnCoreNLP-1.2.jar"
        Sha256 = "9e2811cdbc2ddfc71d04be5dc36e185c88dcd1ad4d5d69e4ff2e1369dccf7793"
    }
    @{
        RelativePath = "models/wordsegmenter/vi-vocab"
        Url = "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/v1.2/models/wordsegmenter/vi-vocab"
        Sha256 = "0a47c5b55bbce163029d37730a67b9479740388695c29c106c112b815613eaa5"
    }
    @{
        RelativePath = "models/wordsegmenter/wordsegmenter.rdr"
        Url = "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/v1.2/models/wordsegmenter/wordsegmenter.rdr"
        Sha256 = "9e62f96bd93e37a24f364238e8d8ae986fa5dad6dbc9f4eae622ab3651b7fa06"
    }
)

foreach ($asset in $assets) {
    $target = Join-Path $vnCoreDir $asset.RelativePath
    New-Item -ItemType Directory -Force -Path (Split-Path $target) | Out-Null
    Invoke-WebRequest -UseBasicParsing -Uri $asset.Url -OutFile $target
    $actual = (Get-FileHash -Algorithm SHA256 $target).Hash.ToLowerInvariant()
    if ($actual -ne $asset.Sha256) {
        throw "VnCoreNLP checksum mismatch: $($asset.RelativePath)"
    }
}
```

Sau đó đặt `SYNTHESIS_VNCORENLP_DIR` trong `.env`. Chỉ cần đặt
`SYNTHESIS_VNCORENLP_JAVA_PATH` khi `java -version` chưa chạy được trong terminal
mới.

CLI lưu tập cặp đã đề cử vào `evidence_relation_candidates.json` kèm
fingerprint riêng của fact ledger, embedding endpoint/model/profile, batch
size, retrieval config, phiên bản tokenizer và candidate-selection version.
Các lần resume đọc cache này và không embed hoặc tokenize lại toàn corpus; đổi
bất kỳ input liên quan nào làm cache bị từ chối thay vì tái sử dụng âm thầm.
`--prepare-candidates-only` cho phép tạo/kiểm cache rồi dừng trước mọi LLM call,
để người chạy xem số cặp và dự toán chi phí trước khi xác nhận quan hệ.

Graph tạo ra evidence bundle:

- Một fact: single-hop
- Nhiều fact trong cùng văn bản: synthesis
- Nhiều fact thuộc các văn bản khác nhau: multi-hop
- Quy định cũ và mới: temporal

### Ràng buộc cấu trúc khi lấy bundle

Hai ràng buộc tất định, kiểm ngay lúc lấy mẫu, không cần LLM:

- Bundle multi-hop chỉ nhận các fact thuộc **các văn bản khác nhau**. Đây là
  ràng buộc cứng, không phải mô tả: cùng văn bản thì bundle bị hạ xuống loại
  synthesis.
- Độ trùng giữa hai bundle bất kỳ bị chặn trên: `|A ∩ B| / |A| <= τ`, với
  `τ` nằm trong cấu hình. Không có chặn này, phần lớn scenario sẽ tái sử dụng
  gần như cùng một cụm fact và benchmark mất độ đa dạng thực trong khi số
  lượng mẫu trông vẫn lớn.

## 5. Tạo scenario

Pipeline tạo scenario có cấu trúc trước khi yêu cầu LLM viết câu hỏi.

```json
{
  "scenario_id": "...",
  "query_type": "temporal_penalty",
  "query_form": "specific",
  "answerability": "answerable",
  "persona": "tai_xe_ca_nhan",
  "query_style": "no_diacritics",
  "query_length": "short",
  "reasoning_hops": 2,
  "as_of_date": "2025-02-10",
  "fact_ids": ["fact_1", "fact_2"],
  "bridge": ["..."],
  "gold_chunk_ids": ["chunk_1", "chunk_2"]
}
```

### Query taxonomy

- Mức phạt của một hành vi
- Phạt tiền và trừ điểm hoặc tước giấy phép
- Biện pháp khắc phục hậu quả
- Đối tượng và loại phương tiện áp dụng
- Thẩm quyền xử phạt
- Trình tự xử lý vi phạm
- So sánh hai trường hợp gần giống
- Tổng hợp nhiều hành vi trong một tình huống
- Áp dụng quy định tại một thời điểm
- Quy định trước và sau khi sửa đổi
- Không đủ căn cứ để kết luận
- Câu hỏi ngoài phạm vi xử phạt giao thông đường bộ

### Bốn giá trị answerability

Nhị phân answerable và unanswerable là không đủ, và với luật giao thông thì
lẫn lộn hai trạng thái khác hẳn nhau:

| Giá trị | Định nghĩa | Hành vi đúng của hệ thống |
| --- | --- | --- |
| `answerable` | Evidence đủ để kết luận | Trả lời kèm căn cứ |
| `partially_answerable` | Evidence đủ cho một phần | Trả lời phần có căn cứ, nêu rõ phần thiếu |
| `underspecified` | Thiếu dữ kiện từ người dùng, hỏi thêm là giải quyết được | Hỏi lại |
| `unanswerable` | Corpus không chứa căn cứ, hỏi thêm cũng không giải quyết được | Từ chối kết luận |

Phân biệt then chốt là `underspecified` **có thể** giải quyết bằng cách hỏi lại
người dùng, còn `unanswerable` thì không, dù hỏi thêm bao nhiêu. Đây là trường
hợp phổ biến nhất trong thực tế: "vượt đèn đỏ phạt bao nhiêu" là
`underspecified` vì chưa rõ loại phương tiện, không phải `unanswerable`.

### Ba trục đa dạng bề mặt

Nếu không lấy mẫu trên các trục này, toàn bộ benchmark sẽ là tiếng Việt pháp lý
chuẩn mực, khác hẳn cách người dùng thật gõ câu hỏi.

- `persona`: tài xế cá nhân, chủ doanh nghiệp vận tải, cán bộ xử phạt, luật sư,
  người học lái. Persona sinh từ chính corpus, không viết tay.
- `query_style`: đúng chính tả, không dấu, sai chính tả, kiểu tìm kiếm từ khóa.
- `query_length`: ngắn, trung bình, dài.
- `query_form`: `specific` neo vào hành vi và mức phạt cụ thể; `abstract` hỏi
  nguyên tắc hoặc khái niệm. Trục này độc lập với số hop.

Ba trục trên chỉ đổi bề mặt câu hỏi, không đổi evidence.

### Evolution có kiểm soát

Sau khi có câu hỏi nền, evolution tăng độ khó bằng các phép biến đổi **giữ
nguyên evidence**:

- `concretizing`: thêm tình tiết cụ thể vào tình huống
- `reasoning`: buộc suy luận từ quy định thay vì tra cứu trực tiếp
- `hypothetical`: đặt giả định "nếu ... thì"

Các phép làm thay đổi evidence — so sánh hai trường hợp, gộp nhiều context —
không phải evolution; chúng thuộc tầng lấy bundle (synthesis, multi-hop,
temporal) ở mục 4. Ranh giới này giữ cho bất biến cốt lõi của pipeline:
evidence được chốt trước, câu hỏi biến đổi sau.

Số vòng evolution bị chặn trong cấu hình và chuỗi evolution đã áp dụng được
lưu vào candidate để phân tích difficulty theo từng phép.

### Lấy mẫu đa dạng thay vì cân bằng hậu kiểm

Phân bổ mẫu theo trọng số `bundle_kind` trong config bằng weighted round-robin,
nhờ đó single-hop không chiếm hết ngân sách trước khi synthesis, multi-hop và
temporal được xét. Trong từng loại, bundle được luân phiên; persona ưu tiên cặp
persona/bundle chưa dùng, còn query style, length, form, type và answerability
ưu tiên giá trị đang ít xuất hiện. Cách này đạt độ phủ ngay tại lúc sinh, không
cần loại bỏ hậu kiểm.

Taxonomy, danh sách persona và phân bố scenario phải nằm trong file cấu hình
có phiên bản, không hardcode trong source code.

## 6. Sinh single-turn candidates

LLM chỉ nhận scenario và gold evidence đã được chọn trước, sau đó sinh câu hỏi
tiếng Việt, đáp án tham chiếu, các claim bắt buộc và căn cứ của từng claim.

```json
{
  "sample_id": "...",
  "interaction_type": "single_turn",
  "scenario_id": "...",
  "query_vi": "oto vuot den do ngay 10/02/2025 bi phat bao nhieu",
  "reference_answer_vi": "...",
  "required_claims": [
    {
      "claim": "Mức phạt tiền áp dụng",
      "fact_id": "fact_1",
      "evidence": [
        { "chunk_id": "chunk_1" }
      ]
    },
    {
      "claim": "Số điểm giấy phép lái xe bị trừ",
      "fact_id": "fact_1",
      "evidence": [
        { "chunk_id": "chunk_2" }
      ]
    }
  ]
}
```

Nguyên tắc:

- Không để LLM tự tìm hoặc thay đổi evidence sau khi đã sinh câu hỏi.
- Mỗi claim phải khai báo `fact_id` mà nó phái sinh từ đó.
- Prompt nhận thêm bản tóm tắt văn bản nguồn để định hướng ngữ cảnh, nhưng bản
  tóm tắt **không phải** evidence và không được trích dẫn.
- Sinh candidate bằng nhiều model thuộc các họ khác nhau để giảm bias riêng của
  một model.
- `answer_fields` — mức phạt tiền, số điểm trừ, hình phạt bổ sung, biện pháp
  khắc phục — được pipeline **chép cơ học từ fact đã trích xuất**, không để LLM
  sinh lại. Đây là phần đáp án so khớp tất định được khi đánh giá.
- Candidate kèm `reasoning_vi`: chuỗi lập luận nối evidence tới đáp án. Trường
  này cho phép đánh giá tính minh bạch của mẫu.

`--candidates-per-scenario` kiểm soát số lần sinh cho mỗi scenario. Vì không
còn cổng nào lọc bớt (xem "Nguyên tắc kiến trúc" ở đầu tài liệu), con số này
quyết định trực tiếp số candidate cuối cùng — không phải "sinh dư rồi lọc"
như thiết kế ban đầu, mà đơn giản là số biến thể câu hỏi muốn có cho mỗi
scenario.

### Vòng tinh chỉnh bề mặt — đã bỏ

Bản đầu mục này có một bước critic chấm phong cách câu hỏi (cách gõ, độ dài,
dạng cụ thể/trừu tượng) rồi viết lại tối đa `max_quality_retries` lần trước
khi qua cổng 1. **Đã bỏ cùng lúc với hai cổng chất lượng.** Phong cách câu
hỏi giờ được kiểm soát trực tiếp trong prompt sinh (few-shot ví dụ cụ thể
theo từng `query_style`, mốc số từ theo `query_length`) và nhắc lại trong
prompt evolve (giữ nguyên cách gõ gốc + ràng buộc lại độ dài), không qua
vòng critique riêng nào.

## 7. Ràng buộc căn cứ tất định — đã bỏ (cổng 1)

Bản đầu mục này mô tả một cổng tất định (không dùng LLM, không dùng người):
với mỗi phần tử `evidence` của mỗi claim, đối chiếu `chunk_id` có nằm trong
evidence set thật của fact không và so khớp một span trích dẫn — loại cơ học
candidate nào sai.

**Đã bỏ theo quyết định "tin tưởng hoàn toàn LLM"** (xem "Nguyên tắc kiến
trúc" ở đầu tài liệu). Candidate không bị loại bằng một quality gate. Code
chỉ chiếu `chunk_id` LLM trả về lên evidence set thật của fact khi dựng
`required_claims`; `expected_chunk_ids` vẫn lấy từ scenario đã chốt.
`grounding_score` không còn được tính.

## 8. Sinh multi-turn candidates — ngoài phạm vi hiện tại

Multi-turn phát triển từ các fact đã trích xuất, không sinh hội thoại tự do từ đầu.

Bằng chứng từ các benchmark multi-turn đã công bố cho thấy hội thoại sinh hoàn
toàn tự động thất bại ở đúng những chỗ cần nhất: câu unanswerable sinh tự động
chất lượng thấp, và độ đa dạng evidence trong hội thoại synthetic thấp hơn
nhiều lần so với hội thoại người viết. Vì vậy phân tầng:

- Lượt `answerable` và `partially_answerable`: sinh tự động, không qua cổng
  kiểm tra nào (cả hai cổng đã bỏ, xem mục 7 và 10).
- Lượt `underspecified` và `unanswerable`: người viết hoặc người sửa, không
  chấp nhận sinh tự động thuần túy.

### Sinh bằng user simulator hướng mục tiêu

Các lượt sinh tự động không được viết một mạch cả hội thoại bằng một LLM call.
Chúng sinh ra từ vòng lặp mô phỏng hai vai:

1. **Conversation goal** dựng cơ học từ scenario: persona, tình huống, và
   danh sách `required_claims` phải được bộc lộ hết qua hội thoại.
2. **Simulator đóng vai người dùng.** Mỗi lượt nó nhận lịch sử hội thoại và
   goal state hiện tại, rồi tiết lộ thông tin **dần** theo persona — không
   dồn mọi dữ kiện vào lượt đầu, giống người dùng thật.
3. **Oracle đóng vai trợ lý.** Oracle chỉ được cấp gold evidence của
   scenario, và **không phải** hệ RAG đang được đánh giá — nếu dùng chính hệ
   RAG, lỗi của nó sẽ thấm vào ground truth. Lượt trả lời của oracle được
   dùng thẳng như mọi candidate khác (không qua cổng nào).
4. **Goal state cập nhật sau mỗi lượt**: claim nào đã bộc lộ, dữ kiện nào
   người dùng đã cung cấp. Bản cập nhật này chính là `state_delta` của turn.
5. **Dừng** khi goal hoàn thành hoặc chạm `max_turns`. Hội thoại lưu
   `goal_completion` và `stop_reason`.

Goal state phải được nạp lại cho simulator ở **từng lượt**, không để nó tự
suy từ lịch sử. Đo đạc đã công bố trên user simulator cho thấy simulator chỉ
dựa vào lịch sử hỏng theo đúng các kiểu cần tránh nhất: quên thành phần mục
tiêu, mâu thuẫn với dữ kiện chính nó đã cung cấp, và dừng sai thời điểm — ba
kiểu này chiếm phần lớn lỗi quan sát được.

Tiết lộ thông tin dần còn cho một lợi ích cấu trúc: lượt `underspecified`
xuất hiện **tự nhiên** — lượt đầu chưa nêu loại phương tiện là một mẫu hỏi-lại
thật, không phải mẫu dàn dựng. Nhãn cuối của các lượt này vẫn do người xác
nhận theo phân tầng ở trên.

Mỗi turn lưu:

```json
{
  "turn_id": "...",
  "user_input": "Ngoài tiền phạt còn bị trừ điểm không?",
  "standalone_query": "Người điều khiển ô tô vượt đèn đỏ ngày 10/02/2025 có bị trừ điểm giấy phép lái xe không?",
  "reference_answer_vi": "...",
  "depends_on_turn_ids": ["turn_1", "turn_2"],
  "state_delta": {},
  "required_claims": [],
  "answerability": "answerable",
  "turn_role": "follow_up"
}
```

`standalone_query` dùng để đánh giá query rewriting và conversational
retrieval. `turn_role` phân biệt lượt hỏi sâu thêm và lượt làm rõ.

Các dạng turn cần bao phủ:

- Hỏi sâu thêm
- Bổ sung thời điểm
- Thay đổi phương tiện, đối tượng hoặc điều kiện
- Yêu cầu so sánh
- Người dùng sửa thông tin đã cung cấp
- Thiếu dữ kiện và hệ thống phải hỏi lại
- Câu hỏi không còn nằm trong domain

Mốc phân bố lấy từ hành vi người dùng thật, dùng làm mục tiêu chứ không phải
con số tự đặt:

- Khoảng 85% số lượt là non-standalone
- Khoảng 25% số lượt thuộc `underspecified` hoặc `unanswerable`
- Không quá 2 lượt `unanswerable` trong cùng một hội thoại
- Trung bình ít nhất 1 co-reference mỗi hội thoại

## 9. Khai thác hard negatives — ngoài phạm vi hiện tại

Công đoạn này **cần một retriever đang hoạt động**. Hiện tại repository chưa có
module retrieval, nên mục này bị chặn cho tới khi retriever tồn tại. Không
chặn các mục trước đó.

Với mỗi query, retriever lấy các chunk gần nghĩa nhưng không nằm trong gold
evidence. Ứng viên hard negative có thể là:

- Cùng hành vi nhưng khác phương tiện
- Cùng phương tiện nhưng khác hành vi
- Mức phạt gần giống nhưng căn cứ khác
- Khoản hoặc điều liền kề
- Quy định cũ đã bị sửa đổi
- Quy định về thẩm quyền thay vì mức phạt
- Quy định đúng chủ đề nhưng không đủ để trả lời

LLM verifier phải xác nhận hard negative không chứa đủ căn cứ trả lời. Một
chunk không thuộc gold evidence không tự động trở thành negative vì có thể tồn
tại nhiều căn cứ tương đương hoặc bổ sung.

Loại trước bằng cơ chế tất định các chunk đã thuộc evidence set của bất kỳ fact
nào xuất hiện trong scenario, để verifier không phải xử lý các trường hợp hiển
nhiên.

## 10. Kiểm chứng LLM độc lập — đã bỏ (cổng 2)

Bản đầu mục này mô tả một cổng dùng model/prompt khác generator, kiểm 5
chiều (stand alone, question specificity, question/answer grounding, domain
và temporal), cộng cơ chế **hop necessity**: nếu bỏ một fact mà đáp án vẫn đủ
căn cứ thì verifier gán lại `reasoning_hops` theo mức thật — cơ chế bắt
shortcut cho câu hỏi multi-hop sinh tự động. Mục này cũng mô tả việc gán
nhãn lại `query_type`/`answerability` theo quan sát của verifier, tách khỏi
nhãn đặt hàng ban đầu qua các trường `requested_*`.

**Đã bỏ theo quyết định "tin tưởng hoàn toàn LLM"** (xem "Nguyên tắc kiến
trúc" ở đầu tài liệu). Không còn verifier độc lập nào kiểm tra candidate sau
khi sinh — `query_type`, `answerability`, `reasoning_hops` trong output đúng
y giá trị scenario đã yêu cầu, không có nhãn "quan sát được" tách riêng,
và không có cơ chế nào bắt shortcut cho multi-hop "ảo" (candidate khai nhiều
hop nhưng thực chất không cần đủ số fact đã khai).

## 11. Đo leakage và difficulty — ngoài phạm vi hiện tại

Mỗi candidate chạy trong bốn chế độ. Ba chế độ đầu không cần retriever.

### Closed-book

Không cung cấp context, đo khả năng model trả lời bằng parametric knowledge.
Kết quả lưu dưới dạng leakage score để phân tích, không nhất thiết tự động loại
câu hỏi pháp luật thực khỏi benchmark.

### Oracle context

Cung cấp đúng gold evidence để kiểm tra evidence có đủ và đo riêng năng lực
generator.

### Oracle trộn nhiễu

Cung cấp gold evidence trộn với các chunk top-k do retriever trả về. Đây là chế
độ sát thực tế nhất: nó đo được generator có bị nhiễu bởi chunk sai khi gold
đã nằm trong context hay không. Hai chế độ oracle thuần và full RAG đều không
tách được lỗi này.

### Baseline retrieval

Chạy retriever hiện tại và lưu:

- Rank của từng gold chunk
- Similarity với hard negatives
- Số evidence cần kết hợp
- Reasoning hops
- Query-evidence semantic distance

Không loại một candidate chỉ vì baseline không retrieve được; đó có thể là một
test case khó và có giá trị chẩn đoán.

## 12. Kiểm soát độ phủ trên bảng fact — ngoài phạm vi hiện tại

Độ phủ đo trên **bảng fact**, không đo trên tập câu hỏi. Chỉ có cách này mới
trả lời được câu hỏi quan trọng nhất là còn fact nào chưa được kiểm tra.

Chỉ số chính:

- Tỷ lệ fact có ít nhất một candidate
- Danh sách fact chưa được phủ, xuất tường minh trong coverage report
- Số candidate trên mỗi fact, phát hiện fact bị khai thác quá mức

Các chiều cân bằng khác, theo cấu hình có phiên bản:

- Query type và query form
- Persona, query style, query length
- Single-turn và multi-turn
- Văn bản nguồn
- Hành vi vi phạm
- Loại phương tiện hoặc đối tượng
- Loại chế tài
- Single-hop và multi-hop
- Temporal và non-temporal
- Bốn giá trị answerability
- Mức độ khó
- Evidence dạng paragraph và table

### Khử trùng lặp có giữ lại thông tin

Gom cụm embedding của câu hỏi, ngưỡng cosine `0.9`, mỗi cụm giữ một đại diện.
Kích thước cụm được lưu thành `salience_weight` của mẫu đại diện thay vì bị
vứt đi.

Một fact sinh ra nhiều câu hỏi tương tự thường là fact quan trọng hơn. Xóa các
bản trùng mà không ghi lại số lượng là làm mất chính thông tin đó.

## 13. Review candidate — ngoài phạm vi pipeline

Pipeline không triển khai module, CLI, trạng thái hay output dành cho review.
Candidate hoàn tất ở trạng thái `completed` trong `CandidateProgressRecord`
và được chuyển thẳng sang các bước downstream. Nếu dùng agent bên ngoài để
review, kết quả đó là artifact độc lập, không chặn pipeline và không thay đổi
candidate đã sinh.

## 14. Chia dataset — ngoài phạm vi hiện tại

```text
ground_truth_dev.jsonl
ground_truth_test.jsonl
ground_truth_hidden_test.jsonl
```

Chia theo `fact_id`, không chia theo câu hỏi. Mọi candidate phái sinh từ cùng
một fact, và mọi fact nối với nhau trong evidence graph, phải nằm trong cùng
một split.

Chia theo fact khiến ràng buộc chống rò rỉ trở thành hệ quả của cấu trúc dữ
liệu, không phải một bước lọc riêng có thể quên.

- `dev`: điều chỉnh retrieval, prompt và top-k
- `test`: đánh giá nội bộ định kỳ
- `hidden_test`: không sử dụng trong quá trình tuning

## 15. Ground-truth schema cuối

```text
schema_version
question_id              (= candidate sample_id)
question                 (= candidate query_vi)
expected_answer          (= candidate reference_answer_vi)
expected_chunk_ids       (= candidate gold_chunk_ids)
scenario_id
interaction_type         (= single_turn)
bundle_kind
bridge
required_claims          (claim, fact_id, evidence[chunk_id])
answer_fields            (chép cơ học từ fact, so khớp tất định)
reasoning_vi
evolutions
query_type
query_form
persona
query_style
query_length
answerability
as_of_date
reasoning_hops
source_fact_ids
source_fact_content_sha256
generator_model
```

`question`, `expected_answer`, và `expected_chunk_ids` là ba trường evaluation
CLI đọc trực tiếp. Các trường còn lại giữ provenance và nhãn synthesis.

## 16. Đánh giá RAG

Mọi chỉ số dưới đây phải báo cáo **tách theo `query_type` và theo
`answerability`**, không chỉ báo cáo giá trị tổng.

Tuning trên tập không cân bằng cho ra siêu tham số sai: tỷ trọng của một nhánh
trong hybrid search có thể tối ưu trên tập đầy đủ nhưng sai hẳn nếu chỉ đo trên
nhóm câu hỏi dễ. Giá trị tổng che mất điều đó.

### Retrieval

- Gold evidence Recall@k
- Precision@k
- MRR
- nDCG@k
- Hard-negative rejection
- Multi-hop evidence coverage
- Conversational retrieval consistency

Chỉ tính trên các mẫu `answerable` và `partially_answerable`.

### Generator với oracle context

- Claim precision
- Claim recall và completeness
- Faithfulness
- Citation precision và recall
- Temporal correctness
- Table reasoning accuracy
- Accuracy theo field: so khớp **tất định** trên `answer_fields` — mức tiền,
  số điểm trừ, hình phạt bổ sung. Phần này không cần LLM judge, nên vừa rẻ
  vừa không nhiễu, và là chỉ số đáng tin nhất khi hai model chấm lệch nhau.

### End-to-end RAG

- Answer correctness
- Evidence coverage
- Hallucination rate
- Completeness
- Citation correctness
- Temporal correctness
- Multi-turn consistency
- Knowledge retention: hệ thống không hỏi lại thông tin người dùng đã cung
  cấp ở lượt trước. Đo được nhờ goal state đã lưu theo từng turn — dữ kiện
  nào đã tiết lộ ở turn nào là có sẵn trong ground truth.

### Chấm điểm có điều kiện trên answerability

Abstention không phải một chỉ số riêng đứng cạnh các chỉ số khác. Điểm
generation của một mẫu bị **đặt về 0** khi hành vi của hệ thống không khớp
nhãn answerability:

- `unanswerable` mà hệ thống vẫn kết luận
- `underspecified` mà hệ thống kết luận thay vì hỏi lại
- `answerable` mà hệ thống từ chối trả lời

Nếu không điều kiện hóa, một đáp án trôi chảy cho câu hỏi vốn không trả lời
được vẫn ăn điểm cao, và benchmark thưởng cho đúng hành vi nguy hiểm nhất
trong lĩnh vực pháp lý.

Dùng một LLM judge riêng để phân loại hành vi phản hồi thành trả lời, hỏi lại
hoặc từ chối.

## 17. Cấu trúc output đề xuất

```text
data/benchmark_synthesis/
├── document_metadata.jsonl
├── extraction_progress.jsonl
├── legal_facts.jsonl
├── evidence_relation_candidates.json
├── evidence_relations.jsonl
├── evidence_relation_progress.jsonl
├── scenarios.jsonl
├── candidates/
│   ├── single_turn.jsonl
│   └── single_turn.progress.jsonl
└── ground_truth.jsonl
```

## Tài liệu tham khảo phương pháp

Các quyết định thiết kế trong tài liệu này chắt lọc từ những nguồn sau:

- **ViHERMES** (arXiv 2602.07361, benchmark multihop trên văn bản quy phạm y
  tế tiếng Việt) — ràng buộc khác-văn-bản cho bundle multi-hop, chặn trùng
  lặp `τ` giữa các bundle, structural validation theo số hop, trường
  reasoning tường minh.
- **mtRAG / MTRAG-UN** — bốn giá trị answerability, các mốc phân bố
  multi-turn, chấm điểm điều kiện trên answerability.
- **YourBench** — trích dẫn nguyên văn kiểm bằng fuzzy match (cổng 1).
- **Know Your RAG** (arXiv 2411.19710) — sinh từ statement trước rồi mới ra
  câu hỏi, gán nhãn lại sau sinh, báo cáo metric tách theo nhãn.
- **RAGAS** — scenario sampling đa trục, `bridge` trên cạnh evidence graph.
- **DeepEval** (synthesizer và conversation simulator) — cổng chất lượng
  chunk, vòng tinh chỉnh bề mặt có feedback, evolution có kiểm soát, vòng
  lặp simulator với goal và tiêu chí dừng.
- **τ-bench và nghiên cứu goal alignment cho user simulator** (TACL, arXiv
  2507.20152) — goal state tracking nạp lại từng lượt, tiết lộ thông tin dần,
  phân loại lỗi simulator.
- **ALQAC** — tiêu chí relevance tối giản cho gold evidence: một điều luật là
  relevant khi và chỉ khi câu hỏi trả lời được bằng điều luật đó.

## Thứ tự triển khai

Các mục 1 tới 8 không cần retriever và nên hoàn thành trước.

1. Hoàn thiện temporal metadata cho corpus. **Chưa làm** — việc rà soát dữ
   liệu, không phải code.
2. Định nghĩa typed schema cho `LegalFact`, `EvidenceReference`, trong đó
   `fact_id` và `fact_content_sha256` tách bạch. ✅ **Đã xong** cho
   `LegalFact`/`EvidenceReference`/`EvidenceRelation`/`EvidenceBundle`/
   `Scenario`; `GroundTruthCandidate`, `Conversation`, `ConversationTurn`
   còn lại cho Phase 3.
3. Implement lọc chất lượng chunk và legal-fact extraction với provenance ở
   mức `chunk_id`. ✅ **Đã xong.**
4. Implement ràng buộc căn cứ tất định. ✅ **Đã xong**, ở mức candidate (mục
   7, Phase 3) — `binding_gate.check_binding`. Không còn cổng span-match ở
   mức fact: fact ledger đã bỏ (xem "Nguyên tắc kiến trúc" ở đầu tài liệu).
5. ~~Xây fact ledger và quy trình phê duyệt~~ — **đã bỏ**, thay bằng tin
   tưởng LLM ở tầng trích xuất fact. Pilot thật trên 168/2024/NĐ-CP còn chờ.
6. Tạo file cấu hình versioned cho query taxonomy, persona, evolution và phân
   bố scenario. ✅ **Đã xong.**
7. Implement evidence graph có lưu `bridge` và ràng buộc cấu trúc bundle.
   ✅ **Đã xong**, phạm vi MVP: bundle đúng 2 fact trên mỗi quan hệ đã xác
   nhận, chưa nối chuỗi thành bundle 3+ fact.
8. Implement single-turn generation kèm vòng tinh chỉnh bề mặt, cổng 1, cổng
   2 và gán nhãn lại. ✅ **Đã xong, sau đó đã bỏ.** Toàn bộ vòng tinh chỉnh
   bề mặt, cổng 1 và cổng 2 bị bỏ theo quyết định "tin tưởng hoàn toàn LLM"
   (mục 6/7/10) — chỉ còn generation + evolution.
9. Chạy pilot single-turn và ổn định schema candidate. ✅ **Đã xong.** Review
   bằng agent bên ngoài không thuộc pipeline và không chặn bước tiếp theo.
10. Export candidate sang schema evaluation bằng `legal-export-groundtruth`.
    ✅ **Đã xong.** Đây là điểm kết thúc của deliverable hiện tại.

Retriever-specific hard negatives, difficulty/leakage, multi-turn và dataset
splits đã được loại khỏi phạm vi hiện tại.

Không bắt đầu sinh dữ liệu hàng loạt trước khi schema fact và temporal
metadata đã ổn định.

## Kế hoạch triển khai kỹ thuật

> **Lưu ý đọc phần dưới:** đây là nhật ký triển khai theo thời điểm, giữ
> nguyên để tham khảo lịch sử thiết kế. **Phase 3** (cổng 1, cổng 2, vòng
> tinh chỉnh bề mặt/`CritiqueSurface`, `binding_gate.py`, `verification.py`,
> `review_status`) đã được triển khai đúng như mô tả dưới đây, **rồi sau đó
> bị bỏ hoàn toàn** theo quyết định ở "Nguyên tắc kiến trúc" đầu tài liệu và
> mục 6/7/10. Coi mọi mô tả module/BAML/test của Phase 3 dưới đây là lịch sử,
> không phải trạng thái code hiện tại — trạng thái hiện tại nằm trong
> `candidate_pipeline.py` (`generate_candidates`, chạy concurrent, không gate).

Bản dịch 13 bước ở trên sang module, file BAML, config và test cụ thể, theo
quy ước đã có của repo (`chunker/`, `parser/`, `embedding/`). Tuân thủ
[AGENTS.md](../AGENTS.md): thay đổi nhỏ và tập trung theo từng phase, type
hint ở API công khai, `pathlib`, không dùng `print()`, không thêm dependency
trừ khi thư viện hiện có không đủ.

### Quyết định vị trí: ban đầu tái dùng `evaluation/`, sau đó đổi tên thành `benchmark_synthesis/`

`src/legal_rag/evaluation/` đã tồn tại nhưng rỗng — đây từng là nơi chứa
pipeline Ragas cũ (`groundtruth.py`, `schemas.py`, `snapshot.py`,
`synthesis.py`) trước khi bị gỡ ở commit `38475a6`. Toàn bộ code mới của kế
hoạch này ban đầu đặt lại vào `src/legal_rag/evaluation/`, chia theo module
một-file-một-trách-nhiệm thay vì gộp bốn file lớn như thiết kế cũ, không tạo
package song song `groundtruth/`.

`data/evaluation/evaluation-data.md` lúc đó trỏ tới API
`generate_groundtruth_candidates(...)` và biến môi trường `RAGAS_*` đã không
còn tồn tại; nội dung này được viết lại ở Phase 0.

**Đổi lại sau đó:** tên `evaluation/` gây hiểu lầm — nó đọc như "đo lường chất
lượng hệ RAG", trong khi toàn bộ module ở đây chỉ *sinh* dữ liệu ground truth
(fact → evidence graph → scenario → candidate), không đo gì cả. Việc
đo lường thật (Recall@k, Faithfulness, Answer correctness...),
mục 16) là một mối quan tâm khác, chưa có code, sẽ cần vị trí riêng của nó khi
tới Phase 8. Do đó `src/legal_rag/evaluation/` và `data/evaluation/` được đổi
tên thành `src/legal_rag/benchmark_synthesis/` và `data/benchmark_synthesis/`
— khớp với mục đích thật (sinh benchmark) và không còn xung đột tên với phần
đánh giá RAG thật sẽ thêm sau. Extra `evaluation` trong `pyproject.toml`
(nhóm dependency hiện dùng: `rapidfuzz`, `sentence-transformers`) giữ
nguyên tên, vì nó tiếp tục phục vụ cả code synthesis hiện tại lẫn code
đánh giá RAG thật ở Phase 8.

### Phase 0 — Hạ tầng dùng chung ✅ Đã triển khai

Ba phần này được ít nhất hai phase sau dùng lại, nên làm trước và làm đúng
một lần, tránh trùng lặp giữa các module.

**Module encode embedding — hiện chưa tồn tại.** `embedding/tokenizer.py` chỉ
đếm token; không có module nào sinh vector. Đây là hạ tầng bắt buộc cho Phase
1 (gộp chunk theo tương đồng), Phase 2 (tìm quan hệ evidence ứng viên) và
Phase 8 (dedup theo cụm). Thêm `src/legal_rag/embedding/encoder.py`: bọc
`sentence-transformers` (đã có sẵn dưới extra `vietnamese`), nạp model theo
đúng `EmbeddingProfile.model_id` đang cấu hình cho retrieval — để không gian
embedding dùng sinh ground truth khớp với không gian embedding RAG thật sự
dùng. Hàm trả về vector đã chuẩn hóa L2, so khớp bằng tích vô hướng.

**Dependency `rapidfuzz` — đã gỡ lại.** Thêm ở Phase 0 vì Cổng 1 cần Levenshtein partial ratio để
kiểm `quoted_span` khớp với text chunk. `difflib` trong thư viện chuẩn không
có phép so khớp "chuỗi con tốt nhất" đúng ngữ nghĩa partial ratio; tự viết lại
phép này là nguồn lỗi tinh vi. `rapidfuzz` nhỏ, thuần Python/C, không kéo
theo dependency nặng — hợp lý theo tiêu chí "thư viện hiện có không đủ" của
AGENTS.md. Đặt trong extra `evaluation`, không phải dependency chính: toàn bộ
pipeline synthesis vốn đã được tách khỏi core (parser/chunker) qua extra này,
nên `rapidfuzz` đi cùng chỗ với `sentence-transformers` thay vì
buộc mọi người dùng `legal-parse`/`legal-chunk` phải cài thêm nó.

Khi Cổng 1 bị bỏ (mục 7), `rapidfuzz` không còn nơi nào dùng — đã gỡ khỏi
`pyproject.toml` và `uv.lock`.

**Atomic JSONL writer dùng chung.** `parser/writer.py` và `chunker/writer.py`
hiện có cùng một khối logic ghi tệp tạm rồi `replace()` nguyên tử, lặp lại y
hệt. Thêm một module thứ ba (`benchmark_synthesis/writer.py`) sẽ là lần lặp
thứ ba — đủ điều kiện theo AGENTS.md để tách hàm dùng chung. Thêm
`src/legal_rag/io_utils.py` chứa `write_jsonl_atomic(path, records)`, refactor
hai module cũ gọi vào đây, rồi `benchmark_synthesis/writer.py` dùng lại. Đây là thay
đổi cross-cutting nhỏ, nên làm thành một commit riêng trước khi viết code
Phase 1.

**Cấu hình.** `SynthesisSettings` nằm trong
`benchmark_synthesis/config.py` và đọc từ file JSON versioned được truyền rõ
qua `--synthesis-config`. File này sở hữu embedding profile, retry, thời gian
chờ giữa các attempt, threshold, batch size, giới hạn output và temperature cho từng LLM operation. Trong luồng
synthesis, `.env` chỉ cung cấp provider/credential/deployment settings; cấu hình
env của các subsystem khác không thuộc thay đổi này. Persona, query taxonomy,
phân bố scenario và evolution tiếp tục nằm ở các file JSON versioned riêng dưới
`data/config/groundtruth/`. Các CLI đọc bằng `Path.read_text` + `json.loads`,
không có đường dẫn config ngầm định.

### Phase 1 — Trích xuất fact (mục 2 ở trên) ✅ Đã triển khai

Module trong `benchmark_synthesis/`:

- `models.py` — `LegalFact`, `EvidenceReference` dạng
  `@dataclass(frozen=True, slots=True)` với `to_dict()`, hằng
  `LEGAL_FACT_SCHEMA_VERSION`, theo đúng khuôn `chunker/models.py`.
- `chunk_screening.py` — gọi BAML hỏi nhị phân "có nội dung chế tài không"
  theo batch, trả về `ChunkScreeningRecord` (chunk_id, has_penalty_content,
  reason) cho **mọi** chunk đầu vào, kể cả chunk bị loại — không tự chia
  accepted/excluded trong bộ nhớ, để CLI ghi thẳng ra file cho bước sau đọc
  lại.
- `fact_extraction.py` — dựng cửa sổ đọc theo thứ tự gồm tối đa 2 chunk
  `Preceding`, `Target` và 1 chunk `Following`; các neighbor chỉ đi qua
  `previous_chunk_id` và `next_chunk_id` đã được chunker lưu. Screening nhận
  structural context tách riêng khỏi source text. Sau đó code gọi BAML trích fact rồi gán `fact_id` và
  `fact_content_sha256` qua `sha256_json`. Nội dung pháp lý vẫn lấy từ typed
  LLM output; `statement` là mệnh đề bắt buộc, còn `conduct`/`subject` là
  optional để schema không làm mất các quy định về thẩm quyền hay thủ tục.
  `chunk_id` phải thuộc reading window và là toàn bộ provenance
  cần lưu cho retrieval ground truth; pipeline không yêu cầu span trích dẫn.
  Không gán thêm
  `unit_ids`/`source_refs` vì hai field này không thuộc `EvidenceReference`.
- `writer.py` — đọc/ghi JSONL nguyên tử cho `chunk_screening.jsonl` và
  `legal_facts.jsonl`, dùng lại `io_utils.write_jsonl_atomic`.

BAML trong `baml_src/`:

- `chunk_screening.baml` — class `ChunkScreeningItem` (chunk_id, text) và
  `ChunkScreeningResult` (chunk_id, has_penalty_content, reason), function
  `ScreenChunks(chunks: ChunkScreeningItem[]) -> ChunkScreeningResult[]` gọi
  theo batch được quy định bởi `stages.chunk_screening.batch_size` trong
  `synthesis_v1.json`, kèm persona và few-shot examples. Kết quả đối chiếu
  ngược lại theo `chunk_id`, không theo thứ tự mảng trả về.
- `legal_fact_extraction.baml` — enum `ChunkWindowRole`, class
  `ChunkWindowEntry`, `EvidenceEntry`, `LegalFact` khớp JSON ở mục 2, function
  `ExtractLegalFacts(window: ChunkWindowEntry[]) -> LegalFact[]`, kèm persona
  và few-shot examples. Window phân biệt `Preceding`, `Target`, `Following`
  theo ba role `Preceding`, `Target`, `Following`.

CLI tách thành các artifact riêng, nối nhau qua file — theo đúng cách các phase
sau nối nhau (`legal_facts.jsonl` → `evidence_relations.jsonl`/`scenarios.jsonl`
→ `single_turn.jsonl`), không gọi ngầm trong cùng 1 process nữa:

- `legal-screen-chunks` (`chunk_screening_cli.py`): nhận đường dẫn chunk JSONL
  của một văn bản, ghi `chunk_screening.jsonl` — 1 dòng verdict cho mỗi chunk,
  kể cả chunk bị loại (audit được, khác với bản trước đó không ghi gì). Chạy
  lại (resume) trên cùng output path chỉ sàng lọc chunk chưa có verdict, giữ
  nguyên các verdict cũ. Một batch gọi LLM lỗi (timeout, rate limit...) được
  thử lại theo `llm_execution.max_retries` trong synthesis config; nếu vẫn lỗi
  thì batch đó không được đánh dấu hoàn thành, command thoát lỗi sau khi giữ
  lại các batch đã thành công để resume ở lần chạy sau. `chunk_screening.jsonl`
  được ghi lại
  (atomic) sau **mỗi batch** thay vì chỉ một lần ở cuối, nên nếu tiến trình
  bị dừng đột ngột giữa chừng (không phải lỗi batch, mà là crash/kill hẳn
  tiến trình), các batch đã xong trước đó vẫn không mất.
- `legal-resolve-document-metadata` (`document_metadata_cli.py`): nhận đường dẫn
  chunk JSONL của một văn bản, sắp root chunks theo `chunk_index`, gọi LLM
  **một lần** trên 5 chunk đầu và 5 chunk cuối, rồi ghi
  `document_metadata.jsonl` — một dòng duy nhất mang
  `document_number`, `valid_from`, `valid_to`. Nếu output đã tồn tại và hợp lệ,
  command coi work unit metadata đã hoàn thành và không gọi lại LLM.
- `legal-prepare-groundtruth-document` (`document_preparation_cli.py`): chạy hai
  command logic trên trong cùng một process và song song hóa chúng. Đây là cách
  chạy mặc định nên dùng khi chuẩn bị artifact cho một văn bản mới; hai CLI
  riêng vẫn hữu ích khi cần chạy lại hoặc debug từng artifact.
- Mỗi artifact resume ở bước document preparation có sidecar fingerprint
  `<artifact>.run.json`. Artifact cũ không có sidecar phải được sinh lại bằng
  output path mới hoặc xóa có chủ đích trước khi chạy lại; pipeline không tự
  đóng dấu lại artifact legacy vì như vậy có thể trộn cache sai.
- `legal-extract-facts` (`fact_extraction_cli.py`): nhận đường dẫn chunk JSONL,
  `chunk_screening.jsonl` **và** `document_metadata.jsonl`, lọc lại chunk có
  `has_penalty_content=true`,
  trích fact, ghi thẳng `legal_facts.jsonl` — không còn ledger, không còn
  bước duyệt tay. Mỗi `LegalFact` mang thêm `source_chunk_id` — chunk Target
  đã sinh ra fact đó — làm provenance. Resume đọc từ artifact riêng
  `extraction_progress.jsonl`, ghi một dòng cho **mỗi Target đã đọc xong, kể
  cả Target trả về danh sách rỗng**. Đây là điểm khác biệt quan trọng: suy
  "chunk nào đã chạy" từ `source_chunk_id` của fact là sai, vì một Target trả
  về rỗng hợp lệ (biểu mẫu hành chính, thủ tục không kèm chế tài) không để lại
  fact nào — trên dữ liệu thật 249/347 Target rơi vào nhóm này, tức 72% công
  việc đã xong sẽ bị chạy lại. Thứ tự ghi là **facts trước, progress sau**:
  crash giữa hai lần ghi chỉ làm Target đó chạy lại ở lần sau và fact trùng
  được khử theo `fact_id`, còn thứ tự ngược lại sẽ mất fact vĩnh viễn.
  Một lệnh gọi LLM lỗi được thử lại tối đa
  `DEFAULT_EXTRACTION_MAX_RETRIES` lần; nếu vẫn lỗi thì bỏ qua chunk đó (resume
  ở lần sau). File output được ghi lại (atomic) sau **mỗi Target** thay vì chỉ
  một lần ở cuối, nên tiến trình bị crash giữa chừng không mất fact đã trích
  trước đó.
- `legal-backfill-extraction-progress` (`extraction_progress_backfill_cli.py`):
  backfill một lần cho các văn bản đã trích fact xong **trước khi** có
  `extraction_progress.jsonl`. Ghi một dòng progress cho mọi chunk
  `has_penalty_content=true` trong file screening. Chỉ đúng khi lần chạy trích
  fact đã hoàn tất cả văn bản; chunk nào bị bỏ qua sau khi retry hết (log
  `Skipping fact extraction for chunk`) phải truyền qua `--skipped-chunk-id`
  để lần chạy sau vẫn trích lại.
- `legal-migrate-legal-facts` (`legal_fact_migration_cli.py`): migration một lần
  từ `LEGAL_FACT_SCHEMA_VERSION` `1.1.0` sang `1.2.0`, cho các
  `legal_facts.jsonl` đã trích xuất **trước khi** có
  `document_metadata.jsonl`. Nhận file fact và file metadata, dập lại
  `document_number` cùng ngày hiệu lực còn trống, tính lại `fact_id` và
  `fact_content_sha256`, ghi lại tại chỗ dưới schema mới — **không gọi lại LLM
  trích fact**. Facts trùng định danh sau khi tính lại được hợp nhất theo
  `fact_id`, đúng ngữ nghĩa định danh nội dung. File không phải schema `1.1.0`
  bị từ chối, nên chạy lại lần hai không làm hỏng dữ liệu đã migrate.

- `legal-rebind-fact-evidence` (`fact_evidence_rebinding_cli.py`): migration
  không gọi LLM cho fact schema hiện hành đã được tạo trước contract exact
  binding. Nhận một file fact và file chunk cùng văn bản, tách citation ghép
  paragraph thành các span nguyên văn, giữ nguyên `fact_id` và tính lại
  `fact_content_sha256`. Citation không thể bind chính xác làm command lỗi
  thay vì sửa gần đúng.

  Vì `document_number` nằm trong bộ trường định danh, migration này **đổi
  `fact_id` của mọi fact**. Nó chỉ an toàn khi chưa có artifact nào tham chiếu
  `fact_id` (`evidence_relations.jsonl`, `scenarios.jsonl`, `single_turn.jsonl`);
  nếu đã có thì các artifact đó phải sinh lại.

Hạ tầng dùng chung mới phát sinh khi triển khai, ngoài phạm vi ban đầu của
Phase 1 nhưng cần thiết để nó chạy được:

- `baml_transport.py` — tách phần gọi HTTP non-streaming và dựng
  `ClientRegistry` ra khỏi `parser/image_captioner.py`, vì đây đã là lần
  dùng thứ ba (chunk quality, fact extraction, image caption) của cùng một
  khối logic.
- `chunker/writer.py` bổ sung `read_chunks` — trước đó chunker chỉ ghi output
  cuối, chưa có gì đọc lại `*.chunks.jsonl`, nhưng CLI trích xuất fact cần
  đọc đúng file đó làm đầu vào.

Test: `tests/benchmark_synthesis/test_fact_extraction.py`,
`test_chunk_screening.py`, `test_document_metadata.py`,
`test_evaluation_models.py`, `test_evaluation_writer.py`,
`tests/test_baml_transport.py`, `tests/test_chunk_screening_cli.py`,
`tests/test_document_metadata_cli.py`, `tests/test_fact_extraction_cli.py`,
`tests/test_extraction_progress_backfill_cli.py`,
`tests/test_legal_fact_migration_cli.py`,
`tests/test_fact_evidence_rebinding_cli.py`
(tích hợp CLI đầu-cuối với client LLM giả mỗi bên). Windowing test
trực tiếp không cần mock; các BAML client wrapper test theo mẫu mock
`execute_baml_request` trong `tests/parser/test_baml_captioning.py`.

### Phase 2 — Evidence graph và scenario sampling (mục 6, 7) ✅ Đã triển khai

- `evidence_graph.py` — `find_candidate_fact_pairs` tìm cặp fact ứng viên trong
  toàn bộ fact đã trích xuất (không giới hạn cùng văn bản, vì multi-hop cần
  đúng cặp khác văn bản) bằng **statement-only hybrid retrieval + standard
  RRF**. Dense BGE-M3 và BM25L tiếng Việt cùng index `statement`, mỗi leg lấy
  top 30 rồi RRF không trọng số lấy top 10 cho mỗi fact. Lexical token gồm âm
  tiết có dấu, từ ghép VnCoreNLP 1.2 và bigram. Lý do cần nhánh lexical: một điều
  khoản sửa đổi **chép lại
  gần nguyên văn** điều khoản gốc — `336/2025` và `168/2024` trùng nhau 47
  `provision_path` — nên quan hệ `amended_by` biểu hiện thành trùng lặp từ
  vựng, thứ mà dense embedding làm nhòe thành "đều nói về phạt giao thông".
  RRF chỉ dùng thứ hạng nên không phải chuẩn hóa hoặc cộng trực tiếp hai thang
  điểm cosine và BM25 khác nhau.

  **Vì sao hybrid search không rơi vào mục "Heuristics" của `AGENTS.md`.**
  Quy tắc "không regex/keyword matching/handcrafted score" nhắm vào việc dùng
  chúng để **tự ra quyết định ngữ nghĩa** thay cho LLM — ví dụ chấm điểm liên
  tục rồi so ngưỡng để quyết "chunk này có chế tài hay không" (chính điều đã
  bị bỏ ở `chunk_screening.py`, xem mục Phase 1). Hybrid search + RRF ở đây là
  kỹ thuật **retrieval-stage**: nó chỉ đề cử cặp `fact_id` nào đáng hỏi, còn
  quyết định "có quan hệ hay không, loại gì" luôn thuộc về
  `confirm_evidence_relations` (LLM), độc lập với cách cặp đó được đề cử ra
  sao. Sai ở bước đề cử chỉ gây bỏ sót một quan hệ thật (false negative),
  không thể tự tạo ra một quan hệ giả — khác với heuristic "embedding
  candidate" từng bị bỏ ở mục 2, nơi ngữ cảnh sai lọt thẳng vào cửa sổ đọc và
  có thể khiến LLM trích xuất fact bịa đặt từ đó.

  Cần nhánh lexical vì không có tín hiệu tất định nào nối hai fact với nhau —
  khác với chunk (có `previous_chunk_id` và trích dẫn "theo Điều X" để tra
  cứu), một `LegalFact` không mang trường tham chiếu chéo tới fact khác; so
  mọi cặp bằng LLM (596 fact ⇒ 177.310 lệnh gọi) cũng không khả thi. Hybrid
  search là công cụ đúng chỗ cho lớp bài toán "đề cử ứng viên trước khi LLM
  phán quyết", không phải ngoại lệ dò từng trường hợp. `confirm_evidence_relations`
  gọi BAML xác nhận từng cặp, hướng quan hệ và `bridge`; mỗi fact summary gồm
  cả `conditions` và exact quoted evidence. `classify_bundle_kind` áp ràng
  buộc cấu trúc — `amended_by` luôn là `temporal` bất kể văn bản, còn lại thì
  khác văn bản mới là `multi_hop`, cùng văn bản hạ xuống `synthesis` — biểu
  diễn trực tiếp bằng logic phân loại, không phải bước "hạ cấp" riêng sau khi
  đã tạo bundle; `sample_bundles` chặn trùng theo `τ`.
- `confirm_evidence_relations` theo đúng kỷ luật retry-rồi-skip của Phase 1/1.5:
  một lệnh gọi lỗi được thử lại tối đa `DEFAULT_RELATION_CONFIRMATION_MAX_RETRIES`
  lần rồi bỏ qua cặp đó; `on_pair_confirmed` được gọi sau **mỗi** cặp, dù cặp
  đó được xác nhận có quan hệ hay không. Khoá resume của CLI là
  `evidence_relation_progress.jsonl` (`EvidenceRelationProgressRecord`), không
  phải `evidence_relations.jsonl` — cùng lý do đã sửa ở `legal-extract-facts`:
  một cặp bị xác nhận "không liên quan" là kết quả hợp lệ nhưng không sinh ra
  `EvidenceRelation` nào, nên suy "cặp nào đã kiểm" từ file quan hệ sẽ khiến
  cặp đó bị gọi LLM lại ở mọi lần chạy sau. Candidate pairs được lưu trong
  cache fingerprinted; resume hợp lệ không gọi lại embedding hoặc tokenizer.
  Relation và progress cùng mang `run_id`; CLI từ chối output cũ nếu fact,
  embedding profile, tham số relation, endpoint, model hoặc prompt version đã
  đổi. Candidate pairs có artifact cache fingerprinted riêng để resume không
  chạy lại embedding toàn corpus. Candidate, relation và bundle đều dùng thứ
  tự tie-break ổn định theo ID.
- `scenario_sampling.py` — nạp taxonomy/persona/phân bố từ config, phân bổ theo
  trọng số bundle kind rồi cân bằng các trục bề mặt như mô tả ở mục 5. Với
  bundle nhiều fact, `as_of_date` chỉ được đặt khi nằm trong giao của mọi
  khoảng hiệu lực; nếu không có ngày chung thì để `null`.

**Hai điểm lệch so với kế hoạch ban đầu, có lý do rõ:**

- `ConfirmEvidenceRelationBatch(items) -> EvidenceRelationBatchResult[]` (mỗi
  item là một cặp `fact_a, fact_b`) **không**
  nhận `candidate_relation_type` như phác thảo gốc. Không có tín hiệu tất
  định nào để đoán trước loại quan hệ trước khi LLM đọc nội dung — nếu đoán
  bằng heuristic thì vi phạm chính câu "Không xác định quan hệ chỉ bằng độ
  tương đồng hoặc từ khóa" ở mục 4. LLM tự chọn loại quan hệ từ enum
  `RelationType`, trả `is_related=false` khi không đủ căn cứ thay vì trả
  `null` ở gốc kết quả — tránh sự mơ hồ khi bọc toàn bộ object trong nullable.
- **Phạm vi MVP của `sample_bundles`: bundle đúng 2 fact**, ứng với một quan
  hệ đã xác nhận, không nối chuỗi quan hệ thành bundle 3+ fact (dù mục 4 có
  nhắc "tổng hợp nhiều hành vi trong một tình huống"). Ràng buộc trùng `τ`
  vẫn đúng và có tác dụng ngay ở phạm vi này (chặn hai quan hệ trùng cùng một
  cặp fact); nối chuỗi lớn hơn để lại cho vòng sau khi có nhu cầu thật.
- **`sample_scenarios` chỉ sinh `answerable` và `partially_answerable`.**
  Bundle luôn dựng từ fact đã trích xuất — tức luôn có căn cứ thật — nên về
  mặt cấu trúc không thể sinh ra `underspecified` hay `unanswerable` từ đó được.
  Hai loại này cần một đường sinh khác, không dựa trên bundle (§8 phần
  multi-turn đã có manh mối: `unanswerable` cần người viết hoặc mining riêng,
  không sinh tự động từ evidence thật).

BAML: `evidence_relations.baml` — enum `RelationType` (8 giá trị khớp danh
sách quan hệ ở mục 4), class `LegalFactSummary`, `ConfirmedRelation`,
`EvidenceRelationBatchItem`, `EvidenceRelationBatchResult`, function
`ConfirmEvidenceRelationBatch` (xác nhận nhiều cặp fact trong một lời gọi LLM,
số cặp mỗi lô do `evidence_relation.batch_size` trong synthesis config quyết
định). Giá trị `RelationType` phía BAML dùng PascalCase
theo đúng quy ước `ChunkWindowRole`; phía Python dùng `Literal` snake_case
khớp ví dụ JSON ở mục 4 (`"amended_by"`), dịch qua bảng ánh xạ tường minh
trong `evidence_graph.py` — không suy luận chuỗi.

Config: `data/config/groundtruth/query_taxonomy.json`,
`data/config/groundtruth/personas.json`,
`data/config/groundtruth/scenario_distribution.json`,
`data/config/groundtruth/evolution.json` (file config được tạo ở Phase 2 theo
đúng mục 6; logic *áp dụng* evolution vẫn thuộc Phase 3).

CLI: `legal-build-evidence-graph` (`evidence_graph_cli.py`) — đọc mọi fact từ
`legal_facts.jsonl`, xác nhận quan hệ chưa được kiểm (đọc resume từ
`evidence_relation_progress.jsonl`), lấy mẫu bundle, sinh scenario, ghi
`evidence_relations.jsonl` và `scenarios.jsonl`. `scenarios.jsonl` là output
mới phát sinh khi triển khai, chưa có trong cấu trúc output đề xuất ban đầu
ở mục 17 — bổ sung vì Phase 3 cần scenario đã chốt và ổn định, không tính
lại mỗi lần sinh candidate. `evidence_relations.jsonl` được ghi lại (atomic)
sau **mỗi cặp** được xác nhận, theo thứ tự quan hệ trước rồi mới đến progress
— giống hệt thứ tự đã chọn ở `legal-extract-facts` — nên crash giữa hai lần
ghi chỉ khiến cặp đó bị kiểm lại, không bao giờ mất quan hệ đã xác nhận.
Bundle sampling và scenario sampling là code tất định, chạy lại toàn bộ mỗi
lần trên tập quan hệ hiện có — không cần resume riêng vì không tốn LLM call.

Test: `test_evidence_graph.py` (bao gồm retry/skip/callback theo mẫu
`test_fact_extraction.py`), `test_scenario_sampling.py` (weighted bundle-kind,
giao khoảng hiệu lực và luân phiên theo trục ít dùng nhất),
`test_evaluation_writer.py` (round-trip
`EvidenceRelationProgressRecord`), `tests/test_evidence_graph_cli.py` (tích
hợp CLI đầu-cuối: resume bỏ qua cặp đã kiểm, ghi tiến độ khi bị ngắt giữa
chừng, không gọi lại LLM cho cặp đã xác nhận "không liên quan").

### Phase 3 — Sinh single-turn theo batch (mục 8) ✅ Đã triển khai

- `candidate_generation.py` — `generate_candidate_batch` gọi BAML sinh tối đa
  `candidate_generation.batch_size` candidate từ các `Scenario` (Phase 2) +
  evidence đã chốt trong một request; mỗi claim chỉ tham chiếu `chunk_id` và
  code chỉ giữ các id thuộc evidence thật của fact. `answer_fields` chép trực
  tiếp từ `LegalFact`, không đi qua LLM. `evolve_candidate_batch` áp từng
  phép evolution cho cả batch, giữ nguyên claims/evidence/answer_fields.
- `candidate_pipeline.py` chia các task chưa hoàn thành thành batch 5, chạy
  tối đa 10 batch đồng thời. Generation và từng vòng evolution đều
  checkpoint riêng từng `task_id`; một batch lỗi sau retry không đánh dấu
  các task đó là hoàn thành.

BAML: `candidate_generation.baml` — `GenerateSingleTurnCandidateBatch` nhận
`CandidateGenerationBatchInput[]`; `EvolveCandidateBatch` nhận
`CandidateEvolutionBatchInput[]`. Mỗi output giữ nguyên `task_id` để Python
ghép lại đúng scenario dù model thay đổi thứ tự phần tử.

CLI: `legal-generate-candidates` (`candidate_generation_cli.py`) — đọc
`legal_facts.jsonl` và `scenarios.jsonl`; với mỗi scenario sinh
`--candidates-per-scenario` attempt độc lập, chạy generation → evolution rồi
ghi candidate hoàn thành. Batch size và concurrency lấy từ
`synthesis_v1.json`, không phải tham số dòng lệnh.

Test: `test_candidate_generation.py`, `test_candidate_pipeline.py` (batch
generation/evolution, bounded concurrency, retry và resume) và
`tests/test_candidate_generation_cli.py` tích hợp CLI đầu-cuối.

### Phase 4 — Pilot và ổn định schema (mục 9) ✅ Đã hoàn tất

Không có code review trong phase này. Pipeline đã chạy pilot single-turn và
khóa schema candidate hiện tại. Review bằng agent bên ngoài là hoạt động độc
lập, không phải điều kiện để chuyển sang Phase 5.

### Phase 5 — Export evaluation ground truth ✅ Đã triển khai

- `groundtruth_export.py` — ánh xạ candidate sang schema evaluation nhưng giữ
  lại metadata synthesis.
- `groundtruth_export_cli.py` — đọc `candidates/single_turn.jsonl` và ghi
  nguyên tử `ground_truth.jsonl` qua lệnh `legal-export-groundtruth`.

### Các phase mở rộng đã loại khỏi phạm vi hiện tại

Các thiết kế dưới đây chỉ còn là ghi chú lịch sử, không thuộc deliverable
hiện tại.

#### Hard negatives và difficulty

- `hard_negatives.py` — chạy retriever, loại trước các chunk đã thuộc
  evidence set của bất kỳ fact nào bằng tập hợp, gọi BAML xác nhận negative
  thật sự không đủ căn cứ.
- `leakage.py` — chạy candidate qua ba chế độ closed-book / oracle /
  oracle-trộn-nhiễu, ghi leakage score và difficulty vào candidate.

#### Multi-turn simulator

- `multiturn_simulator.py` — vòng lặp hai vai: goal state dựng từ
  `required_claims` của scenario, simulator tiết lộ dần, oracle chỉ đọc gold
  evidence, cập nhật `state_delta` sau mỗi lượt, dừng khi đạt goal hoặc chạm
  `max_simulator_turns`.

BAML mới: `multiturn_simulation.baml` — function `SimulateUserTurn(goal_state,
history) -> UserTurn`, function `GenerateOracleResponse(gold_evidence,
history) -> OracleTurn`.

Các lượt `underspecified`/`unanswerable` không sinh bằng simulator — quy
trình thủ công nằm ngoài phạm vi code, không cần module.

#### Split, coverage và đánh giá mở rộng

- `dedup.py` — cụm embedding cosine `dedup_cosine_threshold`, gán
  `salience_weight`.
- `coverage.py` — báo cáo tỷ lệ fact có candidate, danh sách fact chưa phủ.
- `splitting.py` — chia `dev`/`test`/`hidden_test` theo thành phần liên thông
  của evidence graph chứa `fact_id`, không theo từng candidate.

Phần đánh giá retrieval/generator/end-to-end hiện dùng metric tất định thuần
Python trong `src/legal_rag/evaluation/`; nếu sau này thêm metric LLM-judge thì
cần khai báo dependency trực tiếp ở thời điểm đó.

### Tổng hợp thay đổi ngoài `benchmark_synthesis/`

- **Bỏ fact ledger và bước chuyên gia duyệt tay ở tầng fact** (mục 3 ở trên):
  xóa `fact_ledger.py`, `fact_review_cli.py` (và script `legal-review-facts`
  trong `pyproject.toml`), `tests/benchmark_synthesis/test_fact_ledger.py`.
  `fact_extraction.py` không còn `FactExtractionError`/`FactExtractionRejection`
  — `build_legal_fact` tra `chunks_by_id[chunk_id]` trực tiếp, để `KeyError`
  raise tự nhiên nếu LLM trích dẫn sai, thay vì validate và loại fact.
  `writer.py` bỏ `read_fact_ledger`/`write_fact_ledger`. `binding_gate.py`
  (Cổng 1 ở tầng candidate, Phase 3) bỏ tham số `ledger_by_fact_id` và kiểm
  tra "fact approved" — vẫn giữ nguyên kiểm tra `chunk_id` thuộc evidence set
  thật và span-match, đây là quyết định tách biệt: chỉ bỏ tầng duyệt người,
  không bỏ kiểm tra tất định ở tầng candidate. `evidence_graph_cli.py` và
  `candidate_generation_cli.py` không còn lọc theo `approved`, dùng thẳng
  mọi fact trong `legal_facts.jsonl`. ✅ **Đã áp dụng.**
- `pyproject.toml`: extra `evaluation` đổi thành `rapidfuzz`,
  `sentence-transformers` — gỡ `deepeval`, `ragas`, `langchain-community`,
  `langchain-openai` (không còn code nào import sau khi pipeline Ragas cũ bị
  xóa); thêm console script mới cho từng CLI ở các phase sau. ✅ **Đã áp
  dụng** ở Phase 0.
- `benchmark_synthesis/config.py`: thêm `SynthesisSettings` đọc từ
  `data/config/groundtruth/synthesis_v1.json`; bỏ synthesis tunables khỏi
  environment settings. ✅ **Đã áp dụng.**
- `io_utils.py`: `write_jsonl_atomic` dùng chung; `parser/writer.py` và
  `chunker/writer.py` refactor gọi vào đây. ✅ **Đã áp dụng.**
- `embedding/encoder.py`: `EmbeddingEncoder` bọc `sentence-transformers`,
  nạp lazy để không bắt buộc cài đặt khi không dùng. ✅ **Đã áp dụng.**
- `data/evaluation/evaluation-data.md`: viết lại theo pipeline mới hoặc xóa.
  ✅ **Đã áp dụng** ở Phase 0.
- `src/legal_rag/evaluation/` và `data/evaluation/` đổi tên thành
  `src/legal_rag/benchmark_synthesis/` và `data/benchmark_synthesis/`, còn
  data guide trung gian đã được xóa. `evaluation_cli.py` đổi tên thành
  `fact_extraction_cli.py`, để tên khớp mục đích thật (sinh benchmark) thay vì
  đọc như đo lường RAG. ✅ **Đã áp dụng.**
- `baml_transport.py`: tách khỏi `parser/image_captioner.py` khi cần dùng lại
  cho `chunk_screening.py`/`fact_extraction.py`. ✅ **Đã áp dụng** ở Phase 1,
  không nằm trong kế hoạch gốc.
- `chunk_screening.py` (trước đó `chunk_quality.py`): đổi từ chấm 4 tiêu chí
  liên tục + so ngưỡng sang hỏi nhị phân `has_penalty_content` — LLM đáng tin
  hơn với quyết định nhị phân so với tự hiệu chỉnh điểm số liên tục. Bỏ
  `chunk_quality_threshold` khỏi synthesis config.
  ✅ **Đã áp dụng.**
- `chunker/writer.py`: thêm `read_chunks` — CLI trích xuất fact cần đọc lại
  `*.chunks.jsonl`. ✅ **Đã áp dụng** ở Phase 1, không nằm trong kế hoạch gốc.
- `pyproject.toml`: thêm console script `legal-extract-facts`,
  `legal-review-facts` (Phase 1), `legal-build-evidence-graph` (Phase 2).
  ✅ **Đã áp dụng.**
- `data/config/groundtruth/`: năm file config versioned
  (`query_taxonomy.json`, `personas.json`, `scenario_distribution.json`,
  `evolution.json`, `synthesis_v1.json`). ✅ **Đã áp dụng.**
- `synthesis_v1.json`: cấu hình riêng cho generation, critique, evolution và
  verification thay vì dùng environment variables hoặc dùng chung ngầm định.
  ✅ **Đã áp dụng.**
- `scenario_sampling.py`: thêm `EvolutionConfig`/`load_evolution_config`,
  `load_query_type_descriptions`, `load_persona_descriptions` — thuần cộng
  thêm, không sửa các hàm/kiểu dữ liệu đã có từ Phase 2. ✅ **Đã áp dụng** ở
  Phase 3.
- `pyproject.toml`: thêm console script `legal-generate-candidates`
  (Phase 3). ✅ **Đã áp dụng.**
- Tách cổng sàng lọc chunk ra khỏi `legal-extract-facts` thành CLI riêng
  `legal-screen-chunks` (`chunk_screening_cli.py`), ghi kết quả ra
  `chunk_screening.jsonl` để `legal-extract-facts` đọc lại — thay vì gọi
  ngầm trong cùng một lần chạy như trước, để bước sàng lọc có audit trail
  riêng và các phase nối nhau nhất quán qua file, giống mọi ranh giới phase
  khác trong pipeline. `chunk_screening.py`: `screen_chunks()` đổi trả về
  từ `(accepted, excluded)` sang `ChunkScreeningRecord` cho mọi chunk (kể cả
  chunk được chấp nhận), việc lọc `has_penalty_content=true` chuyển sang CLI.
  `writer.py`: thêm `write_chunk_screening`/`read_chunk_screening`.
- `pyproject.toml`: thêm console script `legal-screen-chunks`. ✅ **Đã áp
  dụng**, không nằm trong kế hoạch gốc.
- Tách metadata cấp văn bản (`document_number`, ngày hiệu lực) ra khỏi output
  của LLM trích fact thành bước riêng `legal-resolve-document-metadata`
  (`document_metadata.py`, `document_metadata_cli.py`,
  `baml_src/document_metadata.baml`), vì đây là thuộc tính của văn bản mà cửa
  sổ đọc cấp chunk không chứa. `models.py`: thêm `build_legal_fact` làm
  constructor duy nhất tính `fact_id`/`fact_content_sha256`, để bước trích fact
  và migration dùng chung một công thức băm; `LEGAL_FACT_SCHEMA_VERSION` lên
  `1.2.0`. `synthesis_v1.json`: thêm giới hạn output, temperature và số chunk
  mở/đóng cho metadata. `writer.py`: thêm
  `write_document_metadata`/`read_document_metadata`. `pyproject.toml`: thêm
  console script `legal-resolve-document-metadata` và
  `legal-migrate-legal-facts`. ✅ **Đã áp dụng**, không nằm trong kế hoạch gốc.
- `find_candidate_fact_pairs` đổi từ dense-only sang hybrid (dense + BM25) hợp
  nhất bằng RRF, để bắt được quan hệ `amended_by` biểu hiện qua trùng lặp
  nguyên văn giữa văn bản sửa đổi và văn bản gốc. Dense embedding sau đó được
  chuyển sang BGE-M3 qua provider riêng; BM25 dùng tokenizer hybrid tiếng Việt
  được khóa bằng VnCoreNLP 1.2 và chạy theo batch khi thực sự phải tạo candidate cache.
  ✅ **Đã áp dụng**, không nằm trong kế hoạch gốc.
- Tách khoá resume của `legal-extract-facts` khỏi `source_chunk_id` của fact
  sang artifact riêng `extraction_progress.jsonl`, vì một Target trả về danh
  sách rỗng là kết quả hợp lệ nhưng không để lại fact nào để suy ra.
  `fact_extraction.py`: thêm `ExtractionProgressRecord`; `writer.py`: thêm
  `write_extraction_progress`/`read_extraction_progress`; `pyproject.toml`:
  thêm console script `legal-backfill-extraction-progress` cho dữ liệu đã
  trích trước thay đổi này. ✅ **Đã áp dụng**, không nằm trong kế hoạch gốc.
