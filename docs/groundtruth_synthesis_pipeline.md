# Ground-Truth Synthesis Pipeline

## Mục tiêu và phạm vi

Pipeline này sinh dữ liệu ground truth để đánh giá hệ thống RAG trong phạm vi
**xử phạt vi phạm giao thông đường bộ tại Việt Nam**. Benchmark hỗ trợ cả câu
hỏi single-turn và hội thoại multi-turn, đồng thời đánh giá riêng retrieval,
generation và toàn bộ luồng RAG.

Dữ liệu do LLM sinh chỉ là candidate. Chỉ những mẫu đã vượt qua kiểm tra tự
động mới được đưa vào ground truth.

## Nguyên tắc kiến trúc: tin tưởng LLM có kiểm tra tất định

**Đã đổi so với thiết kế ban đầu:** bản đầu tài liệu này đặt điểm phê duyệt
con người tại từng fact pháp lý qua một fact ledger (`extracted →
span_verified → needs_expert_review → approved | rejected`, cộng `stale`).
Quyết định hiện tại: tin tưởng hoàn toàn output của LLM ở tầng trích xuất
fact — bỏ hẳn fact ledger và bước chuyên gia duyệt tay. Không còn khái niệm
fact `approved`/`stale`; mọi fact trong `legal_facts.jsonl` được Phase 2/3
dùng thẳng, không lọc theo trạng thái nào.

Hai cổng chất lượng còn lại vẫn tất định, không cần người, áp dụng ở tầng
candidate (không phải tầng fact):

```text
Cổng 1  Ràng buộc căn cứ tất định   không dùng LLM, không dùng người
Cổng 2  Kiểm chứng LLM độc lập      dùng LLM khác generator
```

Một candidate chỉ hợp lệ khi mọi căn cứ của nó nằm trong evidence set thật của
một fact đã trích xuất.

```text
.chunks.jsonl
    -> Legal facts (typed output, kèm quoted span)
    -> Evidence graph trên mọi fact đã trích xuất
    -> Scenario sampling (persona x style x length)
    -> Single-turn candidates
    -> Ràng buộc căn cứ tất định                    [cổng 1]
    -> Kiểm chứng LLM độc lập                       [cổng 2]
    -> Multi-turn: simulator hướng mục tiêu + người cho lượt khó
    -> Hard negatives + difficulty (cần retriever)
    -> Review theo rủi ro trên mẫu
    -> Versioned ground_truth.jsonl
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
      "chunk_id": "...",
      "quoted_span": "..."
    }
  ]
}
```

Nguyên tắc:

- Một fact chỉ chứa nội dung có bằng chứng trực tiếp.
- Không sử dụng kiến thức sẵn có của model làm căn cứ.
- Nếu một fact cần nhiều chunk thì phải lưu tất cả chunk cần thiết.
- Mỗi `evidence` phải kèm `quoted_span` trích nguyên văn từ `text` của chunk.
- `chunk_id` phục vụ retrieval evaluation, lấy trực tiếp từ trích dẫn của LLM,
  không tra cứu lại `unit_ids`/`source_refs` từ chunk gốc.

### Sàng lọc chunk trước khi trích xuất

Trước khi trích xuất fact, một critic model rẻ trả lời một câu hỏi nhị phân
cho mỗi chunk: đoạn này có chứa nội dung chế tài cụ thể (hành vi, mức phạt,
thẩm quyền, thủ tục, đối tượng áp dụng) hay không. Chunk bị đánh giá không
có nội dung chế tài bị bỏ qua và ghi vào danh sách loại trừ kèm lý do do
chính model đưa ra — thường là mục lục, tiêu đề chương, đoạn dẫn nhập không
mang chế tài.

Các chunk được gộp theo batch (mặc định 8 chunk/lần gọi) thay vì gọi LLM
riêng cho từng chunk, để giảm số lần round-trip. Mỗi chunk trong batch mang
theo `chunk_id`, và model phải trả về đúng `chunk_id` đó trong từng kết quả —
kết quả được đối chiếu ngược lại theo `chunk_id`, không dựa vào thứ tự trả về,
để tránh lệch kết quả giữa các chunk khi batch lớn.

Cổng rẻ này đứng trước để bảo vệ hai công đoạn đắt phía sau: LLM call trích
xuất và thời gian chuyên gia duyệt fact. Câu hỏi nhị phân này thay cho cách
chấm điểm liên tục rồi so ngưỡng: LLM đáng tin hơn nhiều với quyết định
nhị phân rõ ràng so với việc tự hiệu chỉnh điểm số liên tục 0.0-1.0 (dễ bị
dồn cụm, khiến một ngưỡng cố định gần như vô nghĩa trong thực tế). Danh sách
chunk bị loại phải audit được — lý do lấy trực tiếp từ model, không phải câu
chữ tính toán từ điểm số.

### Gộp chunk khi một fact cần nhiều nguồn

Một hành vi có thể có căn cứ nằm rải trên nhiều chunk, theo hai kiểu khác nhau,
xử lý khác nhau:

- **Chunk liền kề bị cắt bởi giới hạn token.** Chunker cắt một điều khoản dài
  thành hai chunk theo `previous_chunk_id`/`next_chunk_id` đã có sẵn trong
  schema chunk. Gộp bằng cách kéo thêm chunk liền trước/sau vào cùng cửa sổ
  đọc, thuần cơ học, không cần LLM đoán.
- **Chunk không liền kề nhưng cùng một hành vi.** Ví dụ mức phạt tiền nằm ở
  Điều 6 nhưng số điểm trừ giấy phép quy định ở một điều hoặc phụ lục khác.
  Hai cơ chế bổ sung, tăng dần độ tin cậy cần thiết:
  1. **Trích dẫn chéo tường minh.** Nếu chunk chứa cụm dẫn chiếu kiểu "theo
     Điều X", "quy định tại khoản Y Điều Z", trích các số điều/khoản này bằng
     regex — đây là phân tích từ vựng nhỏ, được phép theo AGENTS.md — rồi tra
     `provision_path` tương ứng để lấy đúng chunk đó vào cùng cửa sổ đọc.
     Tất định, không cần LLM cho bước tìm kiếm.
  2. **Tương đồng embedding trong cùng văn bản.** Khi không có trích dẫn
     tường minh, lấy các chunk gần nghĩa nhất **trong cùng `source_name`**
     làm ứng viên đưa vào cửa sổ đọc. Đây là gợi ý, không phải căn cứ: LLM
     trích xuất fact vẫn phải tự quyết có dùng ứng viên này không, và phải để
     lại `quoted_span` nếu dùng. Không mở rộng tìm kiếm sang văn bản khác ở
     bước này — quan hệ khác văn bản thuộc về evidence graph ở mục 4.

### Định danh fact tách khỏi nội dung fact

`fact_id` là hash trên các trường **định danh**: `conduct`, `subject`,
`document_number`, `provision_path`. `fact_content_sha256` là hash trên **toàn
bộ** nội dung fact — giữ lại để phát hiện nội dung fact đổi giữa các lần trích
xuất, dù không còn gắn với trạng thái duyệt nào (xem "Nguyên tắc kiến trúc" ở
đầu tài liệu).

Cặp này lặp lại đúng kỷ luật mà chunker đang dùng cho `chunk_id` và
`content_sha256`: trích xuất lại corpus không làm đổi `fact_id` khi cách diễn
đạt thay đổi, vì `fact_id` chỉ phụ thuộc các trường định danh.

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
  "confirmed_by_model": "..."
}
```

`bridge` được đưa vào prompt sinh câu hỏi multi-hop làm chủ đề bắc cầu. Không
có nó, câu multi-hop thường bị rút gọn thành single-hop vì một chunk đã đủ trả
lời. Với quan hệ temporal, `bridge` chính là điều khoản bị sửa đổi.

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

Ba trục trên chỉ đổi bề mặt câu hỏi, không đổi evidence, nên không làm tăng
chi phí review.

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

Đặt quota theo cụm fact `số mẫu mỗi cụm = ceil(tổng mẫu / số cụm)`, sau đó
chọn tham lam theo thứ tự ưu tiên: cặp persona và fact chưa từng xuất hiện,
rồi `query_style` đang ít, rồi `query_length` đang ít. Cách này đạt độ phủ ngay
tại lúc sinh, không cần loại bỏ hậu kiểm.

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
        { "chunk_id": "chunk_1", "quoted_span": "..." }
      ]
    },
    {
      "claim": "Số điểm giấy phép lái xe bị trừ",
      "fact_id": "fact_1",
      "evidence": [
        { "chunk_id": "chunk_2", "quoted_span": "..." }
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
  một model, không chỉ đa dạng hóa phía verifier.
- `answer_fields` — mức phạt tiền, số điểm trừ, hình phạt bổ sung, biện pháp
  khắc phục — được pipeline **chép cơ học từ fact đã duyệt**, không để LLM
  sinh lại. Đây là phần đáp án so khớp tất định được khi đánh giá.
- Candidate kèm `reasoning_vi`: chuỗi lập luận nối evidence tới đáp án. Trường
  này tăng tốc review và cho phép đánh giá tính minh bạch của mẫu.

Chi phí sinh rẻ hơn chi phí kiểm chứng nhiều lần, nên sinh dư rồi lọc mạnh tốt
hơn sinh ít bằng pipeline nhiều bước.

### Vòng tinh chỉnh bề mặt có giới hạn

Trước khi vào cổng 1, một critic chấm câu hỏi kèm feedback; nếu dưới ngưỡng,
**chỉ bề mặt câu hỏi** được viết lại theo feedback, tối đa
`max_quality_retries` lần. Claims, evidence và `answer_fields` bị đóng băng
trong suốt vòng này.

Vòng tinh chỉnh thuộc công đoạn sinh và mọi lần viết lại đều được ghi log,
nên không mâu thuẫn với nguyên tắc kiểm chứng không tự sửa: cổng 1 và cổng 2
vẫn chỉ loại, không sửa.

## 7. Ràng buộc căn cứ tất định

Cổng 1. Chạy trước mọi kiểm chứng bằng LLM, không tốn LLM call.

Với mỗi phần tử `evidence` của mỗi claim:

1. `chunk_id` phải nằm trong evidence set của fact `fact_id`.
2. `quoted_span` phải khớp `text` của chunk theo Levenshtein partial ratio với
   ngưỡng tối thiểu `0.85`.

Candidate bị loại cơ học nếu bất kỳ điều kiện nào sai. Điểm grounding của
candidate là trung bình partial ratio trên toàn bộ span, lưu vào
`verification_results` để phân tích.

Đây là kiểm tra **từ vựng** về tính xác thực của trích dẫn, không phải phán
đoán ngữ nghĩa. Câu hỏi "căn cứ này có đủ để kết luận không" vẫn thuộc về LLM
verifier ở mục 10. Ranh giới này phải giữ đúng: không mở rộng cổng 1 thành
rule engine cho các quyết định ngữ nghĩa.

Cổng này bắt đúng dạng lỗi mà LLM verifier yếu nhất là bịa căn cứ, với chi phí
gần bằng không. Ngưỡng `0.85` nằm trong cấu hình có phiên bản.

## 8. Sinh multi-turn candidates

Multi-turn phát triển từ các fact đã duyệt, không sinh hội thoại tự do từ đầu.

Bằng chứng từ các benchmark multi-turn đã công bố cho thấy hội thoại sinh hoàn
toàn tự động thất bại ở đúng những chỗ cần nhất: câu unanswerable sinh tự động
chất lượng thấp, và độ đa dạng evidence trong hội thoại synthetic thấp hơn
nhiều lần so với hội thoại người viết. Vì vậy phân tầng:

- Lượt `answerable` và `partially_answerable`: sinh tự động, qua cổng 1 và 2.
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
   RAG, lỗi của nó sẽ thấm vào ground truth. Lượt trả lời của oracle qua cổng
   1 như mọi candidate khác.
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

## 9. Khai thác hard negatives

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

## 10. Kiểm chứng LLM độc lập

Cổng 2. Dùng model hoặc prompt khác với generator để giảm self-confirmation
bias. Chỉ chạy trên candidate đã qua cổng 1.

Năm chiều kiểm chứng:

1. **Stand alone** — câu hỏi có tự đứng vững không, hoặc với multi-turn thì
   `standalone_query` có giữ nguyên ý nghĩa của lượt không.
2. **Question specificity** — câu hỏi có đủ cụ thể để có một đáp án xác định,
   hay quá chung chung.
3. **Question grounding** — evidence có đủ để trả lời câu hỏi không.
4. **Answer grounding** — mọi claim trong đáp án có được evidence hỗ trợ không,
   và đáp án có bỏ sót chế tài bắt buộc nào không.
5. **Domain và temporal** — câu hỏi có thuộc phạm vi xử phạt giao thông đường
   bộ không, `as_of_date` có nằm trong khoảng hiệu lực của mọi fact liên quan
   không, và có nhầm hành vi, đối tượng hoặc phương tiện không.
6. **Hop necessity** — với bundle nhiều fact: nếu bỏ một fact mà đáp án vẫn
   đủ căn cứ thì mẫu không thật sự cần số hop đã khai. Verifier gán lại
   `reasoning_hops` theo mức thật. Đây là cơ chế bắt shortcut — lỗi phổ biến
   nhất của câu hỏi multi-hop sinh tự động.

Với multi-turn kiểm thêm: một lượt có sử dụng thông tin từ lượt tương lai
không. Với hard negatives kiểm thêm: negative có thực sự không đủ căn cứ không.

### Gán nhãn lại sau khi sinh

Nhãn `query_type` và `answerability` đặt trong scenario là nhãn **đặt hàng**,
không phải nhãn thật. Thực nghiệm trên các pipeline tương tự cho thấy nhãn
quan sát được lệch đáng kể so với nhãn yêu cầu.

Verifier phải gán nhãn lại một cách độc lập. Ground truth lưu nhãn quan sát
được. Chênh lệch giữa nhãn đặt hàng và nhãn quan sát được ghi vào coverage
report để hiệu chỉnh phân bố ở lần sinh sau.

Candidate không đạt bị loại hoặc chuyển sang trạng thái cần review. Pipeline
không tự động sửa và che giấu lỗi.

## 11. Đo leakage và difficulty

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

## 12. Kiểm soát độ phủ trên bảng fact

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

## 13. Review theo rủi ro

Cổng 3 đã tiêu thụ phần lớn ngân sách chuyên môn tại mục 3. Ở tầng candidate,
mục tiêu không phải duyệt hết mà là **đo được tỷ lệ lỗi**.

Luồng trạng thái của candidate:

```text
generated
    -> binding_verified        (qua cổng 1)
    -> automatically_verified  (qua cổng 2)
    -> sampled_for_review | accepted
    -> approved | rejected
```

Bắt buộc đưa vào review:

- Candidate phái sinh từ fact vừa đổi `fact_content_sha256`
- Toàn bộ lượt `underspecified` và `unanswerable`
- Toàn bộ candidate multi-hop và temporal
- Candidate có điểm grounding sát ngưỡng `0.85`

Phần còn lại lấy mẫu ngẫu nhiên với cỡ mẫu đủ để ước lượng tỷ lệ lỗi. Tỷ lệ
lỗi ước lượng được công bố cùng mỗi bản phát hành ground truth.

Người duyệt ở tầng này không kiểm lại nội dung pháp lý của fact, vì việc đó đã
xong ở mục 3. Họ chỉ xác nhận:

- Tình huống hợp lý và không mơ hồ
- Đáp án không bỏ sót chế tài mà fact đã nêu
- Hard negatives không phải căn cứ thay thế
- Hội thoại multi-turn không tự mâu thuẫn

## 14. Chia dataset

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
sample_id
conversation_id
turn_id
interaction_type
scenario_id
query_vi
standalone_query
reference_answer_vi
required_claims          (claim, fact_id, evidence[chunk_id, quoted_span])
answer_fields            (chép cơ học từ fact, so khớp tất định)
reasoning_vi
evolutions               (chuỗi phép evolution đã áp dụng)
gold_evidence
hard_negative_chunk_ids
query_type
query_form
persona
query_style
query_length
answerability            (bốn giá trị)
answerability_observed
as_of_date
reasoning_hops
difficulty
salience_weight
grounding_score
source_fact_ids
source_fact_content_sha256
corpus_snapshot_id
generator_model
generator_prompt_version
verifier_model
verification_results
review_status
goal_completion          (mức hội thoại, từ simulator)
stop_reason              (mức hội thoại, từ simulator)
```

`source_fact_content_sha256` cho phép phát hiện candidate `stale` khi fact
nguồn thay đổi, mà không phải sinh lại toàn bộ dataset.

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
├── legal_facts.jsonl
├── evidence_relations.jsonl
├── scenarios.jsonl
├── candidates/
│   ├── single_turn.jsonl
│   └── multi_turn.jsonl
├── reviews/
│   └── reviewed_candidates.jsonl
├── ground_truth/
│   ├── dev.jsonl
│   ├── test.jsonl
│   └── hidden_test.jsonl
└── reports/
    ├── generation_report.json
    ├── coverage_report.json
    └── binding_report.json
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
3. Implement lọc chất lượng chunk và legal-fact extraction kèm `quoted_span`.
   ✅ **Đã xong.**
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
   2 và gán nhãn lại. ✅ **Đã xong.** Generator và verifier còn dùng chung một
   model (khác prompt); multi-model để giảm bias sâu hơn còn để lại sau.
9. Chạy pilot, đo tỷ lệ lỗi trên mẫu review, ổn định schema candidate.
10. Implement retriever. Đây là điều kiện chặn cho hai mục kế tiếp.
11. Implement hard-negative mining, difficulty scoring và leakage analysis.
12. Implement multi-turn bằng user simulator hướng mục tiêu với oracle bám
    gold evidence; người viết hoặc sửa các lượt `underspecified` và
    `unanswerable`.
13. Xuất các split theo `fact_id` và implement đánh giá retrieval,
    oracle-generator, oracle trộn nhiễu và end-to-end.

Không bắt đầu sinh dữ liệu hàng loạt trước khi schema fact và temporal
metadata đã ổn định.

## Kế hoạch triển khai kỹ thuật

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
(fact → duyệt → evidence graph → scenario → candidate), không đo gì cả. Việc
đo lường thật (Recall@k, Faithfulness, Answer correctness... qua `deepeval`,
mục 16) là một mối quan tâm khác, chưa có code, sẽ cần vị trí riêng của nó khi
tới Phase 8. Do đó `src/legal_rag/evaluation/` và `data/evaluation/` được đổi
tên thành `src/legal_rag/benchmark_synthesis/` và `data/benchmark_synthesis/`
— khớp với mục đích thật (sinh benchmark) và không còn xung đột tên với phần
đánh giá RAG thật sẽ thêm sau. Extra `evaluation` trong `pyproject.toml`
(nhóm dependency: `deepeval`, `rapidfuzz`, `sentence-transformers`) giữ
nguyên tên, vì nó sẽ tiếp tục phục vụ cả code synthesis hiện tại lẫn code
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

**Dependency mới: `rapidfuzz`.** Cổng 1 cần Levenshtein partial ratio để
kiểm `quoted_span` khớp với text chunk. `difflib` trong thư viện chuẩn không
có phép so khớp "chuỗi con tốt nhất" đúng ngữ nghĩa partial ratio; tự viết lại
phép này là nguồn lỗi tinh vi. `rapidfuzz` nhỏ, thuần Python/C, không kéo
theo dependency nặng — hợp lý theo tiêu chí "thư viện hiện có không đủ" của
AGENTS.md. Đặt trong extra `evaluation`, không phải dependency chính: toàn bộ
pipeline synthesis vốn đã được tách khỏi core (parser/chunker) qua extra này,
nên `rapidfuzz` đi cùng chỗ với `deepeval` và `sentence-transformers` thay vì
buộc mọi người dùng `legal-parse`/`legal-chunk` phải cài thêm nó.

**Atomic JSONL writer dùng chung.** `parser/writer.py` và `chunker/writer.py`
hiện có cùng một khối logic ghi tệp tạm rồi `replace()` nguyên tử, lặp lại y
hệt. Thêm một module thứ ba (`benchmark_synthesis/writer.py`) sẽ là lần lặp
thứ ba — đủ điều kiện theo AGENTS.md để tách hàm dùng chung. Thêm
`src/legal_rag/io_utils.py` chứa `write_jsonl_atomic(path, records)`, refactor
hai module cũ gọi vào đây, rồi `benchmark_synthesis/writer.py` dùng lại. Đây là thay
đổi cross-cutting nhỏ, nên làm thành một commit riêng trước khi viết code
Phase 1.

**Cấu hình.** Thêm `EvaluationSettings` vào `settings.py`, theo đúng mẫu
`ChunkerSettings` — nạp từ biến môi trường:
`span_match_threshold` (mặc định `0.85`), `bundle_overlap_threshold` (`τ`),
`max_quality_retries`.
Danh sách persona, query taxonomy, phân bố scenario và evolution nằm ở file
JSON versioned dưới `data/config/groundtruth/`. **Đã đổi so với dự tính ban
đầu:** không dùng `importlib.resources` như `embedding/profiles.py`, vì
`data/config/` nằm ngoài cây package `src/legal_rag/` nên không phải là tài
nguyên gói hợp lệ. Các CLI nhận đường dẫn config qua tham số dòng lệnh tường
minh (`--query-taxonomy`, `--personas`, `--scenario-distribution`), đọc bằng
`Path.read_text` + `json.loads` thuần — đúng phong cách các CLI hiện có
(`legal-chunk`, `legal-extract-facts`) vốn luôn nhận đường dẫn tường minh,
không có đường dẫn ngầm định.

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
- `fact_extraction.py` — dựng cửa sổ đọc theo ba nguồn ở mục "Gộp chunk khi
  một fact cần nhiều nguồn" (tối đa 2 chunk liền trước qua chuỗi
  `previous_chunk_id`, trích dẫn chéo bằng regex "Điều N" tra theo
  `build_provision_index`, tương đồng embedding cùng `source_name` qua
  `find_embedding_candidates`), gọi BAML trích fact, rồi gán `fact_id` và
  `fact_content_sha256` qua `sha256_json`. **Tin tưởng hoàn toàn output LLM**
  (xem "Nguyên tắc kiến trúc" ở đầu tài liệu): `evidence.chunk_id` và
  `evidence.quoted_span` được giữ nguyên như LLM trả về, không tra cứu lại
  chunk gốc để gán thêm `unit_ids`/`source_refs` (đã bỏ hai field này khỏi
  `EvidenceReference` vì không có nơi nào trong pipeline đọc lại chúng).
- `writer.py` — đọc/ghi JSONL nguyên tử cho `chunk_screening.jsonl` và
  `legal_facts.jsonl`, dùng lại `io_utils.write_jsonl_atomic`.

BAML trong `baml_src/`:

- `chunk_screening.baml` — class `ChunkScreeningItem` (chunk_id, text) và
  `ChunkScreeningResult` (chunk_id, has_penalty_content, reason), function
  `ScreenChunks(chunks: ChunkScreeningItem[]) -> ChunkScreeningResult[]` gọi
  theo batch (mặc định 8 chunk/lần, xem `DEFAULT_SCREENING_BATCH_SIZE` trong
  `chunk_screening.py`), kèm persona và few-shot examples. Kết quả đối chiếu
  ngược lại theo `chunk_id`, không theo thứ tự mảng trả về.
- `legal_fact_extraction.baml` — enum `ChunkWindowRole`, class
  `ChunkWindowEntry`, `EvidenceEntry`, `LegalFact` khớp JSON ở mục 2, function
  `ExtractLegalFacts(window: ChunkWindowEntry[]) -> LegalFact[]`, kèm persona,
  few-shot examples, và cảnh báo cho vai trò `CitedProvision` (số điều có thể
  trùng ngẫu nhiên với luật khác — regex chỉ đề xuất ứng viên, LLM tự quyết
  định có dùng hay không).

CLI tách thành hai bước riêng, nối nhau qua file — theo đúng cách các phase
sau nối nhau (`legal_facts.jsonl` → `evidence_relations.jsonl`/`scenarios.jsonl`
→ `single_turn.jsonl`), không gọi ngầm trong cùng 1 process nữa:

- `legal-screen-chunks` (`chunk_screening_cli.py`): nhận đường dẫn chunk JSONL
  của một văn bản, ghi `chunk_screening.jsonl` — 1 dòng verdict cho mỗi chunk,
  kể cả chunk bị loại (audit được, khác với bản trước đó không ghi gì).
- `legal-extract-facts` (`fact_extraction_cli.py`): nhận đường dẫn chunk JSONL
  **và** `chunk_screening.jsonl`, lọc lại chunk có `has_penalty_content=true`,
  trích fact, ghi thẳng `legal_facts.jsonl` — không còn ledger, không còn
  bước duyệt tay.

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
`test_chunk_screening.py`, `test_evaluation_models.py`,
`test_evaluation_writer.py`, `tests/test_baml_transport.py`,
`tests/test_chunk_screening_cli.py`, `tests/test_fact_extraction_cli.py`
(tích hợp CLI đầu-cuối với client và encoder giả mỗi bên). Windowing test
trực tiếp không cần mock; các BAML client wrapper test theo mẫu mock
`execute_baml_request` trong `tests/parser/test_baml_captioning.py`.

### Phase 2 — Evidence graph và scenario sampling (mục 6, 7) ✅ Đã triển khai

- `evidence_graph.py` — `find_candidate_fact_pairs` dùng `EmbeddingEncoder`
  tìm cặp fact ứng viên trong toàn bộ fact đã trích xuất (không giới hạn cùng
  văn bản, vì multi-hop cần đúng cặp khác văn bản); `confirm_evidence_relations`
  gọi BAML xác nhận từng cặp và lấy `bridge`; `classify_bundle_kind` áp ràng
  buộc cấu trúc — `amended_by` luôn là `temporal` bất kể văn bản, còn lại thì
  khác văn bản mới là `multi_hop`, cùng văn bản hạ xuống `synthesis` — biểu
  diễn trực tiếp bằng logic phân loại, không phải bước "hạ cấp" riêng sau khi
  đã tạo bundle; `sample_bundles` chặn trùng theo `τ`.
- `scenario_sampling.py` — nạp taxonomy/persona/phân bố từ config, thuật
  toán tham lam theo quota đã mô tả ở mục 5.

**Hai điểm lệch so với kế hoạch ban đầu, có lý do rõ:**

- `ConfirmEvidenceRelation(fact_a, fact_b) -> ConfirmedRelation` **không**
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
sách quan hệ ở mục 4), class `LegalFactSummary`, `ConfirmedRelation`, function
`ConfirmEvidenceRelation`. Giá trị `RelationType` phía BAML dùng PascalCase
theo đúng quy ước `ChunkWindowRole`; phía Python dùng `Literal` snake_case
khớp ví dụ JSON ở mục 4 (`"amended_by"`), dịch qua bảng ánh xạ tường minh
trong `evidence_graph.py` — không suy luận chuỗi.

Config: `data/config/groundtruth/query_taxonomy.json`,
`data/config/groundtruth/personas.json`,
`data/config/groundtruth/scenario_distribution.json`,
`data/config/groundtruth/evolution.json` (file config được tạo ở Phase 2 theo
đúng mục 6; logic *áp dụng* evolution vẫn thuộc Phase 3).

CLI: `legal-build-evidence-graph` (`evidence_graph_cli.py`) — đọc mọi fact từ
`legal_facts.jsonl`, xác nhận quan hệ, lấy mẫu bundle, sinh scenario, ghi
`evidence_relations.jsonl` và `scenarios.jsonl`. `scenarios.jsonl` là output
mới phát sinh khi triển khai, chưa có trong cấu trúc output đề xuất ban đầu
ở mục 17 — bổ sung vì Phase 3 cần scenario đã chốt và ổn định, không tính
lại mỗi lần sinh candidate.

Test: `test_evidence_graph.py`, `test_scenario_sampling.py` (thuật toán tham
lam test bằng fixture fact cố định, kiểm quota trần theo `ceil`, kiểm luân
phiên theo trục ít dùng nhất), `tests/test_evidence_graph_cli.py` (tích hợp
CLI đầu-cuối).

### Phase 3 — Sinh single-turn và hai cổng kiểm chứng (mục 8) ✅ Đã triển khai

- `candidate_generation.py` — `generate_candidate` gọi BAML sinh candidate từ
  một `Scenario` (Phase 2) + evidence đã chốt; mỗi claim chỉ tham chiếu
  `chunk_id`, còn `quoted_span` luôn được code tự lấy lại từ evidence thật
  của fact — không tin bản LLM tự chép lại, cùng kỷ luật với
  `fact_extraction.build_legal_fact` ở Phase 1; `answer_fields` chép trực
  tiếp từ `LegalFact`, không đi qua LLM. `refine_surface` gọi lại
  `CritiqueSurface` tối đa `max_quality_retries` lần, chỉ đổi `query_vi`,
  đóng băng claims/evidence/answer_fields. `evolve_candidate` áp các phép
  evolution đã cấu hình, giữ nguyên evidence, chỉ đổi `query_vi` và
  `reference_answer_vi`.
- `binding_gate.py` — Cổng 1: `check_binding` kiểm `chunk_id` thuộc evidence
  set thật của fact, và
  `rapidfuzz.fuzz.partial_ratio` giữa `quoted_span` và text chunk vượt
  `span_match_threshold`. Thuần hàm, không gọi LLM, **không raise cho input
  không hợp lệ** — trả về `BindingCheckResult` (passed, grounding_score,
  violations) vì candidate bị loại là kết quả bình thường của một cổng lọc,
  không phải lỗi chương trình.
- `verification.py` — Cổng 2: `verify_candidate` gọi BAML `VerifyCandidate`
  kiểm 6 chiều bằng model/prompt độc lập với generator, gán nhãn lại
  `observed_query_type`/`observed_answerability`/`necessary_reasoning_hops`.
  `passed` là phép AND tất định trên 6 giá trị bool do code tính, không tin
  thẳng một verdict tổng hợp từ LLM.
- `candidate_record.py` — module mới, ngoài kế hoạch ban đầu: `CandidateRecord`
  gộp `CandidateDraft` + `BindingCheckResult` + `VerificationResult` (có thể
  `None` nếu bị Cổng 1 loại) thành bản ghi cuối, với `review_status` suy ra
  tất định (`rejected_by_binding_gate` | `binding_verified` |
  `rejected_by_verification` | `automatically_verified`). Đặt ở module riêng
  để tránh vòng import giữa `candidate_generation.py`, `binding_gate.py` và
  `verification.py`.

**Các điểm lệch so với kế hoạch ban đầu, có lý do rõ:**

- CLI **đọc** `scenarios.jsonl` do Phase 2 sinh ra, không tự chạy lại
  scenario sampling — vì Phase 2 đã tách sampling thành output riêng, ổn
  định, không cần tính lại mỗi lần sinh candidate (đã ghi ở Phase 2).
- Không có "báo cáo loại bỏ theo từng cổng" dạng file riêng. Kết quả mỗi cổng
  gắn trực tiếp vào từng `CandidateRecord` (`binding_check`, `verification`,
  `review_status`), và CLI in tổng hợp số liệu qua log — đơn giản hơn một
  file report riêng mà vẫn audit được từng candidate.
- Generator và verifier dùng **cùng một model** cấu hình trong `Settings`,
  khác nhau ở BAML function/prompt, chưa hỗ trợ hai model riêng biệt như mục
  6 gợi ý ("giảm bias riêng của một model"). Prompt độc lập đã giảm được một
  phần self-confirmation bias; hỗ trợ multi-model là mở rộng để lại cho sau,
  cần thiết kế thêm cấu hình chọn model theo từng vai trò.
- `evolution.json` áp dụng **tất cả** operation đã cấu hình (tới `max_rounds`)
  cho **mọi** candidate qua được Cổng 1, không có lựa chọn candidate nào
  được evolve. Đây là lựa chọn tất định, không hardcode (danh sách operation
  đọc từ config, không viết cứng trong code), nhưng chưa có chính sách chọn
  lọc candidate nào cần evolve — để trống config `operations` là cách tắt
  evolution nếu không cần.

BAML: `candidate_generation.baml` — class `FactContext`, `ClaimReference`,
function `GenerateSingleTurnCandidate`, `CritiqueSurface`, enum
`EvolutionOperation`, function `EvolveCandidate`. `candidate_verification.baml`
— enum `Answerability`, class `VerificationResult`, function `VerifyCandidate`,
dùng chung `FactContext`/`ClaimReference` từ file kia (cùng namespace BAML).

CLI: `legal-generate-candidates` (`candidate_generation_cli.py`) — đọc
`legal_facts.jsonl`, `scenarios.jsonl`, và mọi `*.chunks.jsonl` trong một
thư mục; với mỗi scenario sinh `--candidates-per-scenario` candidate độc
lập, chạy nối tiếp sinh → tinh chỉnh bề mặt → evolution → Cổng 1 → Cổng 2
(chỉ chạy nếu Cổng 1 qua); ghi toàn bộ candidate (kể cả bị loại) vào một
file JSONL kèm `review_status`.

Test: `test_binding_gate.py` — đúng bảng ưu tiên cao nhất của Phase 3
(span khớp hoàn toàn, span dưới ngưỡng, `chunk_id` ngoài evidence set, fact
không tồn tại, chunk không tồn tại, claim rỗng, grounding_score trung bình
nhiều span). `test_candidate_generation.py`,
`test_verification.py` (gồm parametrize trên cả 6 chiều Cổng 2) theo mẫu mock
BAML đã có. `tests/test_candidate_generation_cli.py` tích hợp CLI đầu-cuối.

### Phase 4 — Pilot và ổn định schema (mục 9)

Không có code mới. Chạy Phase 1-3 trên riêng 168/2024/NĐ-CP, đo tỷ lệ loại ở
từng cổng, đưa một mẫu cho chuyên gia xem theo mục 13 (review theo rủi ro),
sau đó mới khóa schema `LegalFact`/`GroundTruthCandidate` trước khi mở rộng
ra toàn corpus.

### Phase 5 — Retriever (mục 10)

Ngoài phạm vi kế hoạch này. `src/legal_rag/retrieval/` đã bị xóa khỏi repo và
cần được dựng lại như một nhánh công việc riêng — Phase 6 không thể bắt đầu
tới khi phase này xong.

### Phase 6 — Hard negatives và difficulty (mục 11, chặn bởi Phase 5)

- `hard_negatives.py` — chạy retriever, loại trước các chunk đã thuộc
  evidence set của bất kỳ fact nào bằng tập hợp, gọi BAML xác nhận negative
  thật sự không đủ căn cứ.
- `leakage.py` — chạy candidate qua ba chế độ closed-book / oracle /
  oracle-trộn-nhiễu, ghi leakage score và difficulty vào candidate.

### Phase 7 — Multi-turn simulator (mục 12)

- `multiturn_simulator.py` — vòng lặp hai vai: goal state dựng từ
  `required_claims` của scenario, simulator tiết lộ dần, oracle chỉ đọc gold
  evidence, cập nhật `state_delta` sau mỗi lượt, dừng khi đạt goal hoặc chạm
  `max_simulator_turns`.

BAML mới: `multiturn_simulation.baml` — function `SimulateUserTurn(goal_state,
history) -> UserTurn`, function `GenerateOracleResponse(gold_evidence,
history) -> OracleTurn`.

Các lượt `underspecified`/`unanswerable` không sinh bằng simulator — quy
trình thủ công nằm ngoài phạm vi code, không cần module.

### Phase 8 — Split, coverage và đánh giá (mục 13)

- `dedup.py` — cụm embedding cosine `dedup_cosine_threshold`, gán
  `salience_weight`.
- `coverage.py` — báo cáo tỷ lệ fact có candidate, danh sách fact chưa phủ.
- `review_sampling.py` — chọn mẫu bắt buộc + mẫu ngẫu nhiên theo mục 13.
- `splitting.py` — chia `dev`/`test`/`hidden_test` theo thành phần liên thông
  của evidence graph chứa `fact_id`, không theo từng candidate.

Phần đánh giá retrieval/generator/end-to-end tái dùng `deepeval` (đã có trong
extra `evaluation`) cho các metric LLM-judge; `Accuracy theo field` ở mục 16
là so khớp tất định thuần Python, không qua `deepeval`.

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
- `pyproject.toml`: extra `evaluation` đổi thành `deepeval`, `rapidfuzz`,
  `sentence-transformers` — gỡ `ragas`, `langchain-community`,
  `langchain-openai` (không còn code nào import sau khi pipeline Ragas cũ bị
  xóa); thêm console script mới cho từng CLI ở các phase sau. ✅ **Đã áp
  dụng** ở Phase 0.
- `settings.py`: thêm `EvaluationSettings`. ✅ **Đã áp dụng.**
- `io_utils.py`: `write_jsonl_atomic` dùng chung; `parser/writer.py` và
  `chunker/writer.py` refactor gọi vào đây. ✅ **Đã áp dụng.**
- `embedding/encoder.py`: `EmbeddingEncoder` bọc `sentence-transformers`,
  nạp lazy để không bắt buộc cài đặt khi không dùng. ✅ **Đã áp dụng.**
- `data/evaluation/evaluation-data.md`: viết lại theo pipeline mới hoặc xóa.
  ✅ **Đã áp dụng** ở Phase 0.
- `src/legal_rag/evaluation/` và `data/evaluation/` đổi tên thành
  `src/legal_rag/benchmark_synthesis/` và `data/benchmark_synthesis/`
  (`evaluation-data.md` → `benchmark-synthesis-data.md`), `evaluation_cli.py`
  đổi tên thành `fact_extraction_cli.py`, để tên khớp mục đích thật (sinh
  benchmark) thay vì đọc như đo lường RAG. ✅ **Đã áp dụng.**
- `baml_transport.py`: tách khỏi `parser/image_captioner.py` khi cần dùng lại
  cho `chunk_screening.py`/`fact_extraction.py`. ✅ **Đã áp dụng** ở Phase 1,
  không nằm trong kế hoạch gốc.
- `chunk_screening.py` (trước đó `chunk_quality.py`): đổi từ chấm 4 tiêu chí
  liên tục + so ngưỡng sang hỏi nhị phân `has_penalty_content` — LLM đáng tin
  hơn với quyết định nhị phân so với tự hiệu chỉnh điểm số liên tục. Bỏ
  `chunk_quality_threshold` khỏi `EvaluationSettings`/`.env.example`.
  ✅ **Đã áp dụng.**
- `chunker/writer.py`: thêm `read_chunks` — CLI trích xuất fact cần đọc lại
  `*.chunks.jsonl`. ✅ **Đã áp dụng** ở Phase 1, không nằm trong kế hoạch gốc.
- `pyproject.toml`: thêm console script `legal-extract-facts`,
  `legal-review-facts` (Phase 1), `legal-build-evidence-graph` (Phase 2).
  ✅ **Đã áp dụng.**
- `data/config/groundtruth/`: bốn file config versioned
  (`query_taxonomy.json`, `personas.json`, `scenario_distribution.json`,
  `evolution.json`). ✅ **Đã áp dụng** ở Phase 2.
- `settings.py`: thêm `candidate_generation_max_output_tokens`,
  `candidate_generation_temperature`, `verification_max_output_tokens`,
  `verification_temperature`. ✅ **Đã áp dụng** ở Phase 3.
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
  `pyproject.toml`: thêm console script `legal-screen-chunks`. ✅ **Đã áp
  dụng**, không nằm trong kế hoạch gốc.
