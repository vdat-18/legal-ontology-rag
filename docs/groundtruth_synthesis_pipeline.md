# Ground-Truth Synthesis Pipeline

## Mục tiêu và phạm vi

Pipeline này sinh dữ liệu ground truth để đánh giá hệ thống RAG trong phạm vi
**xử phạt vi phạm giao thông đường bộ tại Việt Nam**. Benchmark hỗ trợ cả câu
hỏi single-turn và hội thoại multi-turn, đồng thời đánh giá riêng retrieval,
generation và toàn bộ luồng RAG.

Dữ liệu do LLM sinh chỉ là candidate. Chỉ những mẫu đã vượt qua kiểm tra tự
động và được ràng buộc vào một fact pháp lý đã được chuyên gia phê duyệt mới
được đưa vào ground truth.

## Nguyên tắc kiến trúc: fact ledger

Tài nguyên khan hiếm nhất của dự án là thời gian của người có chuyên môn pháp
lý, không phải LLM call. Vì vậy điểm phê duyệt của con người đặt tại **fact
pháp lý**, không đặt tại từng câu hỏi.

- Số fact pháp lý phân biệt trong corpus là hữu hạn và nhỏ. Số câu hỏi có thể
  sinh từ chúng là rất lớn.
- Duyệt một fact là khoản đầu tư một lần, tái sử dụng cho mọi câu hỏi phái
  sinh. Chi phí review chuyển từ `O(số câu hỏi)` sang `O(số fact)`.
- Câu hỏi phái sinh an toàn vì bị **ràng buộc cơ học** vào fact đã duyệt, chứ
  không vì có người đọc lại từng câu.

Ba cổng chất lượng, theo thứ tự từ rẻ đến đắt:

```text
Cổng 1  Ràng buộc căn cứ tất định   không dùng LLM, không dùng người
Cổng 2  Kiểm chứng LLM độc lập      dùng LLM khác generator
Cổng 3  Phê duyệt fact ledger       dùng chuyên gia, một lần cho mỗi fact
```

Một candidate chỉ hợp lệ khi mọi căn cứ của nó nằm trong evidence set của một
fact đã ở trạng thái `approved`. Không có đường nào khác đưa dữ liệu vào ground
truth.

```text
.chunks.jsonl
    -> Legal facts (typed output, kèm quoted span)
    -> Fact ledger phê duyệt bởi chuyên gia         [cổng 3, một lần]
    -> Evidence graph trên các fact đã duyệt
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
- `data/evaluation/groundtruth-corpus-v1-snapshot.json`
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
      "quoted_span": "...",
      "unit_ids": ["..."],
      "source_refs": ["..."]
    }
  ]
}
```

Nguyên tắc:

- Một fact chỉ chứa nội dung có bằng chứng trực tiếp.
- Không sử dụng kiến thức sẵn có của model làm căn cứ.
- Nếu một fact cần nhiều chunk thì phải lưu tất cả chunk cần thiết.
- Mỗi `evidence` phải kèm `quoted_span` trích nguyên văn từ `text` của chunk.
- `chunk_id` phục vụ retrieval evaluation.
- `unit_ids` và `source_refs` phục vụ audit ngược về nguồn parser.

### Lọc chất lượng chunk trước khi trích xuất

Trước khi trích xuất fact, một critic model rẻ chấm mỗi chunk theo bốn tiêu
chí: rõ ràng, đủ sâu, có cấu trúc, đúng chủ đề. Chunk dưới ngưỡng bị bỏ qua
và ghi vào danh sách loại trừ kèm lý do — thường là mục lục, tiêu đề chương,
đoạn dẫn nhập không mang chế tài.

Cổng rẻ này đứng trước để bảo vệ hai công đoạn đắt phía sau: LLM call trích
xuất và thời gian chuyên gia duyệt fact. Ngưỡng nằm trong cấu hình có phiên
bản, và danh sách chunk bị loại phải audit được.

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
bộ** nội dung fact.

Cặp này lặp lại đúng kỷ luật mà chunker đang dùng cho `chunk_id` và
`content_sha256`, và cho hai tính chất bắt buộc:

- Trích xuất lại corpus không làm mất trạng thái phê duyệt, vì `fact_id` không
  đổi khi cách diễn đạt thay đổi.
- Khi `fact_content_sha256` đổi so với lúc duyệt, fact tự động chuyển về trạng
  thái cần duyệt lại, và mọi candidate phái sinh bị đánh dấu `stale`.

Output dự kiến: `data/evaluation/legal_facts.jsonl`.

## 3. Phê duyệt fact ledger

Đây là cổng chất lượng duy nhất cần chuyên môn pháp lý và là công đoạn đắt
nhất của toàn pipeline. Mọi thiết kế phía sau tồn tại để không phải lặp lại nó.

Luồng trạng thái của một fact:

```text
extracted
    -> span_verified
    -> needs_expert_review
    -> approved | rejected
    -> stale        (khi fact_content_sha256 đổi)
```

Người duyệt xác nhận trên từng fact:

- Nội dung chế tài đúng và đầy đủ so với điều khoản nguồn
- `provision_path` trỏ đúng điều, khoản, điểm
- `valid_from` và `valid_to` đúng, kể cả khi văn bản đã bị sửa đổi
- Không nhầm hành vi, đối tượng hoặc loại phương tiện
- `quoted_span` thực sự là căn cứ của nội dung, không phải đoạn lân cận

Duyệt theo cụm để tận dụng ngữ cảnh: gom các fact cùng một điều hoặc cùng một
hành vi vào một phiên duyệt thay vì duyệt theo thứ tự ngẫu nhiên.

Fact ở trạng thái `approved` là nguồn chân lý duy nhất của benchmark. Không
sinh candidate từ fact chưa duyệt.

## 4. Xây evidence graph

Evidence graph chỉ liên kết các fact đã ở trạng thái `approved`.

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
2. Fact `fact_id` phải ở trạng thái `approved` và không `stale`.
3. `quoted_span` phải khớp `text` của chunk theo Levenshtein partial ratio với
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
`approved` nào xuất hiện trong scenario, để verifier không phải xử lý các
trường hợp hiển nhiên.

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

## 12. Kiểm soát độ phủ trên fact ledger

Độ phủ đo trên **bảng fact**, không đo trên tập câu hỏi. Chỉ có cách này mới
trả lời được câu hỏi quan trọng nhất là còn fact nào chưa được kiểm tra.

Chỉ số chính:

- Tỷ lệ fact `approved` có ít nhất một candidate
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
data/evaluation/
├── legal_facts.jsonl
├── fact_ledger.jsonl
├── evidence_relations.jsonl
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

`fact_ledger.jsonl` lưu trạng thái phê duyệt, người duyệt và thời điểm duyệt,
tách khỏi `legal_facts.jsonl` để trích xuất lại không ghi đè lịch sử phê duyệt.

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
   `fact_id` và `fact_content_sha256` tách bạch. **Đã xong** cho
   `LegalFact`/`EvidenceReference`; `Scenario`, `GroundTruthCandidate`,
   `Conversation`, `ConversationTurn` còn lại cho Phase 2-3.
3. Implement lọc chất lượng chunk và legal-fact extraction kèm `quoted_span`.
   **Đã xong.**
4. Implement ràng buộc căn cứ tất định. Chạy nó ngay trên output trích xuất để
   loại các fact có span không khớp trước khi tốn thời gian chuyên gia.
   **Đã xong**, ở mức fact — `fact_ledger.verify_spans`. Cổng 1 ở mức
   candidate (mục 7) vẫn thuộc Phase 3.
5. Xây fact ledger và quy trình phê duyệt. Chạy pilot trên một văn bản duy
   nhất, ưu tiên 168/2024/NĐ-CP, cho tới khi schema fact ổn định. **Máy trạng
   thái đã xong**; pilot thật trên 168/2024/NĐ-CP còn chờ vì cần LLM call
   thật và người duyệt.
6. Tạo file cấu hình versioned cho query taxonomy, persona, evolution và phân
   bố scenario.
7. Implement evidence graph có lưu `bridge` và ràng buộc cấu trúc bundle.
8. Implement single-turn generation kèm vòng tinh chỉnh bề mặt, cổng 1, cổng
   2 và gán nhãn lại.
9. Chạy pilot, đo tỷ lệ lỗi trên mẫu review, ổn định schema candidate.
10. Implement retriever. Đây là điều kiện chặn cho hai mục kế tiếp.
11. Implement hard-negative mining, difficulty scoring và leakage analysis.
12. Implement multi-turn bằng user simulator hướng mục tiêu với oracle bám
    gold evidence; người viết hoặc sửa các lượt `underspecified` và
    `unanswerable`.
13. Xuất các split theo `fact_id` và implement đánh giá retrieval,
    oracle-generator, oracle trộn nhiễu và end-to-end.

Không bắt đầu sinh dữ liệu hàng loạt trước khi schema fact, temporal metadata
và quy trình phê duyệt fact ledger đã ổn định. Sinh candidate hàng loạt trên
một fact ledger chưa ổn định sẽ tạo ra khối lượng lớn dữ liệu `stale` ngay khi
fact được sửa.

## Kế hoạch triển khai kỹ thuật

Bản dịch 13 bước ở trên sang module, file BAML, config và test cụ thể, theo
quy ước đã có của repo (`chunker/`, `parser/`, `embedding/`). Tuân thủ
[AGENTS.md](../AGENTS.md): thay đổi nhỏ và tập trung theo từng phase, type
hint ở API công khai, `pathlib`, không dùng `print()`, không thêm dependency
trừ khi thư viện hiện có không đủ.

### Quyết định vị trí: tái dùng `evaluation/`, không tạo package mới

`src/legal_rag/evaluation/` đã tồn tại nhưng rỗng — đây từng là nơi chứa
pipeline Ragas cũ (`groundtruth.py`, `schemas.py`, `snapshot.py`,
`synthesis.py`) trước khi bị gỡ ở commit `38475a6`. Toàn bộ code mới của kế
hoạch này đặt lại vào `src/legal_rag/evaluation/`, chia theo module một-file-
một-trách-nhiệm thay vì gộp bốn file lớn như thiết kế cũ. Không tạo package
song song `groundtruth/`.

`data/evaluation/evaluation-data.md` hiện trỏ tới API `generate_groundtruth_candidates(...)`
và biến môi trường `RAGAS_*` đã không còn tồn tại. Cập nhật hoặc xóa file này
ở Phase 0 để không ai đi theo hướng dẫn sai.

### Phase 0 — Hạ tầng dùng chung

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
hệt. Thêm một module thứ ba (`evaluation/writer.py`) sẽ là lần lặp thứ ba —
đủ điều kiện theo AGENTS.md để tách hàm dùng chung. Thêm
`src/legal_rag/io_utils.py` chứa `write_jsonl_atomic(path, records)`, refactor
hai module cũ gọi vào đây, rồi `evaluation/writer.py` dùng lại. Đây là thay
đổi cross-cutting nhỏ, nên làm thành một commit riêng trước khi viết code
Phase 1.

**Cấu hình.** Thêm `EvaluationSettings` vào `settings.py`, theo đúng mẫu
`ChunkerSettings` — nạp từ biến môi trường: `chunk_quality_threshold`,
`span_match_threshold` (mặc định `0.85`), `bundle_overlap_threshold` (`τ`),
`dedup_cosine_threshold` (`0.9`), `max_quality_retries`, `max_simulator_turns`.
Danh sách persona, query taxonomy, phân bố scenario và evolution nằm ở file
JSON versioned dưới `data/config/groundtruth/`, đọc bằng cùng cơ chế
`importlib.resources` như `embedding/profiles.py`.

### Phase 1 — Trích xuất fact và fact ledger (mục 2, 3 ở trên) — Đã triển khai

Module trong `evaluation/`:

- `models.py` — `LegalFact`, `EvidenceReference` dạng
  `@dataclass(frozen=True, slots=True)` với `to_dict()`, hằng
  `LEGAL_FACT_SCHEMA_VERSION`, theo đúng khuôn `chunker/models.py`.
- `chunk_quality.py` — gọi BAML chấm 4 tiêu chí, lọc theo
  `chunk_quality_threshold`, ghi danh sách chunk bị loại kèm lý do.
- `fact_extraction.py` — dựng cửa sổ đọc theo ba nguồn ở mục "Gộp chunk khi
  một fact cần nhiều nguồn" (liền kề qua `previous_chunk_id`/`next_chunk_id`,
  trích dẫn chéo bằng regex "Điều N" tra theo `build_provision_index`, tương
  đồng embedding cùng `source_name` qua `find_embedding_candidates`), gọi BAML
  trích fact, đối chiếu `evidence.chunk_id` với chunk thật để tự điền
  `unit_ids`/`source_refs` — không tin trực tiếp giá trị LLM trả về — rồi gán
  `fact_id` và `fact_content_sha256` qua `sha256_json`.
- `fact_ledger.py` — máy trạng thái `extracted → span_verified →
  needs_expert_review → approved | rejected`, cộng thêm `stale` khi
  `fact_content_sha256` đổi so với bản đã duyệt. `verify_spans` dùng
  `rapidfuzz.fuzz.partial_ratio` giữa `quoted_span` và text chunk thật —
  cùng cơ chế văn bản với Cổng 1 ở mục 7, áp dụng sớm ngay tại fact thay vì
  đợi tới candidate.
- `writer.py` — đọc/ghi JSONL nguyên tử cho `legal_facts.jsonl` và
  `fact_ledger.jsonl`, dùng lại `io_utils.write_jsonl_atomic`.

BAML trong `baml_src/`:

- `chunk_quality.baml` — class `ChunkQualityResult` (clarity, depth,
  structure, relevance), function `EvaluateChunkQuality(chunk_text: string)`.
- `legal_fact_extraction.baml` — enum `ChunkWindowRole`, class
  `ChunkWindowEntry`, `EvidenceEntry`, `LegalFact` khớp JSON ở mục 2, function
  `ExtractLegalFacts(window: ChunkWindowEntry[]) -> LegalFact[]`.

CLI: `legal-extract-facts` (`evaluation_cli.py`) chạy nối tiếp cổng chất
lượng chunk, trích xuất, đối chiếu ledger, `verify_spans`, và
`queue_for_expert_review` cho mọi fact mới trong một lần gọi; nhận đường dẫn
chunk JSONL của một văn bản, ghi `legal_facts.jsonl` và `fact_ledger.jsonl`.

`legal-review-facts` (`fact_review_cli.py`) là script duyệt tương tác dùng
`print`/`input` trực tiếp — không qua `loguru`, vì đây là giao diện người
dùng thật của công cụ, không phải log chẩn đoán; có ngoại lệ ruff T20 riêng
cho file này trong `pyproject.toml`.

Hạ tầng dùng chung mới phát sinh khi triển khai, ngoài phạm vi ban đầu của
Phase 1 nhưng cần thiết để nó chạy được:

- `baml_transport.py` — tách phần gọi HTTP non-streaming và dựng
  `ClientRegistry` ra khỏi `parser/image_captioner.py`, vì đây đã là lần
  dùng thứ ba (chunk quality, fact extraction, image caption) của cùng một
  khối logic.
- `chunker/writer.py` bổ sung `read_chunks` — trước đó chunker chỉ ghi output
  cuối, chưa có gì đọc lại `*.chunks.jsonl`, nhưng CLI trích xuất fact cần
  đọc đúng file đó làm đầu vào.

Test: `tests/evaluation/test_fact_extraction.py`, `test_fact_ledger.py`,
`test_chunk_quality.py`, `test_evaluation_models.py`,
`test_evaluation_writer.py`, `tests/test_baml_transport.py`,
`tests/test_evaluation_cli.py` (tích hợp CLI đầu-cuối với client và encoder
giả). Windowing và máy trạng thái test trực tiếp không cần mock; các BAML
client wrapper test theo mẫu mock `execute_baml_request` trong
`tests/parser/test_baml_captioning.py`.

### Phase 2 — Evidence graph và scenario sampling (mục 6, 7)

- `evidence_graph.py` — dùng `embedding/encoder.py` tìm cặp fact ứng viên
  (cùng và khác văn bản), gọi BAML xác nhận quan hệ và lấy `bridge`, áp ràng
  buộc cấu trúc bundle (khác văn bản cho multi-hop, chặn trùng `τ`).
- `scenario_sampling.py` — nạp taxonomy/persona/phân bố từ config versioned,
  thuật toán tham lam theo quota đã mô tả ở mục 5.

BAML mới: `evidence_relations.baml` — class `EvidenceRelation` khớp JSON ở
mục 4, function `ConfirmEvidenceRelation(fact_a: LegalFact, fact_b: LegalFact,
candidate_relation_type: string) -> EvidenceRelation | null`.

Config mới: `data/config/groundtruth/query_taxonomy.json`,
`data/config/groundtruth/personas.json`,
`data/config/groundtruth/scenario_distribution.json`,
`data/config/groundtruth/evolution.json`.

Test: `test_evidence_graph.py` (ràng buộc cấu trúc bundle là logic thuần,
test không cần mock LLM), `test_scenario_sampling.py` (thuật toán tham lam
test bằng fixture fact cố định, kiểm phân bố đầu ra).

### Phase 3 — Sinh single-turn và hai cổng kiểm chứng (mục 8)

- `candidate_generation.py` — gọi BAML sinh candidate từ scenario + evidence
  đã chốt; `answer_fields` chép trực tiếp từ `LegalFact`, không đi qua LLM;
  vòng tinh chỉnh bề mặt gọi lại BAML critique tối đa
  `max_quality_retries` lần; evolution áp các phép giữ nguyên evidence.
- `binding_gate.py` — Cổng 1: kiểm `chunk_id` thuộc evidence set, fact
  `approved` và không `stale`, `rapidfuzz.fuzz.partial_ratio` giữa
  `quoted_span` và text chunk vượt `span_match_threshold`. Thuần hàm, không
  gọi LLM, input là output BAML nên phải coi là **không đáng tin** theo đúng
  nguyên tắc "Treat model and tool output as untrusted" của AGENTS.md.
- `verification.py` — Cổng 2: gọi BAML kiểm 6 chiều bằng model/prompt khác
  generator, gán nhãn lại `query_type`/`answerability`/`reasoning_hops`.

BAML mới: `candidate_generation.baml` (function
`GenerateSingleTurnCandidate`, `CritiqueSurface`, `EvolveCandidate`),
`candidate_verification.baml` (function `VerifyCandidate`, output khớp 6
chiều ở mục 10 kèm `hop_necessity`).

CLI: `legal-generate-candidates` — chạy scenario sampling → sinh → cổng 1 →
cổng 2 nối tiếp trong một tiến trình batch, ghi
`data/evaluation/candidates/single_turn.jsonl` và một báo cáo loại bỏ theo
từng cổng.

Test: `test_binding_gate.py` là ưu tiên cao nhất trong toàn bộ Phase 3 — logic
tất định, không cần mock, phải test đầy đủ theo bảng: span khớp hoàn toàn,
span dưới ngưỡng, `chunk_id` ngoài evidence set, fact chưa `approved`, fact
`stale`. `test_candidate_generation.py`, `test_verification.py` theo mẫu mock
BAML đã có.

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
  evidence set của fact `approved` bằng tập hợp, gọi BAML xác nhận negative
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

### Tổng hợp thay đổi ngoài `evaluation/`

- `pyproject.toml`: extra `evaluation` đổi thành `deepeval`, `rapidfuzz`,
  `sentence-transformers` — gỡ `ragas`, `langchain-community`,
  `langchain-openai` (không còn code nào import sau khi pipeline Ragas cũ bị
  xóa); thêm console script mới cho từng CLI ở các phase sau. **Đã áp dụng**
  ở Phase 0.
- `settings.py`: thêm `EvaluationSettings`. **Đã áp dụng.**
- `io_utils.py`: `write_jsonl_atomic` dùng chung; `parser/writer.py` và
  `chunker/writer.py` refactor gọi vào đây. **Đã áp dụng.**
- `embedding/encoder.py`: `EmbeddingEncoder` bọc `sentence-transformers`,
  nạp lazy để không bắt buộc cài đặt khi không dùng. **Đã áp dụng.**
- `data/evaluation/evaluation-data.md`: viết lại theo pipeline mới hoặc xóa.
