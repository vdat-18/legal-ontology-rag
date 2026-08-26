enum ChunkWindowRole {
  Preceding
  PrecedingTarget
  Target
  Following
  FollowingTarget
  ArticleContext
}

class ChunkWindowEntry {
  chunk_id string
  role ChunkWindowRole @description(#"
    Target là chunk cần trích fact. PrecedingTarget và FollowingTarget
    là context cũng được xử lý độc lập như target ở request khác.
  "#)
  context_text string @description(#"
    Ngữ cảnh cấu trúc do parser tạo; không phải source text.
  "#)
  text string
}

class FactExtractionBatchItem {
  target_chunk_id string
  window ChunkWindowEntry[]
}

class EvidenceEntry {
  chunk_id string @description(#"Phải trùng với một chunk_id trong window."#)
}

class LegalFact {
  statement string @description(#"
    Một mệnh đề pháp lý nguyên tử, tự hiểu được và đủ căn cứ.
  "#)
  evidence EvidenceEntry[] @description(#"Ít nhất một chunk trong window, phủ toàn bộ statement."#)
}

class FactExtractionBatchResult {
  target_chunk_id string @description(#"Phải trùng với một target_chunk_id trong danh sách đầu vào."#)
  facts LegalFact[]
}

function ExtractLegalFactBatch(items: FactExtractionBatchItem[]) -> FactExtractionBatchResult[] {
  client OpenAICompatibleProvider
  prompt #"
    {{ _.role("system") }}

    Bối cảnh: chuẩn bị nguồn sự thật cho benchmark RAG hỏi đáp về
    xử lý vi phạm giao thông đường bộ tại Việt Nam.

    Nhiệm vụ: trích legal fact độc lập từ mỗi reading window. Domain
    gồm quy định trực tiếp về hành vi/nghĩa vụ; chủ thể, phương tiện,
    điều kiện/ngoại lệ; chế tài, trừ điểm/tước quyền, khắc phục hậu
    quả; thẩm quyền, trình tự/thời hạn/thủ tục; hiệu lực, sửa đổi hoặc
    chuyển tiếp ảnh hưởng trực tiếp đến việc xử lý vi phạm.

    Reading window:
    - Target là chunk cần trích và phải chứa một phần source text trực
      tiếp cho mỗi fact.
    - Mỗi fact thuộc về Target chứa nội dung pháp lý đặc trưng của quy
      tắc như hành vi, chủ thể, điều kiện, ngoại lệ hoặc hậu quả; tiêu
      đề, câu dẫn, đánh số hay dẫn chiếu chung không đủ để sở hữu fact.
    - Trích mọi nội dung pháp lý thuộc domain có phần đặc trưng nằm
      trong Target, kể cả Target bắt đầu giữa một khoản hoặc danh sách
      từ PrecedingTarget. Mỗi mục, điều kiện, ngoại lệ hoặc hậu quả mới
      xuất hiện trong Target thuộc Target; không giả định request khác
      đã trích nó.
    - Khi Target bắt đầu bằng phần tiếp nối, dùng Preceding hoặc
      PrecedingTarget để khôi phục phần dẫn chung, chủ thể và phạm vi,
      rồi viết fact đầy đủ cho nội dung nằm trong Target.
    - Với FollowingTarget, chỉ dùng phần đầu để hoàn chỉnh đúng câu
      hoặc mục đang viết dở ở cuối Target. Nếu FollowingTarget bắt đầu
      mục, điều kiện, ngoại lệ hay quy tắc mới thì tuyệt đối không lấy;
      nội dung đó thuộc target kế tiếp.
    - Nếu Target kết thúc bằng câu dẫn danh sách mà các mục bắt đầu ở
      FollowingTarget, không tạo fact cho danh sách đó. Nếu Target đã
      chứa một số mục và FollowingTarget chứa các mục còn lại, chỉ trích
      các mục nằm trong Target; không gộp toàn bộ danh sách.
    - Following thường không được xử lý độc lập, nên có thể dùng để
      hoàn chỉnh quy tắc hoặc danh sách đã bắt đầu trong Target khi cần
      thiết và vẫn phải có Target trong evidence.
    - Không trích lại cùng một quy tắc dưới dạng gộp và dạng tách.
    - Structural context chỉ giúp hiểu cấu trúc, không phải source text.

    Output contract:
    - statement là nguồn sự thật duy nhất: một mệnh đề pháp lý
      nguyên tử, tự hiểu được và giữ đủ thành phần cần thiết để áp dụng.
    - statement chỉ viết bằng tiếng Việt. Không chèn từ hoặc ký tự của
      ngôn ngữ khác; chỉ giữ chữ cái Latin, số, đơn vị, mã hạng và ký
      hiệu cần thiết có căn cứ từ source text.
    - Giữ các thành phần của cùng một quy tắc trong một fact. Chỉ
      tách khi các quy tắc độc lập hoặc khác phạm vi, điều kiện hay
      hậu quả; không gộp quy tắc không liên quan.
    - Một câu mới nêu chủ thể hoặc nghĩa vụ mới là quy tắc độc lập và
      phải thành fact riêng, kể cả khi nằm trong cùng một khoản. Không
      nhập nghĩa vụ đó vào fact của câu đứng trước.
    - Không suy ra quan hệ tương ứng chỉ từ vị trí của các mục trong
      nhiều danh sách song song. Chỉ tách thành từng cặp khi Source text
      xác lập rõ ánh xạ; nếu không, giữ cấu trúc quy tắc chung và không
      tự thêm giới hạn cho từng chủ thể.
    - Giữ chính xác ý nghĩa, con số, khoảng, đơn vị, điều kiện và
      ngoại lệ. Không thêm kiến thức, suy đoán hoặc diễn giải rộng hơn
      source text; không tạo fact nếu chưa đủ căn cứ.
    - Giữ nguyên mọi giới hạn về chủ thể, loại phương tiện, hoàn cảnh
      áp dụng và dẫn chiếu điều, khoản; không thay dẫn chiếu cụ thể bằng
      cụm chung như "theo quy định", không mở rộng nhãn nhóm so với nguồn.
    - Không bỏ sót quy định giao thẩm quyền hoặc quy định chi tiết nằm
      trọn trong Target khi nó nêu rõ chủ thể và nội dung được giao.
    - evidence gồm Target và các chunk trong window có Source text đã
      dùng; toàn bộ statement phải được phủ. Không dùng Structural
      context làm evidence.
    - Trả đúng một kết quả cho mỗi input, giữ nguyên target_chunk_id;
      facts=[] là hợp lệ.

    Tự kiểm tra: đã xét hết nội dung pháp lý đặc trưng nằm trong Target,
    kể cả phần tiếp nối; không bỏ sót vì nội dung bắt đầu ở chunk trước;
    nếu evidence có FollowingTarget thì phần được dùng chỉ là đuôi của
    đúng câu hoặc mục bắt đầu trong Target, không phải mục mới; statement
    không tự tạo ánh xạ tương ứng chỉ từ thứ tự liệt kê; không gộp câu
    mới có chủ thể hoặc nghĩa vụ mới vào fact trước; statement thuần
    tiếng Việt, tự đủ nghĩa và evidence phủ toàn bộ statement.

    {{ _.role("user") }}

    Các reading window:

    {% for item in items %}
    === target_chunk_id={{ item.target_chunk_id }} ===
    {% for entry in item.window %}
    [{{ entry.role }}] chunk_id={{ entry.chunk_id }}
    {% if entry.context_text %}
    Structural context (không dùng làm evidence):
    {{ entry.context_text }}
    {% endif %}
    Source text:
    {{ entry.text }}
    ---
    {% endfor %}
    === end target_chunk_id={{ item.target_chunk_id }} ===
    {% endfor %}

    {{ ctx.output_format }}
  "#
}
