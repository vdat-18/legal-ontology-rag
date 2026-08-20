enum ChunkWindowRole {
  Preceding
  Target
  Following
  ArticleContext
}

class ChunkWindowEntry {
  chunk_id string
  role ChunkWindowRole @description(#"
    Target là chunk chính đang được đọc để trích fact. Preceding và Following
    là chunk liền trước và liền sau theo thứ tự văn bản. ArticleContext là
    chunk bổ sung trong cùng Điều để hoàn chỉnh phạm vi và dẫn chiếu pháp lý.
  "#)
  context_text string @description(#"
    Ngữ cảnh cấu trúc do parser lấy từ metadata tài liệu. Có thể dùng để hiểu
    Chương, Mục, Điều và phạm vi áp dụng, nhưng không phải source text để làm
    căn cứ cho fact.
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
    Một mệnh đề pháp lý nguyên tử, đầy đủ và tự hiểu được, bao quát toàn bộ
    chủ thể, hành vi hoặc nghĩa vụ, điều kiện và hậu quả pháp lý có căn cứ.
  "#)
  evidence EvidenceEntry[] @description(#"Ít nhất một phần tử. Mọi nội dung của fact phải có căn cứ ở đây."#)
}

class FactExtractionBatchResult {
  target_chunk_id string @description(#"Phải trùng với một target_chunk_id trong danh sách đầu vào."#)
  facts LegalFact[]
}

function ExtractLegalFactBatch(items: FactExtractionBatchItem[]) -> FactExtractionBatchResult[] {
  client OpenAICompatibleProvider
  prompt #"
    {{ _.role("user") }}

    Bạn trích xuất các fact pháp lý nguyên tử về xử lý vi phạm giao thông đường bộ
    từ từng reading window. Target là chunk cần trích xuất; Preceding và
    Following chỉ giúp hoàn chỉnh quy định bị chia chunk.

    Quy tắc bắt buộc:
    - Chỉ tạo fact cho quy định có căn cứ trực tiếp trong Target. Context có
      thể hoàn chỉnh statement, subject, conditions, hậu quả và provision_path,
      nhưng không được tạo fact chỉ xuất hiện ngoài Target.
    - statement là nguồn sự thật duy nhất: một mệnh đề pháp lý nguyên tử, đầy
      đủ và tự hiểu được. Ghi trực tiếp mọi chủ thể, hành vi hoặc nghĩa vụ,
      điều kiện, mức hoặc hình thức xử phạt, biện pháp, ngoại lệ, thẩm quyền,
      thủ tục và hiệu lực cần thiết cho nội dung đó.
    - Tách riêng các quy tắc, hành vi hoặc đối tượng có điều kiện hay hậu quả
      khác nhau. Không dùng kiến thức bên ngoài, không suy đoán. Không đủ căn
      cứ cho một statement hoàn chỉnh thì bỏ fact.
    - Mọi nội dung của fact phải được phủ bởi evidence từ đúng chunk_id trong
      window. Chỉ tham chiếu chunk thực sự làm căn cứ cho fact.
    - Structural context chỉ dùng để hiểu cấu trúc và provision_path, không
      được dùng làm evidence. Với nhiều VARIANT, đối chiếu Source text
      và thứ tự window để không trộn quy định qua ranh giới Điều.
    - Xử lý từng window độc lập. Trả đúng một kết quả cho mỗi target_chunk_id,
      giữ nguyên id; facts=[] là hợp lệ.

    Tự kiểm tra trước khi trả lời: statement có căn cứ trong Target, tự đủ
    nghĩa, và các chunk evidence đủ cho toàn bộ nội dung của statement.

    Ví dụ ngắn:
    - Target nêu "không chấp hành hiệu lệnh của đèn tín hiệu" và window nêu
      mức phạt tương ứng -> statement bao quát đầy đủ chủ thể, hành vi, mức
      phạt và hậu quả liên quan.
    - Target quy định thẩm quyền xử phạt -> statement giữ trọn quy định.
    - Target chỉ giải thích thuật ngữ, không nêu quy tắc xử phạt -> facts=[].

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
