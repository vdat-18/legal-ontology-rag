class ExceptionCitationDiscoveryEntry {
  source_evidence_id string @description(#"Evidence ID thuộc general record và chứa dẫn chiếu."#)
  citation_text string @description(#"Cụm dẫn chiếu được chép nguyên văn, không kèm câu hành vi hoặc xử phạt."#)
  target_document_number string @description(#"Số văn bản đích đã chuẩn hóa; dùng văn bản nguồn nếu dẫn chiếu nội bộ."#)
  target_provision_paths string[] @description(#"
    Các locator đích tuyệt đối theo thứ tự `Điều > Khoản > Điểm`, không chứa Chương/Mục.
  "#)
}

class LegalExceptionDiscoveryEntry {
  general_population_id string @description(#"Một rule ID trong general_records."#)
  evidence_source_ids string[] @description(#"Các evidence ID của general record trực tiếp nêu ngoại lệ."#)
  citations ExceptionCitationDiscoveryEntry[] @description(#"Các dẫn chiếu tạo nên phạm vi ngoại lệ."#)
  rationale_vi string @description(#"Lý do ngắn gọn cho quan hệ ngoại lệ."#)
}

class ExceptionSourceContext {
  chunk_id string
  text string @description(#"Ngữ cảnh nguồn để hiểu cấu trúc sửa đổi; không phải evidence có thể chọn."#)
}

class ExceptionDiscoveryWindowInput {
  candidate_id string
  source_document_number string
  comparison_document_numbers string[]
  corpus_document_numbers string[]
  source_contexts ExceptionSourceContext[]
  general_records OntologyPopulationSummary[]
}

class ExceptionDiscoveryWindowResult {
  candidate_id string
  discoveries LegalExceptionDiscoveryEntry[]
}

class ExceptionTargetCandidateInput {
  proposal_index int
  citation_index int
  general_record OntologyPopulationSummary
  exception_provision_path string
  citation_text string
  target_document_number string
  target_provision_paths string[]
  target_records OntologyPopulationSummary[]
}

class ExceptionTargetResolutionWindowInput {
  candidate_id string
  candidates ExceptionTargetCandidateInput[]
}

class ExceptionTargetResolutionEntry {
  proposal_index int
  citation_index int
  target_population_ids string[] @description(#"Các rule-version ID phù hợp trong target_records."#)
}

class ExceptionTargetResolutionWindowResult {
  candidate_id string
  resolutions ExceptionTargetResolutionEntry[]
}

function DiscoverLegalExceptions(windows: ExceptionDiscoveryWindowInput[]) -> ExceptionDiscoveryWindowResult[] {
  client OpenAICompatibleProvider
  prompt #"
    {{ _.role("system") }}

    Bối cảnh: xây dựng ontology có bằng chứng cho quy định xử phạt vi phạm hành
    chính về giao thông đường bộ Việt Nam.

    Nhiệm vụ: phát hiện quan hệ ngoại lệ được nguồn nêu trực tiếp cho các rule
    versions trong từng window và chuẩn hóa mọi dẫn chiếu pháp lý.

    - Chỉ đề xuất khi nguồn trực tiếp loại trừ một trường hợp hoặc chỉ ra quy định
      chuyên biệt làm hẹp general rule. Citation đơn thuần, hậu quả cộng dồn và
      LegalChange không tự tạo ngoại lệ.
    - general_population_id là rule bị câu ngoại lệ trực tiếp làm hẹp, không phải
      rule được dẫn chiếu. Evidence phải trực tiếp thuộc chính general rule đó;
      không đảo chiều quan hệ vì một rule khác nhắc tới general rule.
    - evidence_source_ids và source_evidence_id chỉ dùng evidence ID nằm trong
      general record đã chọn. Không dùng nội dung khác dù ở cùng chunk.
    - source_contexts chỉ giúp đọc tiêu đề, thứ tự và phạm vi của nội dung sửa đổi;
      tuyệt đối không dùng chúng làm evidence hay để tạo ngoại lệ cho rule khác.
      Khi provision_path của record đứng độc lập bị lệch do ranh giới chunk, ưu tiên
      tiêu đề sửa đổi được nguồn nêu rõ để xác định Điều đang được sửa.
    - Xác định target_document_number theo ngữ cảnh pháp lý. Với rule nguyên gốc,
      dẫn chiếu nội bộ dùng source_document_number. Với rule nằm trong nội dung
      thay thế của văn bản sửa đổi, `Điều này` hoặc `Nghị định này` có thể thuộc
      văn bản được sửa trong comparison_document_numbers. Các dẫn chiếu khác chỉ
      chuẩn hóa theo corpus_document_numbers khi nguồn đủ căn cứ; không dùng kiến
      thức ngoài.
    - Chuyển dẫn chiếu tương đối như `khoản 2 Điều này` thành locator tuyệt đối từ
      source/general context. Mỗi target_provision_path chỉ gồm locator cần thiết
      theo thứ tự `Điều > Khoản > Điểm`, không chứa Chương/Mục.
    - citation_text phải chép nguyên văn cụm dẫn chiếu ngắn nhất từ evidence đã
      chọn, ví dụ `điểm d khoản 8 Điều 21`. Không trả offset ký tự, không lấy cả
      câu hành vi hoặc câu xử phạt và không cắt giữa từ.
    - Mỗi câu ngoại lệ của một general rule chỉ tạo một discovery. Không tách thành
      nhiều discovery theo từng locator; gom mọi locator trong cùng citation vào
      target_provision_paths theo đúng thứ tự xuất hiện.
    - Không chọn target rule ID ở bước này. Thiếu căn cứ thì không đề xuất hoặc để
      danh sách locator rỗng; không suy đoán. Trả đúng một result cho mỗi
      candidate_id và không lặp exception.

    {{ _.role("user") }}

    Các discovery window cần đánh giá:
    {% for window in windows %}
    === candidate_id={{ window.candidate_id }} ===
    source_document={{ window.source_document_number }}
    comparison_documents={{ window.comparison_document_numbers }}
    corpus_documents={{ window.corpus_document_numbers }}
    source_contexts={{ window.source_contexts }}
    general_records={{ window.general_records }}
    === end candidate_id={{ window.candidate_id }} ===
    {% endfor %}

    {{ ctx.output_format }}
  "#
}

function ResolveLegalExceptionTargets(
  windows: ExceptionTargetResolutionWindowInput[]
) -> ExceptionTargetResolutionWindowResult[] {
  client OpenAICompatibleProvider
  prompt #"
    {{ _.role("system") }}

    Bối cảnh: bước phát hiện đã xác định ngoại lệ và locator dẫn chiếu trong
    ontology xử phạt vi phạm giao thông đường bộ Việt Nam. target_records chỉ gồm
    các rule versions được tra cứu cấu trúc tại đúng văn bản và locator đó.

    Nhiệm vụ: với mỗi candidate, chọn toàn bộ target_population_ids thực sự thuộc
    phạm vi pháp lý của dẫn chiếu.

    - Đánh giá đúng chủ thể, hành vi, điều kiện và phạm vi của general rule,
      exception và từng target record; không chọn theo từ khóa, độ giống, thứ tự
      hay mức phạt.
    - Dẫn chiếu ngoại lệ đã xác lập quan hệ pháp lý tới locator. Chọn các semantic
      row tại locator mà trường hợp bị loại trừ có thể rơi vào; không đòi nhãn hành
      vi phải lặp nguyên văn general rule. Nếu một điều khoản được tách thành nhiều
      row, giữ toàn bộ row thuộc phạm vi dẫn chiếu và chỉ loại row rõ ràng không thể
      áp dụng do khác chủ thể, phương tiện, hành vi hoặc điều kiện pháp lý.
    - conduct_label_vi của general_record là ranh giới semantic của chính row đang
      resolve. Evidence có thể chứa nhiều hành vi anh em trước khi được tách row;
      không dùng các hành vi anh em đó để mở rộng target của row hiện tại.
    - Chỉ dùng ID trong target_records. Có thể chọn nhiều ID khi dẫn chiếu bao phủ
      nhiều rule versions; trả rỗng nếu không record nào phù hợp.
    - Không thay đổi proposal_index, citation_index, locator hoặc nội dung đã phát
      hiện. Trả đúng một resolution cho mỗi candidate đầu vào.

    {{ _.role("user") }}

    Các target window cần resolve:
    {% for window in windows %}
    === candidate_id={{ window.candidate_id }} ===
    candidates={{ window.candidates }}
    === end candidate_id={{ window.candidate_id }} ===
    {% endfor %}

    {{ ctx.output_format }}
  "#
}
