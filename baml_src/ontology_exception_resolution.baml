class ExceptionCitationDiscoveryEntry {
  citation OntologyEvidenceReference @description(#"Evidence chứa nguyên văn dẫn chiếu."#)
  target_document_number string @description(#"Số văn bản đích đã chuẩn hóa; dùng văn bản nguồn nếu dẫn chiếu nội bộ."#)
  target_provision_paths string[] @description(#"
    Các locator đích tuyệt đối theo thứ tự `Điều > Khoản > Điểm`, không chứa Chương/Mục.
  "#)
}

class LegalExceptionDiscoveryEntry {
  general_population_id string @description(#"Một rule ID trong general_records."#)
  evidence OntologyEvidenceReference[] @description(#"Evidence trực tiếp nêu đầy đủ ngoại lệ."#)
  citations ExceptionCitationDiscoveryEntry[] @description(#"Các dẫn chiếu tạo nên phạm vi ngoại lệ."#)
  rationale_vi string @description(#"Lý do ngắn gọn cho quan hệ ngoại lệ."#)
}

class ExceptionDiscoveryWindowInput {
  candidate_id string
  source_document_number string
  corpus_document_numbers string[]
  general_records OntologyPopulationSummary[]
  source_evidence_sources OntologyEvidenceSource[]
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
    - Evidence và citation chỉ chọn từ evidence của general record tương ứng hoặc
      source_evidence_sources cùng nguồn với record đó; không diễn giải lại.
    - Với dẫn chiếu nội bộ, target_document_number là source_document_number.
      Với dẫn chiếu sang văn bản khác, chuẩn hóa theo corpus_document_numbers nếu
      nguồn đủ căn cứ; không dùng kiến thức ngoài.
    - Chuyển dẫn chiếu tương đối như `khoản 2 Điều này` thành locator tuyệt đối từ
      source/general context. Mỗi target_provision_path chỉ gồm locator cần thiết
      theo thứ tự `Điều > Khoản > Điểm`, không chứa Chương/Mục.
    - citation phải là span nguyên văn ngắn nhất chứa riêng cụm dẫn chiếu, ví dụ
      `điểm d khoản 8 Điều 21`, không lấy cả câu hành vi hoặc câu xử phạt.
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
    corpus_documents={{ window.corpus_document_numbers }}
    general_records={{ window.general_records }}
    source_evidence_sources={{ window.source_evidence_sources }}
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
