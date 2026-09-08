enum OntologyConceptKind {
  RegulatedConduct
  LegalSubjectRole
  VehicleCategory
}

enum OntologyConceptRelation {
  SameConcept
  LeftBroaderThanRight
  LeftNarrowerThanRight
  Related
}

class OntologyConceptOccurrenceInput {
  occurrence_id string
  kind OntologyConceptKind
  label_vi string
  population_id string
  document_number string
  provision_path string
  evidence OntologyEvidenceSource[]
}

class ExistingConceptRelationInput {
  left_occurrence_id string
  right_occurrence_id string
  relation OntologyConceptRelation
  preferred_occurrence_id string
  rationale_vi string
}

class ConceptResolutionInventoryInput {
  candidate_id string
  kind OntologyConceptKind
  occurrences OntologyConceptOccurrenceInput[]
  existing_proposals ExistingConceptRelationInput[]
}

class ConceptRelationProposalEntry {
  left_occurrence_id string
  right_occurrence_id string
  relation OntologyConceptRelation
  preferred_occurrence_id string @description(#"
    SameConcept: chọn một occurrence thuộc cùng nhóm SameConcept trong inventory;
    quan hệ khác: chuỗi rỗng.
  "#)
  left_evidence OntologyEvidenceReference[] @description(#"Evidence của left occurrence."#)
  right_evidence OntologyEvidenceReference[] @description(#"Evidence của right occurrence."#)
  rationale_vi string @description(#"Lý do ngắn gọn dựa trên ý nghĩa pháp lý."#)
}

class ConceptResolutionInventoryResult {
  candidate_id string
  proposals ConceptRelationProposalEntry[]
}

function ProposeOntologyConceptRelations(inventories: ConceptResolutionInventoryInput[]) -> ConceptResolutionInventoryResult[] {
  client OpenAICompatibleProvider
  prompt #"
    {{ _.role("system") }}

    Bối cảnh: xây dựng ontology có bằng chứng chỉ cho quy định xử phạt vi phạm
    hành chính về giao thông đường bộ Việt Nam; không dùng cho vụ việc hoặc domain
    khác.

    Mục đích bước 6/6: chuẩn hóa concept occurrences thành identity hoặc taxonomy
    dùng chung mà không làm mất khác biệt về nghĩa pháp lý.

    Nhiệm vụ: chuẩn hóa toàn bộ nhãn trong mỗi inventory cùng loại và chỉ trả
    toàn bộ quan hệ có giá trị cho controlled vocabulary dùng chung. Existing
    proposals là baseline từ lượt trước: giữ lại mọi proposal vẫn đúng, sửa
    proposal sai và bổ sung proposal thiếu. Kết quả trả về là tập thay thế hoàn
    chỉnh, không chỉ là phần thay đổi.

    - Danh sách đầu vào là tập kiểm kê đóng: phải rà toàn bộ occurrence, không
      chỉ trả một số ví dụ nổi bật. Làm theo hai pha và hoàn tất pha trước khi
      sang pha sau. Pha identity: phân hoạch toàn bộ bảng nhãn thành các nhóm
      SameConcept và các singleton. Pha taxonomy: trên các nhóm identity đó, tìm
      quan hệ cha-con trực tiếp và quan hệ liên quan còn lại. Không kết thúc sau
      khi chỉ tìm thấy các cặp nổi bật đầu tiên.
    - Chỉ dùng meaning và evidence hai phía; hình thức nhãn chỉ giúp nhận diện
      cặp cần đọc kỹ, không tự nó chứng minh quan hệ. Không quyết định chỉ bằng
      lexical overlap, cùng locator, mức phạt hoặc loại rule.
    - SameConcept khi hai occurrence cùng chỉ một loại chủ thể hoặc phương tiện
      về mặt pháp lý. Không yêu cầu chúng xuất hiện trong cùng hành vi, điều
      khoản, mức phạt hoặc hoàn cảnh áp dụng; khác biệt giữa các quy định chứa
      occurrence không tự làm thay đổi identity của concept. Chỉ giữ identity
      riêng khi chính nghĩa của nhãn hoặc evidence bổ sung qualifier làm hẹp hay
      đổi loại chủ thể/phương tiện.
    - Khi một nhãn dài hơn chỉ diễn đạt lại chức năng vốn có của loại phương tiện
      hoặc vai trò trong nhãn ngắn, không tạo thêm điều kiện phân loại thực sự,
      hai nhãn vẫn là SameConcept. Chỉ dùng Broader/Narrower khi phần bổ sung xác
      định một tập con pháp lý có thể phân biệt với các thành viên khác của loại.
    - Một vai trò, chức năng hoặc loại chuyên biệt không SameConcept với khái
      niệm bao quát, kể cả khi evidence gọi nó bằng tên nhóm; khi có căn cứ bao
      hàm thì dùng Broader/Narrower. Khác ngưỡng, mức độ, địa điểm, loại đường,
      phương tiện, chủ thể, tái phạm, gây tai nạn hoặc qualifier làm đổi phạm vi
      concept thì không merge.
    - Broader/Narrower chỉ khi một nghĩa pháp lý bao hàm nghĩa kia. Nhãn nhóm
      trực tiếp liệt kê hoặc bao gồm một loại thành viên là concept rộng hơn loại
      thành viên đó, không phải Related. Related chỉ dùng cho liên hệ có giá trị
      nhưng không có identity hoặc bao hàm; cặp phân biệt hoặc không liên quan thì bỏ.
    - Chỉ xuất SameConcept, Broader/Narrower hoặc Related khi có căn cứ rõ.
      Không xuất Distinct và không xuất pair không có quan hệ đáng lưu.
      SameConcept chọn nguyên occurrence_id của một thành viên trong cùng nhóm
      làm preferred_occurrence_id; occurrence này không bắt buộc là endpoint của
      từng proposal. Quan hệ khác để chuỗi rỗng. Không tự viết label. Mọi
      SameConcept trong cùng một nhóm bắc cầu phải chọn cùng preferred occurrence.
    - Trong pha identity, đọc lại toàn bộ bảng nhãn theo nghĩa mà bỏ qua khác biệt
      chữ hoa/thường và cách viết. Các nhãn chỉ khác các yếu tố này phải nằm cùng
      nhóm SameConcept nếu evidence không cho thấy khác phạm vi pháp lý. Các thuật
      ngữ dùng từ khác nhau cũng phải được đối chiếu theo nghĩa và evidence, không
      được bỏ qua chỉ vì không giống nhau về mặt chữ.
    - Mỗi nhóm SameConcept có N thành viên phải xuất đúng N-1 proposal tạo thành
      một graph liên thông. Chọn một preferred occurrence trong nhóm và dùng cùng
      preferred_occurrence_id đó cho mọi proposal của nhóm; preferred occurrence
      không bắt buộc xuất hiện ở hai endpoint của từng proposal. Không xuất thêm
      các cặp bắc cầu suy ra được.
    - Trong pha taxonomy, xét từng nhóm identity như một concept. Ưu tiên cha trực
      tiếp bao quát gần nhất có trong inventory. Mỗi concept chuyên biệt phải có
      quan hệ với cha trực tiếp nếu cha đó hiện diện và evidence xác nhận bao hàm;
      concept phương tiện rất rộng vẫn được dùng làm gốc taxonomy, không bỏ chỉ
      vì nó ít qualifier. Tránh lặp quan hệ bắc cầu suy ra được.
    - Trước khi trả kết quả, dùng bảng nhãn làm checklist: xác nhận tổng số thành
      viên của mọi nhóm identity và đủ N-1 proposal cho từng nhóm nhiều thành viên.
      Sau đó rà từng nhóm theo chiều ngược lại để tìm nhãn tương đương còn bị tách,
      concept chuyên biệt vẫn chưa nối với cha hiện diện, hoặc quan hệ bao hàm bị
      ghi nhầm thành Related. Occurrence không có quan hệ có căn cứ được phép không
      xuất proposal.
    - Mỗi unordered pair xuất tối đa một lần. Evidence và occurrence ID phải
      thuộc đúng inventory. Không tạo axiom, rule identity, applicability hoặc
      legal priority.
    - Trả một result cho mỗi candidate_id, với proposals=[] nếu không có quan hệ.

    {{ _.role("user") }}

    Các controlled-vocabulary inventory cần đánh giá:
    {% for inventory in inventories %}
    === candidate_id={{ inventory.candidate_id }} kind={{ inventory.kind }} ===
    Bảng nhãn bắt buộc kiểm kê:
    {% for occurrence in inventory.occurrences %}
    - occurrence_id={{ occurrence.occurrence_id }} label_vi={{ occurrence.label_vi }}
    {% endfor %}

    Occurrences và evidence chi tiết:
    {% for occurrence in inventory.occurrences %}
    {{ occurrence }}
    {% endfor %}

    Existing proposals cần audit và hợp nhất vào kết quả đầy đủ:
    {% for proposal in inventory.existing_proposals %}
    {{ proposal }}
    {% endfor %}
    === end candidate_id={{ inventory.candidate_id }} ===
    {% endfor %}

    {{ ctx.output_format }}
  "#
}
