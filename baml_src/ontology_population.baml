enum OntologySubjectKind {
  NaturalPerson
  Organization
}

enum OntologyConditionKind {
  Atomic
  AllOf
  AnyOf
  Not
}

enum OntologyConsequenceKind {
  MonetaryPenalty
  Warning
  PointDeduction
  AdditionalSanction
  RemedialMeasure
  ReferencedPenalty
}

enum OntologyCurrency {
  VND
}

enum OntologyValidityMode {
  DocumentDefault
  ExplicitInterval
  Unresolved
}

class OntologyEvidenceSource {
  evidence_id string
  chunk_id string
  text string
}

class OntologyEvidenceReference {
  evidence_id string
  start_char int? @description(#"Offset Unicode inclusive; null dùng toàn source."#)
  end_char int? @description(#"Offset Unicode exclusive; null dùng toàn source."#)
}

class OntologyConditionEntry {
  condition_id string
  kind OntologyConditionKind
  text_vi string
  operand_ids string[] @description(#"Atomic rỗng; AllOf/AnyOf >=2; Not đúng 1."#)
  evidence OntologyEvidenceReference[]
}

class OntologyConsequenceEntry {
  consequence_id string
  kind OntologyConsequenceKind
  minimum_amount string? @description(#"Số thập phân; chỉ tiền phạt."#)
  maximum_amount string? @description(#"Số thập phân; chỉ tiền phạt."#)
  currency OntologyCurrency?
  point_value int? @description(#"Số nguyên dương; chỉ PointDeduction."#)
  consequence_label_vi string?
  minimum_duration_months int?
  maximum_duration_months int?
  target_label_vi string? @description(#"Đối tượng chịu tác động."#)
  responsible_role_label_vi string? @description(#"Vai trò thực hiện RemedialMeasure."#)
  evidence OntologyEvidenceReference[]
}

class OntologyRuleEntry {
  conduct_label_vi string @description(#"Hành vi đã khóa."#)
  subject_kind OntologySubjectKind?
  role_labels_vi string[]
  vehicle_category_labels_vi string[]
  condition_root_id string? @description(#"ID gốc; null khi không có điều kiện riêng."#)
  conditions OntologyConditionEntry[]
  consequences OntologyConsequenceEntry[]
  validity_mode OntologyValidityMode @description(#"Nguồn hiệu lực."#)
  valid_from string? @description(#"YYYY-MM-DD khi có ngày riêng."#)
  valid_to string? @description(#"YYYY-MM-DD; null khi chưa biết."#)
  provision_path string
  evidence OntologyEvidenceReference[]
}

class OntologyPopulationBatchResult {
  target_chunk_id string @description(#"Phải trùng với target_chunk_id đầu vào."#)
  rules OntologyRuleEntry[]
}

class OntologyTargetConductInput {
  target_chunk_id string
  structural_context string
  locator_context string
  evidence_sources OntologyEvidenceSource[]
}

class OntologyTargetConductEntry {
  conduct_id string @description(#"ID cục bộ duy nhất trong Target."#)
  conduct_label_vi string
  provision_path string
  evidence OntologyEvidenceReference[]
}

class OntologyTargetConductBatchResult {
  target_chunk_id string
  conducts OntologyTargetConductEntry[]
}

class OntologyChunkWindowEntry {
  role ChunkWindowRole
  chunk_id string
  structural_context string @description(#"Metadata cấu trúc; không phải source evidence."#)
  evidence_sources OntologyEvidenceSource[]
}

class OntologyPopulationBindingItem {
  target_chunk_id string
  window OntologyChunkWindowEntry[]
  target_conducts OntologyTargetConductEntry[]
  temporal_overrides DocumentTemporalOverride[]
}

class OntologyBoundRuleEntry {
  target_conduct_id string @description(#"Một conduct_id trong target_conducts."#)
  rule OntologyRuleEntry
}

class OntologyPopulationBindingBatchResult {
  target_chunk_id string
  rules OntologyBoundRuleEntry[] @description(#"Mọi rule variant đủ source support cho locked conduct inventory."#)
}

function ExtractOntologyTargetConductBatch(items: OntologyTargetConductInput[]) -> OntologyTargetConductBatchResult[] {
  client OpenAICompatibleProvider
  prompt #"
    {{ _.role("system") }}

    Bối cảnh: xây dựng ontology chỉ cho quy định xử phạt vi phạm
    hành chính về giao thông đường bộ Việt Nam.

    Mục đích bước 1/6: danh mục đóng conduct.

    Thứ tự ưu tiên bắt buộc: (1) bao phủ đủ Target từ đầu đến cuối; (2) giữ đúng
    chủ thể, hành động, đối tượng và điều kiện nguồn; (3) phân rã thành conduct lá;
    (4) gắn locator. Không được bỏ một nhánh nguồn để làm label ngắn hoặc dễ tách.

    - Cổng domain áp dụng trước coverage: chỉ lấy conduct xử phạt thuộc giao thông
      đường bộ. Loại conduct thuộc đường sắt, hàng hải, hàng không hoặc lĩnh vực
      khác, kể cả khi nằm trong văn bản/Điều sửa đổi đồng thời nhiều lĩnh vực.
      Đánh giá từng conduct theo nội dung trực tiếp, không theo tên chung của văn
      bản. Xác định domain sau khi khôi phục toàn bộ phần dùng chung của câu, nhất
      là bổ ngữ cuối xác định loại phương tiện/lĩnh vực cho nhiều chủ thể hoặc
      hành vi; không được xóa bổ ngữ ngoài domain để biến clause thành đường bộ.
      Việc nhắc `cầu, đường`, tải trọng hoặc vận tải không đủ biến một quy định về
      phương tiện đường sắt thành đường bộ. Target chỉ có nội dung ngoài domain
      thì trả conducts rỗng.
    - Cổng bắt buộc: Target trực tiếp (a) gắn conduct với cảnh cáo, tiền phạt, trừ
      điểm, bổ sung, khắc phục/dẫn chiếu mức phạt; hoặc (b) sửa/thay điều khoản xác
      định bằng danh sách conduct vi phạm giao thông. (b) không cần lặp hậu quả.
    - Loại kỷ luật, hình sự/dân sự, cưỡng chế/thi hành, thủ tục/thẩm quyền, nghĩa vụ
      đúng, quyền/nghĩa vụ chung và điều cấm bị phạt nơi khác. Không tự đảo nghĩa vụ
      tích cực thành vi phạm; "xử lý theo pháp luật" không đủ căn cứ.
    - Xét riêng từng nhánh có hiệu lực pháp lý. Câu khẳng định xử phạt một conduct
      vẫn tạo conduct; nhánh chỉ nói `không xử phạt`, `không ra quyết định xử phạt`,
      miễn hoặc chấm dứt xử phạt thì không tạo conduct và không biến hành vi được
      nhắc lại trong nhánh đó thành conduct. Đây không phải dẫn chiếu mức phạt.
    - Quét Target từ ký tự đầu đến cuối. Audit riêng đoạn mở đầu trước heading mới;
      nếu nó tiếp nối Điểm/Khoản/danh sách, Locator context được xác nhận tiêu đề xử
      phạt/cấu trúc chi phối. Không bỏ đoạn mở đầu vì phía sau có heading mới;
      conduct/evidence vẫn chỉ từ Target.
    - Audit từng Điểm/nhánh, không xuất audit: đánh dấu mọi hành động, không hành
      động, trạng thái vi phạm; mỗi đoạn mở đầu, Điểm và nhánh trực tiếp bị xử phạt
      phải có conduct tương ứng hoặc được loại theo cổng bằng căn cứ rõ ràng. Kiểm
      tra lại coverage sau khi phân rã; không bỏ sót khi Target chuyển Khoản/Điều.
    - Trước khi phân rã, xác định chủ thể chịu xử phạt từ câu dẫn mức phạt/cấu trúc
      chi phối. Mọi label phải giữ nguyên chủ thể đó và đủ loại phương tiện/phạm vi;
      không lấy công cụ, phần mềm, giấy tờ, phương tiện hoặc người thực hiện hành
      động phụ làm chủ thể nếu họ không phải đối tượng bị xử phạt.
    - Mỗi phương án tự đủ bị phạt là một conduct lá. Khôi phục phần dùng chung rồi
      tách: (1) mọi vị ngữ độc lập, kể cả động từ dùng như bổ nghĩa bị động;
      `S V1, V2 O` -> `S V1 O`, `S V2 O`, `O V1, V2 trái quy định` -> hai lá;
      (2) các bổ ngữ lựa chọn độc lập cùng một vị ngữ: `S V O1 hoặc O2` ->
      `S V O1`, `S V O2` nếu thay lần lượt O1/O2 đều tạo câu vi phạm tự đủ.
      `P không đúng A, không đúng B` cũng thành hai lá.
      Label chỉ mô tả conduct; không nối hình thức xử phạt, biện pháp khắc phục hay
      cụm hậu quả như `bị phạt`, `bị tước`, `bị buộc` vào conduct_label_vi.
    - Giữ chung danh sách dấu phẩy sau một lần xuất hiện của vị ngữ khi source không
      lặp vị ngữ và không dùng liên từ lựa chọn để tạo nhánh: `V tại A, B`,
      `làm thay đổi A, B`, `P bằng A/B/C` là một lá. Danh sách có thể gồm thành phần,
      thuộc tính, vị trí, phương thức, phạm vi hoặc ngưỡng; không tách chỉ vì từng
      phần có thể ghép thành câu ngắn. Hai đối tượng pháp lý độc lập nối bằng `hoặc`
      vẫn phải tách.
    - Trước khi trả kết quả, đọc lại từng label theo chủ thể-vị ngữ-bổ ngữ. Chỉ tách
      khi có nhiều vị ngữ hoặc các nhánh lựa chọn độc lập trong source; không dùng
      khả năng viết lại thành câu ngắn làm căn cứ duy nhất. Cụm danh từ chứa nhiều
      động từ cùng nhận một qualifier pháp lý, `O V1, V2 trái quy định`, bắt buộc
      thành `O V1 trái quy định` và `O V2 trái quy định`; quy tắc này ưu tiên hơn
      quy tắc giữ danh sách dấu phẩy. Ví dụ tổng quát: `phương tiện chế tạo, cải tạo
      trái quy định` phải thành hai conduct cho `chế tạo` và `cải tạo`. Không được
      trả label vẫn chứa hai động từ cùng đứng trước `trái quy định`.
    - Với `S cho phép/chở/để T V1, V2 hoặc V3`, giữ S là chủ thể chịu xử phạt nhưng
      tách V1/V2/V3 nếu từng hành động của T tự tạo thành phương án bị cấm. Chỉ gộp
      các hành động hoặc khả năng khi pháp lý buộc chúng cùng cấu thành một yêu cầu.
      Áp dụng tương tự cho động từ lồng sau mục đích/khả năng: `S P O để/thực hiện
      việc V1, V2, V3` -> lặp toàn bộ khung `S P O để V1`, `... để V2`, `... để V3`
      khi từng V là một khả năng pháp lý độc lập. Động từ lồng vẫn là vị ngữ cần audit.
    - Chủ thể label là chủ thể chịu xử phạt trong mệnh đề chi phối, không phải người
      thực hiện hành động phụ. Với `S tự V hoặc chở/cho phép/để T V`, tạo `S tự V`
      và `S chở/cho phép/để T V`; không đổi nhánh thứ hai thành `T V`. Kiểm tra lại
      mọi label đều bắt đầu từ S, kể cả khi câu nguồn bắt đầu bằng thuộc tính của O.
    - Nhánh rút gọn phải tự đủ nghĩa: khôi phục cả chủ thể và vị ngữ chi phối bị
      lược, cùng phủ định, đối tượng, qualifier/phạm vi dùng chung. `S V O1; O2`
      mặc định tạo `S V O1`, `S V O2` nếu O2 không mở một vị ngữ chính mới; phải
      lặp lại V trong label, không đổi `S V O` thành `S O` hay hành động khác của S.
      Với `S V1 hoặc V2 C`, nếu C là bổ ngữ dùng chung thì tạo `S V1 C` và
      `S V2 C`; không trả nhánh cụt chỉ có `S V1` hoặc `S V2`.
    - Bảo toàn toàn bộ điều kiện xác định đối tượng, kể cả phần trong ngoặc hoặc
      mệnh đề cuối áp dụng cho nhiều nhánh; ưu tiên đủ nghĩa hơn label ngắn. Với cụm
      danh từ pháp lý được lặp lại hoặc rút gọn, mỗi lá giữ đủ giấy tờ, loại xe, bên
      giữ/cấp và phạm vi. Không thêm động từ/hành động để nối chủ thể với hành vi;
      chỉ dùng nội dung được source trực tiếp hỗ trợ. Không rút ngắn phần trong ngoặc
      nếu việc bỏ nó làm thay đổi đối tượng hoặc điều kiện áp dụng của leaf. Phân loại
      chủ thể để bước 2 bind.
    - Loại định nghĩa, nguyên tắc và locator thuần túy. Dẫn chiếu phải trực tiếp xác
      định conduct và mức/điều khoản xử phạt. Đoạn chỉ thi hành, khắc phục hoặc tính
      hậu quả/số lợi từ conduct dẫn chiếu không tạo conduct mới.
    - Không dùng kiến thức ngoài. Với văn bản sửa đổi, provision_path là điều khoản
      được sửa/thay; toàn bộ Chương/Điều/Khoản/Điểm dùng để chứa hoặc đánh số thao
      tác sửa đổi chỉ là provenance và phải bị loại khỏi path. Lấy locator đích từ
      câu `sửa đổi/bổ sung ... Điều ...` đang chi phối; nếu phần trích dẫn tiếp tục
      ở đầu Target thì dùng Locator context để hoàn thành chính locator đích đó.
      Không nối path của văn bản sửa đổi với path của điều khoản được sửa.
      Chỉ áp dụng việc loại ancestor khi source/context thể hiện rõ đây là thao tác
      sửa đổi một điều khoản khác. Với quy định trực tiếp, phải giữ đủ mọi ancestor
      `Chương`, `Mục`, `Điều` đã có trong Cấu trúc rồi nối `Khoản`, `Điểm` của Target;
      không tự rút path xuống chỉ còn `Điều`.
      provision_path chỉ gồm mã locator theo `Chương > Mục > Điều > Khoản > Điểm`,
      đúng dấu ` > `; không kèm tên/tiêu đề Chương, Mục, Điều và không đoán/lặp
      locator. Context không cấp evidence hay conduct.
    - Mỗi conduct phải dẫn ít nhất một evidence_id được liệt kê trong đúng mục
      Evidence của target_chunk_id hiện tại. Không tự tạo, ghi nhớ hoặc tái sử dụng
      evidence_id từ target hay lần chạy khác; chỉ trả ID nguyên văn đã được cấp.
    {{ _.role("user") }}

    {% for item in items %}
    === target_chunk_id={{ item.target_chunk_id }} ===
    Cấu trúc:
    {{ item.structural_context }}
    Locator context:
    {{ item.locator_context }}
    Evidence:
    {% for evidence in item.evidence_sources %}
    [{{ evidence.evidence_id }}] chunk_id={{ evidence.chunk_id }}
    {{ evidence.text }}
    {% endfor %}
    === end ===
    {% endfor %}

    {{ ctx.output_format }}
  "#
}

function BindOntologyPopulationBatch(items: OntologyPopulationBindingItem[]) -> OntologyPopulationBindingBatchResult[] {
  client OpenAICompatibleProvider
  prompt #"
    {{ _.role("system") }}

    Bối cảnh: xây dựng ontology có bằng chứng chỉ cho quy định xử phạt vi phạm
    hành chính về giao thông đường bộ Việt Nam.

    Mục đích bước 2/6: gắn phạm vi và hậu quả có source support vào conduct đã khóa.

    Cổng hoàn thành: với từng conduct/subject variant, phải quét hết window và xử
    lý mọi clause hậu quả dẫn chiếu đúng provision_path trước khi trả kết quả.
    Output chỉ có tiền phạt trực tiếp trong khi window còn clause cùng locator về
    trừ điểm, bổ sung, khắc phục hoặc dẫn chiếu là output chưa hoàn thành.
    Trong một Điểm/Khoản hậu quả có nhiều câu, phải audit độc lập từng câu sau dấu
    chấm và từng nhóm locator sau dấu chấm phẩy; câu đầu không đại diện cho các câu
    sau. Mọi câu vẫn kế thừa loại hậu quả từ phần dẫn của Điểm/Khoản đó.

    - Giữ nguyên target conducts: trả ít nhất một rule cho từng target_conduct_id,
      giữ đúng conduct_label_vi và provision_path. Inventory rỗng thì rules rỗng;
      không bỏ, gộp hoặc tạo conduct.
    - Audit toàn window cho từng conduct. Chỉ bind khi source hoặc locator xác định
      nó bị chi phối; gần nhau hay giống chữ không đủ. Với từng clause hậu quả,
      đối chiếu toàn bộ danh sách locator rồi thu đủ tiền phạt, cảnh cáo, trừ điểm,
      bổ sung, khắc phục và dẫn chiếu áp dụng cộng dồn. Không dừng sau hậu quả đầu;
      không bỏ hậu quả vì nó nằm xa conduct hoặc có thêm điều kiện. Nếu locator
      khớp nhưng hậu quả chỉ áp dụng khi có điều kiện, vẫn gắn hậu quả và biểu diễn
      điều kiện đó trong cây conditions. Trước khi trả kết quả, lập checklist nội
      bộ cho từng provision_path và đối chiếu lại mọi clause hậu quả trong window.
      Với mỗi nhóm conduct_id cùng provision_path, sau khi xác định các subject_kind
      hợp lệ, lập ma trận từng conduct_id x từng subject_kind x từng clause hậu quả
      có locator khớp: mặc định clause áp dụng cho mọi conduct và subject variant
      trong nhóm; chỉ loại một ô khi chính source giới hạn nhánh đó bằng chủ thể,
      đối tượng, hành động hoặc điều kiện không thể áp dụng. Không được gắn hậu quả
      cho một conduct/subject rồi bỏ các conduct đồng cấp hoặc subject variant còn
      lại chỉ vì chúng được tách từ cùng Điểm hoặc cùng câu phạt.
      Khi clause dẫn chiếu cả Điểm/Khoản, khác biệt câu chữ giữa các conduct lá
      trong Điểm/Khoản không tự tạo ngoại lệ. Clause có nhiều hậu quả cộng dồn phải
      giữ đủ từng hậu quả. Mỗi rule phải có hậu quả.
      Cổng cuối phải audit từng ô ma trận riêng: tạo chữ ký gồm loại hậu quả, đích,
      khoảng giá trị và điều kiện cho từng conduct_id/subject_kind. Các ô cùng
      provision_path phải có cùng tập chữ ký hậu quả phi tiền tệ khi clause không
      giới hạn chủ thể hoặc conduct; khác số tiền và trừ điểm chỉ cho người trực
      tiếp điều khiển là các khác biệt hợp lệ. Một ô đã đủ không chứng minh ô khác
      đã đủ. Nếu thiếu chữ ký ở bất kỳ ô nào, phải bổ sung rule cho chính ô đó rồi
      audit lại toàn bộ ma trận trước khi trả output.
    - Một rule chỉ được chứa các hậu quả có cùng tập điều kiện áp dụng. Hậu quả vô
      điều kiện phải ở rule có condition_root_id=null; hậu quả có các điều kiện
      khác nhau phải tách thành các rule cho cùng target_conduct_id. Không được lấy
      điều kiện của một clause để chi phối tiền phạt hoặc hậu quả từ clause khác.
      Kiểm tra từng consequence trong rule đều được source áp dụng dưới chính cây
      conditions của rule đó. Evidence của mỗi condition và consequence phải chứa
      chính câu phát biểu điều kiện hoặc hậu quả ấy. Không dùng Điểm/câu liền kề,
      dù cùng chunk hoặc cùng loại hậu quả, làm evidence thay cho clause thực tế;
      nếu hai hậu quả ở hai clause thì mỗi hậu quả dẫn evidence của clause riêng.
      Audit provenance cuối cùng cho từng evidence ref: đọc lại riêng text của ref
      và hỏi text đó có tự phát biểu assertion đang dẫn hay không. Nếu không chứa
      chính nội dung hoặc diễn đạt tương đương rõ ràng của condition/consequence,
      ref đó sai và phải bỏ hoặc thay bằng ref của clause thực tế. Một ref đúng
      khác trong cùng danh sách không hợp thức hóa ref sai. Evidence của condition
      chỉ gồm clause phát biểu condition. Nếu consequence kế thừa loại pháp lý từ
      câu dẫn chung, evidence có thể gồm câu dẫn đó và chính clause consequence;
      không thêm các Điểm/câu sibling nằm giữa chúng.
    - Cụm `nếu có`, `nếu đã được cấp` hoặc tương đương không cho phép bỏ hậu quả;
      tạo rule hậu quả với điều kiện có/đã được cấp đối tượng chịu tác động. Điều
      kiện này phải nằm trong cây conditions, không chỉ được giữ như phần mô tả
      trong consequence_label_vi. Nếu rule còn hậu quả vô điều kiện, tách hậu quả
      có điều kiện sang rule riêng cho cùng conduct và subject variant. Cụm
      `trừ trường hợp X` không xóa hậu quả chung; tạo điều kiện Not(X) cho rule đó.
      Nếu cùng conduct có một hậu quả khi X và hậu quả khác khi Not(X), phải giữ cả
      hai rule có điều kiện đối ứng. Không được bỏ âm bản hoặc dương bản.
    - Tiền phạt có thể khác theo NaturalPerson/Organization, nhưng hậu quả bổ sung,
      khắc phục và điều kiện không tự giới hạn theo subject_kind. Khi câu dẫn hoặc
      clause hậu quả áp dụng cho cả cá nhân và tổ chức, phải nhân từng hậu quả đó
      sang cả hai variant; hoàn thành NaturalPerson không thay thế Organization và
      ngược lại. Chỉ giữ một phía khi source trực tiếp giới hạn chủ thể.
    - Xác định chủ thể từ câu phạt, tiêu đề Điều và structural context trước khi
      bind hậu quả. Vai trò chỉ một con người thực hiện được là NaturalPerson, dù
      không có từ "cá nhân"; vai trò chỉ một pháp nhân/tổ chức thực hiện được là
      Organization. Nếu tiêu đề xác định người điều khiển và loại xe, không để
      subject_kind null; giữ role và vehicle category tương ứng.
      Xét tính khả thi riêng cho từng conduct lá, không sao chép máy móc mọi loại
      chủ thể từ câu dẫn chung. Nhánh yêu cầu chính chủ thể trực tiếp điều khiển,
      lái hoặc thực hiện hành động thể chất của con người chỉ có NaturalPerson;
      Organization không được tạo variant đó. Câu dẫn có cả cá nhân và tổ chức vẫn
      cho phép hai variant ở các nhánh mà cả hai thực sự có thể thực hiện, như giao
      phương tiện hoặc để người khác điều khiển.
    - Khi một câu phạt nêu riêng mức/phạm vi cho cá nhân và tổ chức, mọi conduct
      thuộc câu đó phải có đủ hai variant: NaturalPerson chỉ nhận mức cá nhân,
      Organization chỉ nhận mức tổ chức. Không tạo một rule subject null, không
      gộp hai mức vào một consequence và không chỉ tách một số conduct trong cùng
      danh sách. Trước khi trả kết quả, kiểm tra đủ hai variant cho từng conduct_id.
      consequence_label_vi phải thống nhất với subject_kind và các trường số tiền.
      Chỉ null khi source thật sự không xác định loại chủ thể; không dùng
      target_label_vi thay subject. Tách thêm variant khi điều kiện/phạm vi khác nhau.
    - Phân loại pháp lý: Warning cho cảnh cáo; ReferencedPenalty cho dẫn chiếu
      mức phạt (target_label_vi là đích dẫn chiếu); AdditionalSanction chỉ khi source
      nêu hình thức xử phạt bổ sung và đặt hậu quả trong phạm vi đó; không phân loại
      chỉ vì có từ `bị`, `tước`, `tịch thu` hoặc `thu hồi`. RemedialMeasure phải giữ
      đối tượng được tác động trong target_label_vi cùng vai trò thực hiện nếu có;
      dùng loại này cho biện pháp khắc phục/khôi phục/điều chỉnh/thu hồi hồ sơ, giấy
      tờ sau vi phạm.
      Điều kiện dùng cây AND/OR/NOT. Với văn bản sửa đổi định nghĩa/thay conduct
      bên trong một Khoản xử phạt nhưng không lặp lại mức phạt, tạo
      ReferencedPenalty tới chính Khoản/Điều được sửa cho mọi conduct lá tương ứng;
      hậu quả bổ sung hoặc khắc phục không thay thế dẫn chiếu mức phạt nền. Không
      tạo ReferencedPenalty này nếu window đã cho mức phạt trực tiếp hoặc source
      xác định conduct không bị xử phạt. Khi clause khắc phục/tịch thu dẫn chiếu
      locator rồi mô tả giấy tờ, phương tiện hoặc trạng thái cụ thể, đối chiếu nội
      dung từng conduct: giữ clause cùng điều kiện cho mọi conduct có thể thỏa mô
      tả đó, thay vì bỏ toàn bộ hoặc áp dụng cho nhánh không liên quan.
    - Chọn validity_mode theo temporal_overrides: DocumentDefault nếu không áp
      override (để hai ngày null); ExplicitInterval nếu provision có ngày riêng;
      Unresolved nếu căn cứ riêng chưa cho ngày (để hai ngày null). Đối chiếu đúng
      Điều/Khoản/Điểm, không lan sang provision khác hoặc suy từ kiến thức ngoài.
    - Ngay trước khi trả output, so sánh các sibling cùng provision_path. Nếu một
      conduct/subject variant chỉ có tiền phạt nhưng sibling có hậu quả phi tiền tệ
      từ clause áp dụng chung và source không loại variant đó, output chưa hoàn
      thành: sao chép đúng chữ ký clause sang variant thiếu, dùng cùng điều kiện và
      evidence của clause thực tế. Lặp đến khi không còn sibling thiếu hậu quả.
      Sau đó quét ngược từng clause hậu quả trong toàn window: mỗi clause phải được
      đánh dấu đã có rule cho mọi variant áp dụng hoặc có lý do source-explicit để
      không áp dụng. Clause chưa xuất hiện ở bất kỳ sibling nào vẫn không được bỏ;
      đặc biệt hậu quả kèm `nếu có` phải có rule điều kiện. Nếu còn clause áp dụng
      chưa được biểu diễn, bổ sung nó rồi chạy lại cả audit clause và sibling.
    - Không tạo LegalChange, concept relation hay occurrence. Trả một result mỗi
      target_chunk_id sau khi xử lý mọi locked conduct.

    {{ _.role("user") }}

    {% for item in items %}
    === target_chunk_id={{ item.target_chunk_id }} ===
    Temporal overrides:
    {% for override in item.temporal_overrides %}
    provisions={{ override.affected_provision_paths }}
    validity={{ override.valid_from }} -> {{ override.valid_to }}
    unresolved_basis={{ override.unresolved_basis_vi }}
    ---
    {% endfor %}
    Locked target conducts:
    {% for conduct in item.target_conducts %}
    [{{ conduct.conduct_id }}]
    conduct_label_vi={{ conduct.conduct_label_vi }}
    provision_path={{ conduct.provision_path }}
    {% for evidence in conduct.evidence %}
    target_evidence={{ evidence.evidence_id }}
    {% endfor %}
    {% endfor %}
    {% for entry in item.window %}
    [{{ entry.role }}] chunk_id={{ entry.chunk_id }}
    {% if entry.structural_context %}
    Structural context (không phải source evidence):
    {{ entry.structural_context }}
    {% endif %}
    Evidence sources:
    {% for evidence in entry.evidence_sources %}
    [{{ evidence.evidence_id }}] chunk_id={{ evidence.chunk_id }}
    {{ evidence.text }}
    {% endfor %}
    ---
    {% endfor %}
    === end ===
    {% endfor %}

    {{ ctx.output_format }}
  "#
}
