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
    Quy định về giấy phép, phù hiệu, đăng ký, đăng kiểm, bảo hiểm, vận tải,
    đào tạo, sát hạch, sức khỏe, hạ tầng hoặc biển báo chỉ thuộc domain khi
    chính Source text trực tiếp dùng nó để xác định vi phạm, áp dụng chế tài
    hoặc thực hiện một bước xử lý vi phạm. Không trích nghĩa vụ quản lý hay
    thủ tục cấp, đổi, cấp lại thông thường chỉ vì có thể bị vi phạm.

    Thứ tự ưu tiên bắt buộc: (1) chính xác và có căn cứ trực tiếp; (2) giữ
    đủ chủ thể, phạm vi, điều kiện và ngoại lệ; (3) viết nguyên tử, độc lập;
    (4) bao phủ nội dung còn lại. Đây là ground truth nên ưu tiên precision
    hơn recall. Khi input không đủ để xác lập chắc chắn một thành phần làm
    thay đổi nghĩa hoặc phạm vi áp dụng, bỏ fact đó; không suy đoán để tăng
    số lượng fact. Việc bỏ một fact chưa rõ không ảnh hưởng các fact rõ ràng
    khác trong cùng Target.

    Reading window:
    - Target là chunk cần trích và phải chứa một phần source text trực
      tiếp cho mỗi fact.
    - ArticleContext là chunk nguồn ở xa được khôi phục từ ancestry của
      parser. Chỉ dùng Source text của nó để xác lập tiêu đề, locator và
      phạm vi cấu trúc cho Target; được đưa vào evidence khi statement
      dùng thông tin đó, nhưng không trích quy tắc riêng từ ArticleContext.
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
    - Với FollowingTarget, chỉ dùng khi Target bị cắt giữa đúng một câu
      ngữ pháp chưa hoàn thành và phần đầu FollowingTarget nối trực tiếp
      để kết thúc câu đó. Không dùng nội dung bắt đầu một câu, mục, điểm,
      hàng bảng hoặc phần tử danh sách mới, kể cả khi nó tiếp tục cùng
      khoản, cùng câu dẫn hoặc cùng quy tắc tổng thể; nội dung đó thuộc
      target kế tiếp.
    - Nếu Target kết thúc bằng câu dẫn danh sách mà các mục bắt đầu ở
      FollowingTarget, không tạo fact cho danh sách đó. Nếu Target đã
      chứa một số mục và FollowingTarget chứa các mục còn lại, chỉ trích
      các mục nằm trong Target; không gộp toàn bộ danh sách.
    - FollowingTarget chỉ cung cấp phần đuôi của câu bị cắt giữa chừng;
      không dùng để hoàn chỉnh quy tắc, danh sách hoặc cấu trúc chỉ mới
      được giới thiệu trong Target.
    - Ownership gate: trước khi xuất từng fact, xác định phần làm fact
      khác biệt như hành vi, giá trị, ngưỡng, điều kiện hoặc hậu quả. Ít
      nhất một phần khác biệt đó phải nằm trong Source text của Target.
      Nếu nó chỉ xuất hiện trong FollowingTarget thì loại fact, dù Target
      có tiêu đề bảng, câu dẫn, phạm vi chung hoặc tên quy tắc.
    - Không trích lại cùng một quy tắc dưới dạng gộp và dạng tách.
    - Structural context là phạm vi cấu trúc do parser khôi phục. Phải
      dùng nó để viết đủ chủ thể, loại phương tiện và phạm vi Điều khi
      Source text phụ thuộc câu dẫn hoặc chỉ nêu hành vi cụ thể. Giữ
      riêng hai lớp: phạm vi chủ thể từ Điều và đối tượng/phương thức
      cụ thể trong hành vi; không dùng cụm phương tiện trong hành vi để
      thay thế hoặc làm rộng phạm vi chủ thể. Structural context không
      phải evidence và không được dùng để tạo quy tắc mới.

    Output contract:
    Thực hiện ba cổng bắt buộc cho từng fact. Nếu không thể sửa statement
    và evidence để đạt đủ cả ba cổng mà không suy đoán thì xóa fact đó:
    1. Cổng định danh: liệt kê từng cụm định danh trong statement như
       "Điều N", "khoản N" hoặc "điểm X" và ánh xạ từng cụm tới đúng
       chunk có nguyên văn cụm đó trong Source text. Evidence bắt buộc
       chứa chunk_id đã ánh xạ; chunk đứng gần nhưng không chứa cụm không
       thể thay thế. Structural context không được dùng làm căn cứ. Nếu
       source chỉ ghi "Điều này" hoặc "khoản này", tìm đúng chunk chứa
       định danh trong window để dẫn; nếu không có thì chỉ viết lại không
       cần định danh khi chắc chắn không làm đổi nghĩa hoặc phạm vi. Nếu
       không chắc chắn, xóa fact.
    2. Cổng evidence: evidence phải tối thiểu nhưng đủ, luôn chứa Target
       và mọi chunk cung cấp trực tiếp ít nhất một thành phần của statement.
       Không thêm chunk không cung cấp nội dung nào. Thiếu chunk chứa số,
       dẫn chiếu, giá trị, phạm vi hoặc thành phần pháp lý thì xóa fact.
    3. Cổng phạm vi: câu tác nghiệp phụ thuộc tiêu đề, biển báo, bảng,
       nhóm đối tượng hoặc mục cấu trúc phải nêu phạm vi gần nhất được
       input xác lập. Nếu cùng câu có thể thuộc nhiều phạm vi mà không
       thể phân biệt bằng input, không xuất câu đó như một fact độc lập.

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
    - Lập inventory theo đúng thứ tự Target trước khi viết output: xét
      riêng mọi câu, vế sau dấu chấm phẩy và hành động được nối bằng
      "và" hoặc ";". Mỗi nghĩa vụ, quyền hạn, điều kiện hoặc hành động
      có thể áp dụng độc lập phải được xét riêng; không xem vế sau là chi
      tiết phụ chỉ vì nó dùng chung chủ thể với vế trước. Chỉ xuất khi
      chủ thể, hành động, phạm vi và điều kiện đều đủ rõ từ input.
    - Inventory phải bao phủ cả mệnh đề hoàn chỉnh ở đầu hoặc cuối
      Target, kể cả Target kết thúc ngay sau dấu chấm phẩy. Với từng cụm
      động từ quy phạm như phải, không được, chưa giải quyết, thực hiện,
      ban hành, gửi, quản lý, in hoặc dán, phải xét khả năng tạo fact,
      nhưng bỏ qua nếu mệnh đề chưa đủ rõ hoặc phụ thuộc context không có.
      Dấu chấm phẩy ở cuối Target không tự biến mệnh đề đủ chủ ngữ, hành
      động và đối tượng thành câu chưa hoàn thành. Ví dụ tổng quát, nguồn
      "A quản lý X; thực hiện Y;" phải bảo toàn cả X và Y trong output.
    - Cấu trúc đối lập trạng thái như "chưa/không ...; sau khi ... thì
      thực hiện ..." chứa ít nhất hai nghĩa pháp lý: hạn chế trước điều
      kiện và cách xử lý sau khi điều kiện được đáp ứng. Xét riêng cả hai,
      nhưng chỉ xuất vế nào có đủ căn cứ để tự đứng độc lập.
    - Dẫn chiếu phụ thuộc là thành phần bắt buộc nếu nó quyết định phạm vi
      hoặc điều kiện áp dụng. Nếu không thể xác lập an toàn số Điều/khoản
      hay nội dung được dẫn thì bỏ fact, không giữ một phiên bản rộng hơn.
    - Khi chính Target chứa một locator cần thiết, giữ nguyên locator và
      trích quy tắc mà không cần đọc hay diễn giải nội dung được dẫn.
      Dùng các chunk tiêu đề trong window chỉ để thay "Điều này" bằng
      đúng số Điều. Nếu không đủ input để chuẩn hóa locator cần thiết thì
      bỏ fact thay vì suy đoán hoặc xóa locator.
    - Không suy ra quan hệ tương ứng chỉ từ vị trí của các mục trong
      nhiều danh sách song song. Chỉ tách thành từng cặp khi Source text
      xác lập rõ ánh xạ; nếu không, giữ cấu trúc quy tắc chung và không
      tự thêm giới hạn cho từng chủ thể.
    - Giữ chính xác ý nghĩa, con số, khoảng, đơn vị, điều kiện và
      ngoại lệ. Không thêm kiến thức, suy đoán hoặc diễn giải rộng hơn
      source text; không tạo fact nếu chưa đủ căn cứ.
    - Statement phải tự đứng độc lập: không để các tham chiếu phụ thuộc
      ngữ cảnh như "Điều này", "khoản này", "nội dung nêu trên" hoặc
      "hành vi nêu trên". Chỉ ghi số Điều, khoản khi đúng định danh đó
      xuất hiện rõ trong Source text của evidence; không suy ngược từ
      Structural context, số Điều bắt đầu sau Target, thứ tự văn bản hay
      kiến thức ngoài window. Khi source dùng tham chiếu phụ thuộc, thay
      bằng nội dung được Source text trong evidence xác lập trực tiếp.
      Nếu vẫn không thể viết fact tự hiểu được thì không tạo fact đó.
    - Không thêm tiền tố tổng hợp như "Theo Điều N", "Theo khoản N" hoặc
      "Tại Điều N" chỉ để mô tả vị trí cấu trúc của quy tắc. Hãy viết
      trực tiếp chủ thể, hành vi, điều kiện và hậu quả để statement tự đủ
      nghĩa. Chỉ giữ locator khi source dùng nó như một dẫn chiếu pháp lý
      cần thiết để giới hạn hoặc xác định nội dung của chính quy tắc.
    - Một quy tắc hoàn chỉnh trong Target vẫn phải được trích khi chính
      source giới hạn nó bằng dẫn chiếu như "trừ ... điểm X khoản N Điều
      này". Giữ nguyên dẫn chiếu, thay "Điều này" bằng đúng định danh lấy
      từ chunk tiêu đề trong window và dẫn chunk đó. Không đọc, diễn giải
      hoặc dẫn nội dung của điều khoản được tham chiếu nếu nó nằm trong
      FollowingTarget; locator trong Target đã đủ để bảo toàn ngoại lệ.
      Phải quét toàn bộ Preceding, không giả định chunk gần Target nhất
      chứa tiêu đề. Evidence có thể không liền kề; chunk chỉ cung cấp
      tiêu đề và locator vẫn là evidence trực tiếp bắt buộc cho locator.
    - Giữ nguyên mọi giới hạn về chủ thể, loại phương tiện, hoàn cảnh
      áp dụng và dẫn chiếu điều, khoản; không thay dẫn chiếu cụ thể bằng
      cụm chung như "theo quy định", không mở rộng nhãn nhóm so với nguồn.
    - Giữ cả phạm vi làm quy tắc có thể áp dụng đúng, gồm đối tượng,
      địa điểm, tình trạng, hoạt động, loại công trình, biển báo hoặc
      trường hợp cụ thể. Phạm vi phân biệt có thể nằm ở chunk trước nên
      phải đọc và dẫn chunk đó. Khi cùng một câu nguồn lặp lại dưới các
      phạm vi cấu trúc khác nhau, mỗi statement phải nêu rõ phạm vi riêng;
      không xuất câu tác nghiệp đứng một mình nếu câu đó sẽ trở thành fact
      đồng nghĩa với một phạm vi khác hoặc rộng hơn source.
    - Statement phải giữ đầy đủ phạm vi chủ thể được liệt kê trong tiêu
      đề Điều nếu thiếu nó có thể làm quy tắc áp dụng sang chủ thể hoặc
      phương tiện khác. Nếu mục trong Target nêu một loại phương tiện,
      hoạt động hoặc phương thức cụ thể, đặt nó trong phần hành vi nhưng
      vẫn giữ phạm vi chủ thể cấp Điều; không biến riêng cụm đó thành
      chủ thể thay thế.
    - Không bỏ sót quy định giao thẩm quyền hoặc quy định chi tiết nằm
      trọn trong Target khi nó nêu rõ chủ thể và nội dung được giao.
    - Xét riêng từng câu hoàn chỉnh, mục danh sách, hàng bảng và kết
      luận áp dụng trong ví dụ chính thức nằm trong Target. facts=[] hợp
      lệ khi không có nội dung nào vừa thuộc domain, vừa đủ rõ và đủ căn
      cứ để tạo statement độc lập sau khi đọc toàn bộ window.
    - evidence luôn chứa Target nhưng không mặc định chỉ chứa Target.
      Thêm mọi chunk có Source text đã dùng để tạo statement. Phần chỉ
      được khôi phục từ Structural context không cần evidence và Structural
      context không được dùng làm evidence.
    - Kiểm toán evidence sau khi viết từng statement: với từng hành vi,
      chủ thể, phạm vi, điều kiện, ngoại lệ, giá trị, thời hạn, hậu quả,
      số Điều/khoản và nhãn cấu trúc được nêu, phải chỉ ra chunk evidence
      có chính thông tin đó trong Source text. Đặc biệt, nếu Target chỉ có
      mức áp dụng, mục tiếp nối hoặc cụm "Điều này" còn câu dẫn, định danh
      hay phạm vi nằm ở chunk trước thì bắt buộc dẫn cả chunk trước. Chỉ
      trả fact khi hợp evidence trực tiếp phủ toàn bộ statement. Một số
      Điều/khoản xuất hiện ở Source chunk ngoài evidence là bằng chứng
      fact chưa đạt cổng evidence, dù các phần còn lại đã được Target phủ.
    - Kiểm tra từng locator riêng biệt, không kiểm tra chung cả statement:
      với mỗi locator, tìm chunk evidence chứa đúng chuỗi locator trong
      Source text. Chỉ thấy con số riêng lẻ hoặc locator ở chunk ngoài
      evidence đều không đạt. Không chọn một preceding chunk khác thay
      cho chunk thực sự chứa locator.
    - Không được xử lý tham chiếu phụ thuộc bằng cách chỉ xóa dẫn chiếu
      nhưng giữ câu còn lại. Nếu dẫn chiếu xác định trường hợp áp dụng,
      phải giữ định danh có evidence, thay bằng đầy đủ nội dung được
      evidence xác lập, hoặc xóa toàn bộ fact; không xuất câu rộng hơn.
    - Với câu tác nghiệp chung nằm dưới một mục về biển báo, làn đường,
      nhóm phương tiện hoặc phạm vi cấu trúc cụ thể, statement phải gọi
      rõ đúng phạm vi đó. Nếu cùng câu nguồn có thể lặp dưới mục khác,
      câu thiếu tên phạm vi là không hợp lệ và phải bị xóa.
    - Trả đúng một kết quả cho mỗi input, giữ nguyên target_chunk_id;
      facts=[] là hợp lệ.

    Tự kiểm tra: đã xét hết nội dung pháp lý đặc trưng nằm trong Target,
    kể cả phần tiếp nối; không bỏ sót vì nội dung bắt đầu ở chunk trước;
    mỗi câu, vế và hành động độc lập trong Target đã được xét; từng mục
    được ánh xạ sang một fact đủ căn cứ hoặc bị loại vì chưa rõ, đặc biệt
    các nghĩa vụ nằm sau dấu chấm phẩy hoặc từ "và";
    đọc lại Target từ ký tự đầu đến ký tự cuối và đối chiếu từng mệnh đề
    quy phạm với output; không bỏ sót fact đủ rõ, nhưng không tạo fact cho
    hành động phải suy đoán chủ thể, phạm vi, điều kiện hoặc dẫn chiếu;
    với từng fact có FollowingTarget, Target phải kết thúc giữa một câu
    chưa hoàn thành, phần dùng từ FollowingTarget chỉ kết thúc chính câu
    đó, và phần làm fact khác biệt không được chỉ nằm ở FollowingTarget;
    không lấy phần tử mới dù cùng danh sách hoặc quy tắc. Statement đã giữ
    phạm vi chủ thể cấp Điều và không thay nó bằng phương tiện nằm trong
    hành vi; không tự tạo ánh xạ từ thứ tự liệt kê; không gộp quy tắc độc
    lập; không còn tham chiếu phụ thuộc ngữ cảnh hoặc định danh tự đoán;
    mỗi số Điều/khoản và phạm vi nêu trong statement có căn cứ rõ, mọi
    thành phần đã được hợp evidence phủ; statement thuần tiếng Việt và
    tự đủ nghĩa. Quét lại output cuối: xóa mọi fact còn "Điều này",
    "khoản này", số Điều không có nguyên văn trong input, số hoặc dẫn
    chiếu nằm ở Source chunk chưa được liệt kê trong evidence, hay câu
    tác nghiệp không nêu được phạm vi phân biệt. Đây là bộ lọc literal
    cuối cùng, không có ngoại lệ: xóa statement chứa "Điều này", "điều
    này", "khoản này", "nội dung nêu trên" hoặc "hành vi nêu trên".

    Semantic veto cuối cùng, không có ngoại lệ:
    - Nếu câu nguồn giới hạn phạm vi bằng "theo" hoặc "quy định tại"
      rồi dẫn các trường hợp khác, statement phải giữ dẫn chiếu có đủ
      evidence hoặc viết đầy đủ chính các trường hợp đó. Chỉ bỏ phần dẫn
      chiếu và giữ một mô tả chung về đối tượng là làm rộng nguồn: xóa
      fact, không được trả câu kiểu "áp dụng với đối tượng bị xử lý".
    - Nếu câu tác nghiệp như "Khi..." hoặc "Việc..." nằm trong mục có
      tên biển báo, loại làn, nhóm phương tiện hay phạm vi riêng,
      statement phải nêu đúng tên phạm vi đó. Ví dụ, câu nằm trong mục
      của biển R.xxx phải viết "Đối với ... biển R.xxx, khi..."; câu chỉ
      còn "Khi..." là thiếu phạm vi và phải bị xóa.

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

    Output blocker: sau khi đọc xong toàn bộ window và trước khi trả JSON,
    quét lại từng statement. Với mỗi cụm "Điều N", tìm entry có Source
    text chứa nguyên văn "Điều N" và thêm đúng chunk_id của entry đó vào
    evidence. Tiêu đề có thể nằm ở ArticleContext hoặc một Preceding,
    không nhất thiết ở chunk gần Target nhất. Khi Target dùng "Điều này",
    chunk chứa tiêu đề "Điều N." là evidence bắt buộc; chunk Chương/Mục
    hoặc chunk chỉ nhắc Điều N trong dẫn chiếu khác không thể thay thế.
    Không được trả fact nếu evidence cuối cùng chưa chứa đúng chunk này.

    {{ ctx.output_format }}
  "#
}
