class ChunkScreeningItem {
  chunk_id string
  source_name string @description(#"Tên tài liệu nguồn chứa target."#)
  record_scope string @description(#"Phạm vi root hoặc attachment_child của target."#)
  attachment_member_path string? @description(#"Đường dẫn thành viên đính kèm nếu target thuộc attachment."#)
  kind string @description(#"Loại chunk do parser xác định."#)
  context_text string @description(#"
    Ngữ cảnh cấu trúc do parser tạo; không phải source text.
  "#)
  text string
}

class ChunkScreeningResult {
  chunk_id string @description(#"Phải trùng với một chunk_id trong danh sách đầu vào."#)
  has_penalty_content bool @description(#"
    True nếu target có fact trực tiếp giúp xác định hoặc xử lý vi phạm trong domain.
  "#)
  has_document_metadata bool @description(#"
    True nếu target giúp xác định số hiệu hoặc hiệu lực của chính văn bản đang đọc.
  "#)
  reason string @description(#"Một câu tiếng Việt ngắn giải thích quyết định."#)
}

function ScreenChunks(chunks: ChunkScreeningItem[]) -> ChunkScreeningResult[] {
  client OpenAICompatibleProvider
  prompt #"
    {{ _.role("system") }}

    Bối cảnh: chuẩn bị nguồn sự thật cho benchmark RAG hỏi đáp về
    xử lý vi phạm giao thông đường bộ tại Việt Nam.

    Nhiệm vụ: sàng lọc độc lập từng target chunk trước khi trích legal fact.
    Ưu tiên độ chính xác: chỉ giữ quy tắc pháp lý có liên hệ trực tiếp, rõ
    ràng với việc xác định hoặc xử lý vi phạm giao thông đường bộ.

    Mỗi block là một target độc lập. Khi đánh giá một block, bỏ qua hoàn toàn
    text và fact của mọi block khác trong cùng request. Không dùng thứ tự các
    block để nối, bổ sung hoặc hoàn thiện nội dung target. Fact chỉ xuất hiện
    ở block khác không được dùng làm căn cứ cho kết quả hiện tại.

    Cổng quyết định có ưu tiên cao:
    - chỉ đặt true cho quy tắc trực tiếp áp dụng vào một vụ việc cụ thể; câu
      giao nhiệm vụ quản lý chung không phải thẩm quyền xử lý vụ việc;
    - đặt false nếu Target chỉ giao cơ quan quản lý, hướng dẫn, kiểm tra, thanh
      tra, phối hợp, báo cáo, bảo vệ công trình hoặc "xử lý vi phạm theo thẩm
      quyền" mà không tự nêu hành vi, điều kiện, quyền hạn hay bước xử lý cụ
      thể. Viện dẫn điều/khoản khác nhưng không nêu lại nội dung cũng không đủ;
    - đặt false cho điều khoản chỉ giao cơ quan ban hành, sửa đổi hoặc quy định
      danh mục, loại đối tượng, điều kiện hay yêu cầu trong tương lai. Đây là
      thẩm quyền xây dựng quy định, không phải chính quy tắc dùng để xác định vi
      phạm; chỉ đặt true nếu Target tự nêu luôn đối tượng và nghĩa vụ cụ thể;
    - phạm vi điều chỉnh và định nghĩa đứng riêng luôn false, kể cả định nghĩa
      về vi phạm hành chính. Chỉ giữ khi cùng Target còn có một quy tắc trực
      tiếp sử dụng khái niệm đó để xác định hoặc xử lý trường hợp cụ thể.

    Phân biệt rõ thẩm quyền vụ việc với quản lý nhà nước. Chỉ coi là thẩm quyền
    trực tiếp khi Target tự quy định chủ thể được lập biên bản, ra quyết định,
    áp dụng một chế tài/biện pháp, thu thập chứng cứ hoặc thực hiện một bước xác
    định trong hồ sơ vi phạm. Các cụm như "kiểm tra, xử lý theo thẩm quyền" hay
    "thực hiện biện pháp phòng ngừa, ngăn chặn" tự chúng chỉ là nhiệm vụ chung.

    Ngược lại, nghĩa vụ, thời hạn, hạn chế hoặc hậu quả mà chính Target xác định
    phát sinh sau xử phạt, đình chỉ, thu hồi, tạm giữ hay biện pháp xử lý phải
    đặt true. Target không cần nhắc lại hành vi gốc nếu tự nêu rõ sự kiện xử lý
    làm căn cứ phát sinh; phần hành chính đi kèm không làm mất fact hợp lệ.

    Đặt has_penalty_content=true chỉ khi Target text tự nêu ít nhất một nội
    dung có thể dùng trực tiếp trong một vụ việc vi phạm đường bộ:
    - hành vi bị cấm hoặc quy tắc, nghĩa vụ, điều kiện, ngoại lệ trực tiếp
      quyết định hành vi có vi phạm hay không;
    - chủ thể, phương tiện hoặc phạm vi áp dụng của quy tắc đó;
    - chế tài, hậu quả pháp lý hoặc biện pháp khắc phục hậu quả;
    - phát hiện, chứng minh, ngăn chặn, thẩm quyền, trình tự, thời hạn, quyền
      hoặc nghĩa vụ trong quá trình xử lý;
    - hiệu lực hoặc chuyển tiếp trực tiếp thay đổi cách áp dụng các nội dung
      trên.

    Kiểm tra tối thiểu trước khi đặt true: từ riêng Target phải trả lời được
    một câu hỏi pháp lý cụ thể như ai đã làm/không làm gì, trong điều kiện nào,
    có bị cấm, bị xử lý hoặc phải áp dụng thủ tục nào. Nếu chỉ có thể nói chung
    rằng nội dung "liên quan giao thông", "bảo đảm an toàn", "phục vụ quản lý"
    hoặc "có thể dẫn đến vi phạm" thì đặt false. Không tự bổ sung chế tài, hành
    vi vi phạm hoặc liên hệ với vụ việc từ kiến thức ngoài Target.

    Quy định có thể áp dụng cho người tham gia giao thông, chủ xe, đơn vị kinh
    doanh vận tải, cơ sở đào tạo/sát hạch hoặc đơn vị quản lý hạ tầng. Với chủ
    thể quản lý, kinh doanh, đào tạo, xây dựng hoặc bảo trì, chỉ giữ khi Target
    tự nêu hành vi bị cấm/không được phép, căn cứ đình chỉ/thu hồi/xử lý, hậu
    quả pháp lý hoặc thủ tục xử lý tương ứng. Nghĩa vụ vận hành, điều kiện kỹ
    thuật, yêu cầu bảo đảm an toàn, trách nhiệm quản lý hoặc từ "phải",
    "không được ảnh hưởng" tự chúng chưa phải fact vi phạm.

    Đặt false nếu Target chỉ mô tả hoạt động quản lý, điều kiện vận hành hoặc
    kiến thức chuyên ngành mà chưa tạo liên hệ pháp lý trực tiếp với một vi
    phạm đường bộ. Đánh giá chức năng pháp lý của toàn Target, không quyết định
    chỉ vì xuất hiện từ như "vi phạm", "cấm", "bắt buộc", "xử lý", "lấn làn"
    hay "vượt đèn". Trong Target hỗn hợp, chỉ đặt true nếu phần thuộc domain tự
    tạo thành một quy tắc trực tiếp và hoàn chỉnh; việc nhắc thoáng qua giao
    thông đường bộ giữa danh sách nhiều lĩnh vực hoặc nhiều ví dụ không đủ.

    Các tiêu chí loại trừ bên dưới có ưu tiên cao hơn tín hiệu dương đơn lẻ.
    Không cứu một Target chỉ vì tìm thấy một câu có từ "phải", "không được",
    "xử lý vi phạm", tên cơ quan giao thông hoặc một quy tắc hợp lệ nằm trong
    loại source phải loại.

    Luôn đặt false cho:
    - chương trình, thời lượng, nội dung đào tạo; lỗi, điểm, đạt/trượt hoặc quy
      trình thi sát hạch; tình huống giả định, mô phỏng hay ví dụ sư phạm, kể
      cả khi tình huống mô tả một hành vi có vẻ vi phạm;
    - nghĩa vụ tổ chức, báo cáo, lưu dữ liệu hoặc vận hành thông thường của cơ
      sở đào tạo/sát hạch nếu Target không tự nêu căn cứ thu hồi, đình chỉ, chế
      tài hoặc biện pháp xử lý do vi phạm nghĩa vụ đó;
    - hình ảnh, chú thích hình, OCR/mô tả hình, bản vẽ, sơ đồ, bảng kích thước
      hoặc mô tả thuần kỹ thuật về vật liệu, màu sắc, kết cấu, chế tạo, lắp đặt
      báo hiệu hay phương tiện. Một đoạn có cả quy cách và ý nghĩa chỉ được giữ
      khi phần chữ tự nêu rõ hiệu lực, điều cấm hoặc nghĩa vụ chấp hành của
      người tham gia giao thông; hình dạng/kích thước tự nó không tạo nghĩa vụ;
    - quy trình ứng phó sự cố, cứu nạn, tạm dừng khai thác, điều tiết giao
      thông, bảo trì cây xanh/công trình hoặc yêu cầu vận hành an toàn thông
      thường nếu không tự nêu hành vi vi phạm và cách xử lý hành vi đó;
    - phí, hợp đồng, hồ sơ, giám định, tạm ứng hoặc thủ tục bồi thường bảo hiểm,
      bảng tỷ lệ thương tật và nghĩa vụ nội bộ của doanh nghiệp bảo hiểm, kể cả
      khi phát sinh từ tai nạn hoặc nhắc đến rượu, bia, ma túy. Chỉ giữ nghĩa vụ
      phải có bảo hiểm bắt buộc để tham gia giao thông, hành vi không có bảo
      hiểm và việc cơ quan có thẩm quyền kiểm tra, xử lý hành vi đó;
    - nội dung riêng của đường sắt, đường thủy, hàng hải, hàng không, ma túy,
      cai nghiện hoặc lĩnh vực khác ngoài giao thông đường bộ;
    - chính sách, tuyên truyền, tổ chức quản lý, dữ liệu/thống kê, tài chính,
      hỗ trợ hoặc thủ tục hành chính độc lập.

    Quy tắc xử lý vi phạm hành chính chung phải được giữ khi Target tự nêu một
    quy tắc độc lập có thể áp dụng nguyên vẹn cho vụ việc giao thông đường bộ,
    không phụ thuộc vào nội dung riêng của lĩnh vực khác. Ví dụ gồm cách xác
    định mức phạt trong khung, tình tiết tăng/giảm, thời hiệu, lập biên bản,
    tạm giữ và thi hành quyết định. Không yêu cầu Target phải nhắc lại từ
    "đường bộ" nếu bản thân quy tắc trung lập lĩnh vực và áp dụng trực tiếp.
    Danh sách chức danh,
    cơ quan, đối tượng hoặc hành vi pha trộn nhiều lĩnh vực đặt false nếu phần
    đường bộ chỉ là một mục rời rạc và Target không tự xác lập thẩm quyền hay
    cách xử lý riêng cho vi phạm đường bộ. Một chức danh đường bộ hoặc Cảnh sát
    giao thông xuất hiện giữa danh sách nhiều chức danh chuyên ngành không làm
    cả Target thành fact thuộc domain. Các biện pháp giáo dục tại xã,
    trường giáo dưỡng, cai nghiện và biện pháp hành chính đặc biệt đặt false,
    kể cả khi danh sách ví dụ có nhắc một hành vi giao thông. Việc chuyển hồ sơ
    có dấu hiệu tội phạm trong chính quy trình áp dụng các biện pháp đặc biệt
    này vẫn thuộc phần phải loại, không phải thủ tục xử phạt chung. Định
    nghĩa/phân loại chỉ được giữ khi Target đồng thời chứa quy tắc trực tiếp sử
    dụng nó.

    Quy tắc loại biện pháp hành chính đặc biệt áp dụng cho toàn bộ Target: thời
    hạn, thẩm quyền, gửi quyết định, giám sát hoặc quyền của người bị áp dụng
    biện pháp giáo dục tại cộng đồng, trường giáo dưỡng hay cơ sở cai nghiện
    đều penalty=false. Không giữ chúng chỉ vì đây là một dạng "xử lý vi phạm
    hành chính" hoặc vì Target có trình tự pháp lý cụ thể.

    Mục lục, tiêu đề, lời dẫn, căn cứ ban hành, danh mục hoặc tên biểu mẫu và
    thông tin hành chính tự chúng không đủ. Luôn đặt false cho mẫu quyết định,
    biên bản, thông báo, phiếu, sổ theo dõi hoặc giấy phép chưa được lập; phần
    chú thích/hướng dẫn điền mẫu; trường trống, dòng giao/nhận và ô ký tên.
    Ghi chú, căn cứ pháp lý hoặc câu chữ nhìn có vẻ hoàn chỉnh nhưng vẫn nằm
    trong cấu trúc của văn bản sẽ được lập theo mẫu là boilerplate và phải đặt
    false. Nếu Target còn các mục để trống, dấu chấm dẫn, ô điền, phần ký hoặc
    hướng dẫn lập hồ sơ thì toàn Target vẫn là mẫu; một câu hướng dẫn hoàn
    chỉnh nằm trước các trường trống không làm Target hợp lệ. Chỉ phần thuyết
    minh độc lập bên ngoài mẫu, trực tiếp giải thích một quy tắc pháp luật áp
    dụng chung, mới có thể được giữ.

    Dùng Source provenance để hiểu target thuộc văn bản gốc hay tệp đính kèm,
    nhưng không loại chỉ dựa vào provenance: phụ lục vẫn có thể chứa quy tắc
    pháp lý độc lập. Ghi chú hoặc hướng dẫn đơn lẻ của biểu mẫu/tệp đính kèm
    phải đặt false kể cả khi target không chứa trường trống hay ô ký tên.
    Phải xác định chức năng của Target trước khi đọc tín hiệu mệnh lệnh. Nếu
    Target là nội dung được in trực tiếp trên giấy phép, chứng chỉ, thẻ hoặc mặt
    sau của giấy tờ để nhắc người giữ giấy thì đó là hướng dẫn của mẫu giấy tờ
    và bắt buộc đặt penalty=false. Quy tắc loại này ưu tiên hơn các câu như
    phải xuất trình, không được sửa chữa, cấm tẩy xóa hoặc phải cấp lại. Việc
    target nằm trong source root, không có trường trống hoặc dùng câu hoàn
    chỉnh không biến hướng dẫn in trên giấy tờ thành quy định độc lập. Chỉ giữ
    quy định nằm ngoài mẫu, tự xác lập phạm vi áp dụng chung bằng văn bản quy
    phạm thay vì nói về giấy tờ cụ thể đang được in.

    Luôn đặt false cho tên hoặc mục danh mục biển báo, vạch kẻ, cấu kiện hay
    thiết kế chỉ nêu mã và tên như "Biển số ...: Cấm ...". Chỉ đặt true khi
    chính Target còn có câu giải thích hiệu lực pháp lý, điều cấm hoặc nghĩa vụ
    chấp hành. Nội dung phân loại phương tiện, cấu tạo, tiêu chuẩn kỹ thuật hoặc
    yêu cầu thiết kế cũng đặt false nếu không tự nêu vi phạm hay cách xử lý.

    Việc thu thập, kết nối, chia sẻ, lưu trữ hoặc quản lý dữ liệu chỉ để "phục
    vụ xử lý vi phạm" không phải bản thân quy tắc vi phạm hay thủ tục xử lý,
    nên đặt false nếu Target không nêu thêm hành vi, thẩm quyền, trình tự hoặc
    hậu quả pháp lý cụ thể. Hành vi, lỗi và quy tắc trong kỳ thi/sát hạch chỉ
    dùng để chấm đạt/trượt, không phải vi phạm xảy ra trên đường, nên đặt false.

    Thông báo, thống kê, phối hợp hoặc phân công giải quyết tai nạn giao thông
    tự nó không phải xử lý vi phạm. Chỉ giữ khi Target tự nêu hoạt động phát
    hiện, chứng minh hoặc xử lý một hành vi vi phạm cụ thể phát sinh từ vụ tai
    nạn. Nghĩa vụ nội bộ của doanh nghiệp bảo hiểm, hiệp hội hoặc cơ quan quản
    lý bảo hiểm, gồm báo cáo, công khai, khuyến mại và chiết khấu, luôn đặt
    false; chúng không trở thành fact đường bộ chỉ vì liên quan bảo hiểm xe.

    Với từng Target, quyết định has_document_metadata trước rồi mới quyết định
    has_penalty_content; hai field hoàn toàn độc lập và có thể khác nhau. Đặt
    metadata=true khi chính Target chứa ít nhất một thông tin giúp Step 1B xác
    định metadata của toàn văn bản đang đọc:
    - số hiệu chính thức tại phần nhận diện của văn bản nguồn;
    - ngày bắt đầu hoặc hết hiệu lực của toàn văn bản;
    - điều khoản hiệu lực tương đối của toàn văn bản như "có hiệu lực kể từ
      ngày ký", dù ngày ký nằm ở metadata candidate khác. Step 1B sẽ kết hợp
      các candidate để suy ra giá trị cuối cùng.

    Khi quyết định metadata, phải đọc toàn bộ Target từ đầu đến cuối và chủ
    động tìm điều khoản hiệu lực của văn bản nguồn, kể cả khi nó nằm sau một
    đoạn dài sửa đổi, chuyển tiếp hoặc nội dung ngoài domain. Target có thể
    metadata=true nhưng penalty=false; không dùng quyết định penalty để bỏ qua
    việc kiểm tra metadata.

    Trước khi trả metadata=false, kiểm tra lại riêng phần cuối Target. Nếu
    Target tự chứa điều khoản tuyên bố ngày bắt đầu/hết hiệu lực của chính văn
    bản nguồn thì bắt buộc metadata=true, bất kể phần trước dài, hỗn hợp, ngoài
    domain hoặc penalty được quyết định true/false thế nào.

    Phần đầu văn bản nêu số hiệu chính thức là đủ, dù chỉ có thông tin hành
    chính, ngày ký/ban hành và penalty=false. Điều khoản hiệu lực của chính
    văn bản vẫn phải đặt metadata=true khi nằm ở cuối một Target hỗn hợp chứa
    nhiều sửa đổi hoặc nội dung thuộc domain khác.

    Căn cứ cho metadata=true phải xuất hiện trong chính Target text. Không gán
    metadata từ block đứng trước, đứng sau hoặc cùng request; nếu Target chỉ ở
    gần một câu hiệu lực nhưng không tự chứa thông tin đó thì đặt false. Không
    chọn riêng ngày ký/ban hành nếu Target không phải phần nhận diện chính thức;
    ngày áp dụng riêng cho một điều/khoản; metadata của văn bản được dẫn chiếu;
    hoặc số hiệu riêng của quy chuẩn, phụ lục, biểu mẫu hay tài liệu đính kèm.
    Nếu Target đồng thời chứa ngày áp dụng riêng và ngày hiệu lực toàn văn bản,
    metadata=true vì có thông tin hợp lệ về toàn văn bản.

    Phải xác định đúng văn bản mà thông tin metadata đang nói tới. Nếu Target
    thuộc phụ lục, mẫu biên bản, mẫu quyết định hoặc mẫu văn bản hành chính,
    câu như "Quyết định này có hiệu lực kể từ ngày ký" chỉ nói về văn bản sẽ
    được lập theo mẫu, không phải văn bản nguồn đang đọc, nên metadata=false.
    Tương tự, số hiệu văn bản nguồn xuất hiện như căn cứ hoặc dẫn chiếu bên
    trong mẫu không phải metadata của Target. Chỉ chọn số hiệu ở phần nhận diện
    chính thức của văn bản nguồn và điều khoản hiệu lực thực sự điều chỉnh toàn
    văn bản nguồn. Luôn đặt false khi số hiệu chỉ được lặp lại trong tiêu đề
    hoặc dòng giới thiệu phụ lục.

    Với source là văn bản hợp nhất, nội dung thân văn bản vẫn dùng các cụm như
    "Luật này" hoặc "Nghị định này" để nói về văn bản quy phạm được hợp nhất.
    Ngày hiệu lực, hết hiệu lực và điều khoản thi hành trong phần thân đó luôn
    thuộc văn bản gốc, không thuộc văn bản hợp nhất, nên metadata=false. Chỉ
    đặt metadata=true khi chính Target nhận diện rõ "Văn bản hợp nhất", số
    hiệu VBHN hoặc tuyên bố hiệu lực độc lập của chính văn bản hợp nhất. Không
    suy ra hiệu lực VBHN từ hiệu lực của văn bản gốc, dù điều khoản có tiêu đề
    "Hiệu lực thi hành" và nằm trong chính văn bản đang đọc.

    Chỉ Target text làm căn cứ. Structural context chỉ giúp hiểu cấu trúc,
    không tự tạo nội dung pháp lý. Trả đúng một kết quả cho mỗi input, giữ
    nguyên chính xác chunk_id, không tạo hoặc sửa bất kỳ ký tự nào. Reason
    phải là một câu ngắn, hoàn toàn bằng tiếng Việt và không xen từ hay chữ của
    ngôn ngữ khác. Khi hai field khác nhau, reason phải thể hiện căn cứ của cả
    quyết định metadata và penalty.

    Trước khi trả kết quả, tự kiểm tra riêng từng item: đọc lại đúng Target gắn
    với chunk_id đó; xác nhận mọi chi tiết trong reason thực sự xuất hiện trong
    chính Target; xác nhận không lấy câu, ngày hoặc kết luận từ item liền trước
    hay liền sau. Nếu reason mô tả nội dung không có trong Target thì phải đánh
    giá lại item đó. Sau cùng kiểm tra mỗi input có đúng một output và reason
    mô tả đúng chính chunk_id tương ứng.

    {{ _.role("user") }}

    Các chunk cần sàng lọc:
    {% for chunk in chunks %}
    ---
    chunk_id: {{ chunk.chunk_id }}
    Source provenance:
    - source_name: {{ chunk.source_name }}
    - record_scope: {{ chunk.record_scope }}
    - attachment_member_path: {{ chunk.attachment_member_path }}
    - kind: {{ chunk.kind }}
    {% if chunk.context_text %}
    Structural context:
    {{ chunk.context_text }}
    {% endif %}
    Target text:
    {{ chunk.text }}
    ---
    {% endfor %}

    {{ ctx.output_format }}
  "#
}
