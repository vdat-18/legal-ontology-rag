class MultiTurnHistoryInput {
  turn_number int
  question string
  answer string
}

class MultiTurnTargetInput {
  turn_number int
  reference_turn_number int
  reference_question string
  reference_answer string
  reference_facts FactContext[]
  facts FactContext[]
  dialogue_act string
  context_dependency string
  answerability string
}

class MultiTurnTargetGenerationInput {
  task_id string
  persona_description string
  history MultiTurnHistoryInput[]
  target MultiTurnTargetInput
}

class GeneratedMultiTurnTarget {
  turn_number int
  query string @description(#"Cau hoi noi tiep phu thuoc vao cac luot trao doi truoc."#)
  standalone_query string @description(#"Cau hoi day du sau khi khoi phuc thong tin bi luoc bo."#)
  reference_answer string @description(#"Dap an tham chieu dua tren boi canh cu va cac can cu moi."#)
  reasoning string @description(#"Giai thich ngan cach cac can cu moi dan toi dap an."#)
  missing_information string? @description(#"Chi tiet con thieu khi chi co the tra loi mot phan; null khi du can cu."#)
  required_claims ClaimReference[] @description(#"Cac y bat buoc moi, tro dung ma can cu cua luot cuoi."#)
}

class GeneratedMultiTurnTargetBatchItem {
  task_id string
  target GeneratedMultiTurnTarget
}

function GenerateMultiTurnTargetBatch(
  items: MultiTurnTargetGenerationInput[]
) -> GeneratedMultiTurnTargetBatchItem[] {
  client OpenAICompatibleProvider
  prompt #"
    {{ _.role("system") }}

    Tao luot cuoi cho ground truth danh gia RAG hoi thoai trong linh vuc duoc the
    hien boi tai lieu dau vao. Cac luot trao doi truoc da duoc duyet va phai giu
    nguyen. Moi yeu cau chi sinh luot cuoi can danh gia.

    Muc tieu danh gia:
    - Cau hoi phai thuc su can luot trao doi duoc chi dinh de khoi phuc it nhat mot
      chi tiet thiet yeu nhu chu the, hanh vi, doi tuong hoac tinh huong. Chi them
      "vay", "con" hay "quay lai" vao mot cau hoi da day du khong tao ra phu
      thuoc hoi thoai.
    - Cac luot truoc chi cung cap boi canh. Thong tin moi de tra loi trong tam nam
      trong cac can cu cua luot cuoi, vi vay he thong van phai truy xuat tai lieu.
    - Khong hoi lai dieu da duoc tra loi. Khong dua dap an vao cau hoi.
    - Luot bat buoc tham chieu duoc nhac lai rieng trong moi yeu cau. Cau hoi
      phai noi tiep dung luot nay, khong duoc bam nham vao luot gan nhat.

    Can cu:
    - Cac can cu cua luot cuoi la nguon duy nhat cho moi ket luan moi. Khong them
      kien thuc, con so, dieu kien, ngoai le, moc thoi gian hoac dan chieu khong co
      trong cac can cu nay.
    - Co the nhac lai thong tin cu de cau tra loi tu nhien, nhung danh sach y bat
      buoc chi ghi cac ket luan moi duoc ho tro boi can cu cua luot cuoi.
    - Giu day du cac tu han dinh ve chu the, hanh vi, doi tuong, muc dich, cach
      thuc, thoi diem va ngoai le.
    - valid_to null chi co nghia dau vao khong neu ngay ket thuc; khong duoc tu
      khang dinh quy dinh van con hieu luc.
    - Neu noi dung cu va can cu moi noi khac nhau ve cung mot van de do khac moc
      hieu luc, cau hoi phai noi ro dang hoi tai moc nao va dap an phai noi
      ro su thay doi. Khong duoc am tham tra loi trai voi noi dung cu ma bo qua moc
      bat dau hieu luc cua can cu moi.
    - Cac can cu cua luot truoc duoc dua kem de doi chieu thoi gian. Chung chi
      giai thich noi dung cu, khong duoc dua vao danh sach y bat buoc cua luot cuoi. Neu
      cau tra loi cu gan mot moc thoi gian khong duoc chinh cac can cu cu ho tro,
      khong lap lai moc do nhu mot ket luan moi.

    Cau hoi:
    - Viet tieng Viet tu nhien theo persona. Cac truong van ban chi chua noi dung
      danh cho nguoi dung; khong nhac ten truong du lieu, nhan nguon, ma can cu
      trong dau ngoac vuong, hay cach he thong noi bo to chuc dau vao. Khong dung
      cac nhan ky thuat bang tieng Anh nhu Query, History, Fact, Target hay Claim.
    - continuity hoi mot khia canh moi cung chu de; deepening hoi sau hon dua tren
      chi tiet trong cac luot truoc; return quay lai chu de cu sau cac luot xen giua.
      Voi return, cau hoi phai noi ro la quay lai va nhac mot cum tu dac trung cua
      luot duoc tham chieu; khong chi dung "do" hoac "truong hop do" neu co the
      bi hieu la luot gan nhat.
    - anaphora dung dai tu nhu "vay", "truong hop do", "muc nay". Ellipsis luoc
      phan da ro, vi du "Con xe may thi sao?".
    - Moi chu the, loai giay phep, loai phuong tien, moc thoi gian hoac dieu kien
      moi chi xuat hien trong can cu moi ma khong co trong cac luot truoc phai duoc
      noi ro ngay trong cau hoi; khong duoc am tham them no vao cau hoi doc lap.
    - Cau hoi doc lap chi khoi phuc phan da luoc bo bang noi dung co that trong cac
      luot truoc. Hai cau phai cung mot y dinh va cung pham vi; cau hoi doc lap
      khong duoc them chu the, dieu kien hay pham vi ma cau hoi va cac luot truoc khong co.
    - Khong hoi theo kieu bao quat nhu "tat ca truong hop" hay "nhung truong hop
      nao" neu cac can cu moi chi la mot phan danh sach. Khi do phai goi ten chinh
      xac cac truong hop dang duoc hoi.

    Muc do co the tra loi:
    - Khi du can cu: tra loi tron trong tam.
    - Khi chi co the tra loi mot phan: tra loi phan co can cu va noi ro dung mot chi tiet
      ma chinh cau hoi dang yeu cau nhung cac can cu chua xac dinh; khong tu them
      mot van de ngoai cau hoi chi de tao phan thieu.
    - Truong thong tin con thieu phai de trong khi du can cu. Khi chi co the tra loi
      mot phan, truong nay phai neu ngan gon dung mot chi tiet con thieu va chi tiet do phai
      duoc hoi truc tiep trong ca cau hoi phu thuoc va cau hoi doc lap. Neu can tinh
      mot ket qua cu the, cau hoi phai hoi ro ket qua cu the do va phan thong tin
      con thieu phai liet ke dau vao con thieu thay vi lap lai ten ket qua can tinh.
    - Moi yeu cau phai co it nhat mot y bat buoc. Moi y tro dung ma can cu cua luot
      cuoi va chi chon tap can cu nho nhat du tra loi. Moi ma can cu chi duoc dung
      cho mot y; neu co nhieu y cung mot can cu thi gop thanh mot y.

    Do dai mem:
    - Cau hoi thuong khong qua 25 tu.
    - Dap an tham chieu thuong khong qua 180 tu.
    - Giai thich thuong khong qua 80 tu.

    Tra toi da mot ket qua cho moi yeu cau. Giu nguyen task_id. Khong tron noi dung
    hoac ma can cu giua cac yeu cau. Neu khong sinh duoc mot yeu cau, bo qua no thay
    vi tao ket qua sai.

    Truoc khi tra ket qua, tu kiem tra:
    - Neu bo cac luot truoc ma cau hoi van day du chu the va tinh huong de hieu,
      viet lai cau hoi de no phu thuoc that su vao luot tham chieu.
    - Neu cau hoi doc lap co chi tiet khong tim thay trong cau hoi hoac cac luot
      truoc, dua chi tiet moi do vao cau hoi hoac bo no khoi ca hai cau.
    - Neu cau hoi co ve yeu cau danh sach day du, thu hep no dung vao cac truong
      hop ma cac can cu moi thuc su bao phu.

    {{ _.role("user") }}

    Cac target can sinh:
    {% for item in items %}
    ===== TARGET {{ item.task_id }} =====
    Mo ta nguoi hoi: {{ item.persona_description }}
    Cac luot truoc:
    {% for turn in item.history %}
    Luot {{ turn.turn_number }}
    Nguoi hoi: {{ turn.question }}
    Tra loi: {{ turn.answer }}
    {% endfor %}

    Luot cuoi can sinh: {{ item.target.turn_number }}
    Luot truoc can tham chieu: {{ item.target.reference_turn_number }}
    Cau hoi cua luot can tham chieu: {{ item.target.reference_question }}
    Cau tra loi cua luot can tham chieu: {{ item.target.reference_answer }}
    Cac can cu cua luot can tham chieu:
    {% for fact in item.target.reference_facts %}
    [{{ fact.fact_id }}] {{ fact.statement }}
    Van ban: {{ fact.document_number }}
    Hieu luc: {{ fact.valid_from }} den {{ fact.valid_to }}
    {% endfor %}
    Cach noi tiep: {{ item.target.dialogue_act }}
    Kieu phu thuoc ngu canh: {{ item.target.context_dependency }}
    Muc do co the tra loi: {{ item.target.answerability }}
    Cac can cu moi can truy xuat:
    {% for fact in item.target.facts %}
    [{{ fact.fact_id }}] {{ fact.statement }}
    Van ban: {{ fact.document_number }}
    Hieu luc: {{ fact.valid_from }} den {{ fact.valid_to }}
    {% endfor %}
    ===== END TARGET {{ item.task_id }} =====
    {% endfor %}

    {{ ctx.output_format }}
  "#
}
