# Road Traffic Legal Ontology

Thư mục này chứa TBox, SHACL, CQ và cấu hình cho ontology xử lý vi phạm giao thông đường bộ Việt Nam. Retrieval,
GraphRAG và Graphiti không thuộc phạm vi của ontology flow.

## Core flow

```text
requirements + competency questions
  -> glossary + reuse decisions
  -> OWL modules + SHACL + CQ queries
  -> chunk population
  -> within-document assembly
  -> cross-document legal changes
  -> legal exceptions
  -> concept resolution
  -> RDF graph
```

Sau population, mỗi bước semantic dùng một quy trình chung:

```text
bounded structural candidates
  -> typed LLM proposal
  -> LLM component review before canonicalization
  -> minimal ID/reference integrity
  -> deterministic RDF materialization
```

Không có review artifact hoặc semantic validator trong runtime. LLM quyết định nội dung pháp lý; code chỉ giữ typed
parsing, candidate/reference membership, identity cần cho persistence và RDF serialization. SHACL, CQ và ROBOT/HermiT
là QA tách rời, không chặn population hay các bước canonicalization. LLM không trực tiếp sửa TBox.

Runtime được chạy bằng sáu CLI cấp cao: conduct inventory, population binding, assembly, legal change, exception và
concept resolution. Mỗi CLI tự chạy phần candidate/finalization không dùng LLM và giữ checkpoint để kiểm tra hoặc
resume. Riêng assembly gọi thêm LLM để review từng proposal component và loại over-merge trước khi canonical hóa.

## Minimal provenance

Canonical RDF chỉ giữ hai liên kết provenance cho assertion và rule component:

```text
groundedIn -> LegalProvision
sourcedFromChunk -> SourceChunk
```

Text nguồn có thể tồn tại trong JSON trung gian để LLM grounding và resume, nhưng RDF không tạo `EvidenceSpan`,
`exactText`, `containsSpan` hoặc `supportedBy`.

## Main artifacts

- `requirements/ontology_requirements.yaml`: mục tiêu, use cases, scope và granularity.
- `requirements/competency_questions.yaml`: 15 CQ bắt buộc định hướng mô hình.
- `glossary/terms.yaml`: các thuật ngữ domain thực sự được ontology dùng.
- `external_reuse_registry.yaml`: bốn vocabulary ngoài được sử dụng: Dublin Core, PROV-O, SKOS và OWL-Time.
- `modules/*.ttl`: các OWL module source, occurrence, temporal, rule và consequence.
- `shapes/legal_ontology.shacl.ttl`: closed-world data-quality constraints.
- `tests/cq_queries/*.rq`: projection kiểm tra mọi CQ bắt buộc.
- `*_contract.yaml`: typed runtime limits cho population và bốn bước semantic sau đó.

## QA tùy chọn

`tests/ontology/` kiểm tra Turtle, imports, SHACL fixtures, CQ projections và traceability. Các suite
`tests/ontology_population`, `tests/ontology_assembly`, `tests/ontology_change`, `tests/ontology_exception` và
`tests/ontology_concept` kiểm tra toàn bộ instance construction flow.

```powershell
uv run --extra dev pytest `
  tests/ontology `
  tests/ontology_population `
  tests/ontology_assembly `
  tests/ontology_change `
  tests/ontology_exception `
  tests/ontology_concept
```

SHACL/CQ regression, ROBOT/HermiT semantic QA, release packaging và W3ID publication là các công cụ tùy chọn. Chúng
không nằm trong runtime instance-construction flow và chỉ cần chạy khi kiểm tra thiết kế hoặc chuẩn bị release.

Xem `docs/ontology_population_pipeline.md` để biết lệnh chạy và dependency giữa các bước.
