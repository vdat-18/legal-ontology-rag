# Data

Raw source documents are stored in:

```text
data/raw/road_traffic_legal_documents.zip
```

The ingestion code can read directly from this archive. Manual extraction is not
required. Document parsing and chunking are handled by Docling.

The source Drive folder is:

<https://drive.google.com/drive/folders/1RhVMqDr98r5IBqArSrzUYBvoNZTjtLxE>

Use `data/config/road_traffic_penalty_scope.json` for the broad
`in_scope`/`out_scope` review of all raw legal documents.

Use `data/config/groundtruth_corpus_v1.json` as the first focused source corpus
for ground-truth candidate generation.
