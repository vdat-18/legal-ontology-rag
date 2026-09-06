$ErrorActionPreference = "Stop"

# Standalone Step 0 (preflight, no LLM) for the 20 remaining in-scope documents.
$docs = @(
    "51_2024_TT-BGTVT_m_633562",
    "158_2024_ND-CP_636875",
    "118_2021_ND-CP_477969",
    "68_2025_ND-CP_648513",
    "39_2024_TT-BGTVT_m_633952",
    "151_2024_ND-CP_m_619564",
    "79_2024_TT-BCA_634265",
    "12_2025_TT-BCA_645271",
    "165_2024_ND-CP_623287",
    "160_2024_ND-CP_m_624017",
    "218_2026_ND-CP_711730",
    "236_2026_ND-CP_712588",
    "38_2024_TT-BGTVT_m_622477",
    "161_2024_ND-CP_623658",
    "13_2025_TT-BCA_m_645302",
    "47_2024_TT-BGTVT_m_623286",
    "184_2025_ND-CP_664032",
    "14_2025_TT-BXD_m_664151",
    "241_2026_ND-CP_712584",
    "67_2023_ND-CP_m_578283"
)

foreach ($doc in $docs) {
    Write-Host "=== $doc ===" -ForegroundColor Cyan
    uv run --extra dev legal-check-ontology-population `
        "data/processed/groundtruth/$doc.chunks.jsonl" `
        "data/processed/groundtruth/$doc.chunk_screening.jsonl" `
        "data/processed/groundtruth/$doc.document_metadata.jsonl" `
        ontology/population_contract.yaml `
        "data/ontology/preflight/$doc.full-v0.2.step1.json"
}
