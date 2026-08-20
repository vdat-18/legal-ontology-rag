"""Tests for the versioned ontology population contract."""

from __future__ import annotations

import tomllib
from pathlib import Path

from legal_rag.ontology_population.config import PopulationSettings
from legal_rag.ontology_validation.loader import load_yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = PROJECT_ROOT / "ontology" / "population_contract.yaml"
PROMPT_PATH = PROJECT_ROOT / "baml_src" / "ontology_population.baml"


def test_population_contract_loads_typed_execution_limits() -> None:
    """The committed population contract provides bounded execution settings."""
    settings = PopulationSettings.from_path(CONTRACT_PATH)

    assert settings.instance_namespace == "https://w3id.org/legal-ontology-rag/data/"
    assert settings.batch_size == 1
    assert settings.max_preceding_chunks == 2
    assert settings.max_following_chunks == 1
    assert settings.max_article_context_chunks == 32
    assert settings.max_output_tokens == 32768


def test_population_contract_excludes_unsupported_semantic_steps() -> None:
    """Source population does not silently create occurrences or cross-document identity."""
    contract = load_yaml(CONTRACT_PATH)
    excluded = {item["id"] for item in contract["population_scope"]["excluded"]}

    assert excluded == {"occurrence_context", "cross_document_legal_change", "concept_equivalence"}


def test_population_contract_excludes_general_provisions_from_penalty_rules() -> None:
    """General catalogs and procedures cannot become populated PenaltyRule instances."""
    contract = load_yaml(CONTRACT_PATH)

    assert any("General sanction catalogs" in boundary for boundary in contract["semantic_boundaries"])


def test_population_contract_requires_atomic_target_conducts_and_canonical_paths() -> None:
    """Population records stay queryable at conduct level with stable legal locators."""
    contract = load_yaml(CONTRACT_PATH)
    policy = contract["extraction_policy"]

    assert policy["conduct_granularity"] == "atomic_independently_matchable"
    assert policy["requires_target_conduct_evidence"] is True
    assert policy["conduct_inventory"] == {
        "source": "target_only",
        "closed_during_consequence_binding": True,
        "preserves_label_path_and_evidence": True,
    }
    assert policy["target_coverage"] == {
        "exhaustive_specific_conducts": True,
        "internal_llm_audit": True,
        "empty_only_after_complete_audit": True,
    }
    assert policy["consequence_assembly"] == {
        "classification_owner": "llm",
        "include_preceding_structural_heading": True,
        "include_adjacent_explicit_references": True,
        "include_same_article_explicit_references": True,
        "bind_by_specific_provision": True,
        "require_governing_primary_heading": True,
        "separate_conditional_variants": True,
        "single_monetary_penalty_per_variant": True,
        "single_point_deduction_per_variant": True,
        "attach_all_supported_consequences": True,
    }
    assert policy["applicability_exceptions"] == {
        "classification_owner": "llm",
        "representation": "condition_tree",
        "repeat_for_each_atomic_conduct": True,
        "separate_from_conduct_label": True,
        "preserve_local_scope": True,
    }
    assert policy["provision_path"] == {
        "ordered_levels": ["Chương", "Mục", "Điều", "Khoản", "Điểm"],
        "separator": " > ",
        "include_heading_titles": False,
    }


def test_population_contract_versions_complete_rule_assembly_policy() -> None:
    """V0.14 inventories only conducts stated directly in each Target."""
    contract = load_yaml(CONTRACT_PATH)

    assert contract["artifact_version"] == "0.14.0"
    assert any("applicability exception" in boundary for boundary in contract["semantic_boundaries"])


def test_population_prompt_audits_conditional_variants_and_local_exceptions() -> None:
    """The typed prompt protects the reviewed population regression boundaries."""
    prompt = PROMPT_PATH.read_text(encoding="utf-8")

    assert "Với mỗi conduct, xác định heading" in prompt
    assert "ArticleContext" in prompt
    assert "Chỉ bind khi tuple" in prompt
    assert "cùng Khoản nhưng liệt kê Điểm khác" in prompt
    assert "Audit từng target_conduct_id" in prompt
    assert "thuộc phạm vi" in prompt
    assert "variant riêng" in prompt
    assert "quả chính gần nhất còn mở" in prompt
    assert "tối đa một MonetaryPenalty" in prompt
    assert "không thay thế" in prompt
    assert "Đọc toàn bộ các danh mục consequence" in prompt


def test_population_prompt_preserves_vietnamese_legal_structure() -> None:
    """Vietnamese locators and coordinated conduct remain source-faithful."""
    prompt = PROMPT_PATH.read_text(encoding="utf-8")

    assert "danh sách đóng" in prompt
    assert "`d` và `đ` là hai Điểm khác nhau" in prompt
    assert "tuple đầy đủ (Điều, Khoản, Điểm)" in prompt
    assert "khớp chính xác provision của rule" in prompt
    assert "lặp lại ý nghĩa đó trong từng label" in prompt
    assert "không được chỉ ghi chuỗi ID" in prompt
    assert "không biến hành động tuân thủ thành hành vi vi phạm" in prompt
    assert "Không mở rộng một dẫn chiếu" in prompt
    assert "Không dựng conduct từ tên hình phạt" in prompt
    assert "không trỏ tới clause catalog" in prompt
    assert "loại entry không có direct conduct clause" in prompt
    assert "function ExtractOntologyTargetConductBatch" in prompt
    assert "function BindOntologyPopulationBatch" in prompt


def test_population_console_script_is_published() -> None:
    """The package exposes the ontology population command."""
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as file:
        project = tomllib.load(file)["project"]

    assert project["scripts"]["legal-populate-ontology"] == "legal_rag.ontology_population_cli:main"
    assert project["scripts"]["legal-check-ontology-population"] == "legal_rag.ontology_population_preflight_cli:main"
