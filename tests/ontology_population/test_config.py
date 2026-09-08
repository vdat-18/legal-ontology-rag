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
    assert settings.conduct_batch_size == 1
    assert settings.batch_size == 1
    assert settings.max_locator_context_chunks == 8
    assert settings.max_preceding_chunks == 2
    assert settings.max_following_chunks == 1
    assert settings.max_article_context_chunks == 32
    assert settings.max_output_tokens == 32768


def test_population_contract_uses_minimal_schema() -> None:
    """The population contract retains only values consumed by runtime code."""
    contract = load_yaml(CONTRACT_PATH)

    assert set(contract) == {
        "schema_version",
        "ontology_version",
        "instance_namespace",
        "reading_window",
        "llm",
    }


def test_population_prompt_preserves_core_semantic_boundaries() -> None:
    """The typed prompt keeps the few semantic boundaries required by the ontology."""
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    compact_prompt = " ".join(prompt.split())

    assert "chỉ cho quy định xử phạt vi phạm hành chính về giao thông đường bộ Việt Nam" in compact_prompt
    assert "Không dùng kiến thức ngoài" in compact_prompt
    assert "Giữ nguyên target conducts" in compact_prompt
    assert "cây AND/OR/NOT" in compact_prompt
    assert "Không tạo LegalChange" in compact_prompt
    assert "Mục đích bước 1/6" in compact_prompt
    assert "Mục đích bước 2/6" in compact_prompt


def test_population_prompt_preserves_vietnamese_legal_structure() -> None:
    """Vietnamese locators and coordinated conduct remain source-faithful."""
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    compact_prompt = " ".join(prompt.split())

    assert "danh mục đóng" in compact_prompt
    assert "mọi vị ngữ độc lập" in compact_prompt
    assert "P không đúng A, không đúng B" in compact_prompt
    assert "không được bỏ qua mục nằm giữa hai mục đã xử lý" in compact_prompt
    assert "tuyệt đối không tạo conduct hoặc đưa nó vào danh sách coverage hiện tại" in compact_prompt
    assert "không V hoặc V không đầy đủ/không đúng" in compact_prompt
    assert "danh sách các yêu cầu pháp lý độc lập" in compact_prompt
    assert "không chỉ giữ C ở nhánh gần nhất" in compact_prompt
    assert "bước sau sẽ cấu trúc hóa điều kiện" in compact_prompt
    assert "Hai đối tượng pháp lý độc lập nối bằng `hoặc`" in compact_prompt
    assert "Nhánh rút gọn phải tự đủ nghĩa" in compact_prompt
    assert "Không kéo O ngược tới V1/V2" in compact_prompt
    assert "không biến V4 thành `V3 V4 O`" in compact_prompt
    assert "danh sách mục đích `để được V1, V2, V3-Q, V4 O`" in compact_prompt
    assert "V3-Q là một hành động ghép không được tách Q" in compact_prompt
    assert "Chủ thể label là chủ thể chịu xử phạt" in compact_prompt
    assert "Mọi label phải giữ nguyên chủ thể đó và đủ loại phương tiện/phạm vi" in compact_prompt
    assert "không đổi nhánh thứ hai thành `T V`" in compact_prompt
    assert "`làm thay đổi A, B`" in compact_prompt
    assert "mỗi lá giữ đủ giấy tờ, loại xe, bên giữ/cấp và phạm vi" in compact_prompt
    assert "Không thêm động từ/hành động để nối chủ thể với hành vi" in compact_prompt
    assert "locator thuần túy" in compact_prompt
    assert "Chương > Mục > Điều > Khoản > Điểm" in compact_prompt
    assert "provision_path chỉ gồm mã locator" in compact_prompt
    assert "không kèm tên/tiêu đề Chương, Mục, Điều" in compact_prompt
    assert "Không dùng kiến thức ngoài" in compact_prompt
    assert "không dùng ID của block kế trước, kế sau hoặc quay vòng" in compact_prompt
    assert "không cắt ngắn, sửa prefix, suffix" in compact_prompt
    assert "Bối cảnh: xây dựng ontology" in compact_prompt
    assert '{{ _.role("system") }}' in prompt
    assert '{{ _.role("user") }}' in prompt
    assert "function ExtractOntologyTargetConductBatch" in prompt
    assert "function BindOntologyPopulationBatch" in prompt


def test_population_console_script_is_published() -> None:
    """The package exposes the ontology population command."""
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as file:
        project = tomllib.load(file)["project"]

    assert project["scripts"]["legal-populate-ontology"] == "legal_rag.ontology_population_cli:main"
    assert project["scripts"]["legal-check-ontology-population"] == "legal_rag.ontology_population_preflight_cli:main"
    assert (
        project["scripts"]["legal-evaluate-ontology-population"] == "legal_rag.ontology_population_evaluation_cli:main"
    )
