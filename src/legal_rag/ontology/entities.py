"""Custom Graphiti ontology for the Vietnamese traffic-penalty domain.

Defines the entity and edge Pydantic models passed to
``Graphiti.add_episode(entity_types=..., edge_types=..., edge_type_map=...)``.
Field descriptions are in Vietnamese because they steer the LLM extraction on
Vietnamese legal text. These models describe *what to extract*; they are not a
storage schema.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# Entity types
# --------------------------------------------------------------------------- #


class LegalDocument(BaseModel):
    """Một văn bản quy phạm pháp luật (Luật, Nghị định, Thông tư, VBHN)."""

    so_hieu: str | None = Field(default=None, description="Số hiệu văn bản, ví dụ '168/2024/NĐ-CP'.")
    loai: str | None = Field(default=None, description="Loại văn bản: Luật, Nghị định, Thông tư, Văn bản hợp nhất.")
    ngay_hieu_luc: str | None = Field(default=None, description="Ngày có hiệu lực (ISO yyyy-mm-dd nếu xác định được).")


class TrafficViolation(BaseModel):
    """Một hành vi vi phạm hành chính về giao thông đường bộ bị xử phạt."""

    mo_ta: str = Field(description="Mô tả hành vi vi phạm, ví dụ 'vượt đèn đỏ'.")
    linh_vuc: str | None = Field(default=None, description="Lĩnh vực/nhóm hành vi (tốc độ, tín hiệu, nồng độ cồn...).")


class Vehicle(BaseModel):
    """Loại phương tiện áp dụng chế tài (xe máy, ô tô, xe tải...)."""

    loai_phuong_tien: str = Field(description="Tên loại phương tiện đã chuẩn hoá, ví dụ 'xe mô tô, xe gắn máy'.")


class Penalty(BaseModel):
    """Chế tài áp dụng cho một hành vi vi phạm tại một thời điểm hiệu lực."""

    hinh_thuc: str = Field(description="Hình thức: phạt tiền, tước GPLX, tịch thu, cảnh cáo...")
    muc_tien_min: int | None = Field(default=None, description="Cận dưới mức phạt tiền (đồng).")
    muc_tien_max: int | None = Field(default=None, description="Cận trên mức phạt tiền (đồng).")
    tru_diem: int | None = Field(default=None, description="Số điểm bị trừ trên GPLX (nếu có).")


class LegalAuthority(BaseModel):
    """Cơ quan/chức danh có thẩm quyền xử phạt hoặc thực thi."""

    ten: str = Field(description="Tên cơ quan hoặc chức danh có thẩm quyền.")


# --------------------------------------------------------------------------- #
# Edge (relationship) types
# --------------------------------------------------------------------------- #


class HasPenalty(BaseModel):
    """Hành vi vi phạm bị áp dụng một chế tài."""

    provision: str | None = Field(default=None, description="Điều/Khoản/Điểm quy định, ví dụ 'Điều 6 Khoản 3 điểm a'.")


class AppliesTo(BaseModel):
    """Chế tài/hành vi áp dụng cho một loại phương tiện."""


class DefinedIn(BaseModel):
    """Hành vi hoặc chế tài được quy định trong một văn bản pháp luật."""

    provision: str | None = Field(default=None, description="Điều/Khoản/Điểm gốc.")


class Amends(BaseModel):
    """Một văn bản sửa đổi, bổ sung hoặc thay thế văn bản khác."""

    quan_he: str | None = Field(default=None, description="Loại quan hệ: sửa đổi, bổ sung, thay thế, hợp nhất.")


ENTITY_TYPES: dict[str, type[BaseModel]] = {
    "LegalDocument": LegalDocument,
    "TrafficViolation": TrafficViolation,
    "Vehicle": Vehicle,
    "Penalty": Penalty,
    "LegalAuthority": LegalAuthority,
}

EDGE_TYPES: dict[str, type[BaseModel]] = {
    "HAS_PENALTY": HasPenalty,
    "APPLIES_TO": AppliesTo,
    "DEFINED_IN": DefinedIn,
    "AMENDS": Amends,
}

# Constrain which edge types may connect which entity types.
EDGE_TYPE_MAP: dict[tuple[str, str], list[str]] = {
    ("TrafficViolation", "Penalty"): ["HAS_PENALTY"],
    ("TrafficViolation", "Vehicle"): ["APPLIES_TO"],
    ("Penalty", "Vehicle"): ["APPLIES_TO"],
    ("TrafficViolation", "LegalDocument"): ["DEFINED_IN"],
    ("Penalty", "LegalDocument"): ["DEFINED_IN"],
    ("LegalDocument", "LegalDocument"): ["AMENDS"],
}
