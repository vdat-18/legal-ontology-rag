"""Deterministic gold checks for ontology population pilots."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Literal

from legal_rag.ontology_population.models import (
    ConsequenceKind,
    OntologyPopulationRecord,
    PopulationConsequence,
    PopulationProgressRecord,
    SubjectKind,
)
from legal_rag.ontology_validation.loader import load_yaml

POPULATION_GOLD_SCHEMA_VERSION = "0.2.0"
POPULATION_EVALUATION_REPORT_SCHEMA_VERSION = "0.2.0"

EvaluationStatus = Literal["passed", "failed"]

_CONSEQUENCE_KINDS = {
    "MonetaryPenalty",
    "Warning",
    "PointDeduction",
    "AdditionalSanction",
    "RemedialMeasure",
    "ReferencedPenalty",
}
_SUBJECT_KINDS = {"NaturalPerson", "Organization"}


@dataclass(frozen=True, slots=True)
class ConsequenceExpectation:
    """Fields that one populated consequence must match exactly."""

    kind: ConsequenceKind
    minimum_amount: Decimal | None = None
    maximum_amount: Decimal | None = None
    point_value: int | None = None
    consequence_label_vi: str | None = None
    minimum_duration_months: int | None = None
    maximum_duration_months: int | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> ConsequenceExpectation:
        """Parse one trusted versioned consequence expectation."""
        _reject_unknown_fields(
            payload,
            {
                "kind",
                "minimum_amount",
                "maximum_amount",
                "point_value",
                "consequence_label_vi",
                "minimum_duration_months",
                "maximum_duration_months",
            },
            "consequence expectation",
        )
        if payload["kind"] not in _CONSEQUENCE_KINDS:
            msg = f"Unsupported population gold consequence kind: {payload['kind']}"
            raise ValueError(msg)
        return cls(
            kind=payload["kind"],
            minimum_amount=_optional_decimal(payload.get("minimum_amount")),
            maximum_amount=_optional_decimal(payload.get("maximum_amount")),
            point_value=payload.get("point_value"),
            consequence_label_vi=payload.get("consequence_label_vi"),
            minimum_duration_months=payload.get("minimum_duration_months"),
            maximum_duration_months=payload.get("maximum_duration_months"),
        )

    def matches(self, consequence: PopulationConsequence) -> bool:
        """Return whether every declared field matches one consequence."""
        return (
            consequence.kind == self.kind
            and _matches_optional(self.minimum_amount, consequence.minimum_amount)
            and _matches_optional(self.maximum_amount, consequence.maximum_amount)
            and _matches_optional(self.point_value, consequence.point_value)
            and _matches_optional(self.consequence_label_vi, consequence.consequence_label_vi)
            and _matches_optional(self.minimum_duration_months, consequence.minimum_duration_months)
            and _matches_optional(self.maximum_duration_months, consequence.maximum_duration_months)
        )


@dataclass(frozen=True, slots=True)
class PopulationRecordExpectation:
    """One required or forbidden source-grounded rule signature."""

    assertion_id: str
    provision_path: str
    conduct_label_vi: str | None
    subject_kind: SubjectKind | None
    role_labels_vi: tuple[str, ...] | None
    vehicle_category_labels_vi: tuple[str, ...] | None
    evidence_texts: tuple[str, ...]
    consequences: tuple[ConsequenceExpectation, ...]

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> PopulationRecordExpectation:
        """Parse one trusted versioned record expectation."""
        _reject_unknown_fields(
            payload,
            {
                "id",
                "provision_path",
                "conduct_label_vi",
                "subject_kind",
                "role_labels_vi",
                "vehicle_category_labels_vi",
                "evidence_texts",
                "consequences",
            },
            "record expectation",
        )
        if payload.get("subject_kind") is not None and payload["subject_kind"] not in _SUBJECT_KINDS:
            msg = f"Unsupported population gold subject kind: {payload['subject_kind']}"
            raise ValueError(msg)
        return cls(
            assertion_id=payload["id"],
            provision_path=payload["provision_path"],
            conduct_label_vi=payload.get("conduct_label_vi"),
            subject_kind=payload.get("subject_kind"),
            role_labels_vi=_optional_strings(payload.get("role_labels_vi")),
            vehicle_category_labels_vi=_optional_strings(payload.get("vehicle_category_labels_vi")),
            evidence_texts=tuple(payload.get("evidence_texts", ())),
            consequences=tuple(ConsequenceExpectation.from_dict(item) for item in payload.get("consequences", ())),
        )

    def matches(self, record: OntologyPopulationRecord) -> bool:
        """Return whether one population record satisfies this exact signature."""
        if record.provision_path != self.provision_path:
            return False
        if not _matches_optional(self.conduct_label_vi, record.conduct_label_vi):
            return False
        if not _matches_optional(self.subject_kind, record.subject_kind):
            return False
        if self.role_labels_vi is not None and set(record.role_labels_vi) != set(self.role_labels_vi):
            return False
        if self.vehicle_category_labels_vi is not None and set(record.vehicle_category_labels_vi) != set(
            self.vehicle_category_labels_vi
        ):
            return False
        evidence_texts = {
            evidence.exact_text
            for evidence in (
                *record.evidence,
                *(evidence for condition in record.conditions for evidence in condition.evidence),
                *(evidence for consequence in record.consequences for evidence in consequence.evidence),
            )
        }
        return set(self.evidence_texts) <= evidence_texts and _match_distinct_consequences(
            self.consequences, record.consequences
        )


@dataclass(frozen=True, slots=True)
class PopulationGoldCase:
    """Gold assertions for one selected target chunk."""

    source_chunk_id: str
    expected_record_count: int
    required_records: tuple[PopulationRecordExpectation, ...]
    forbidden_records: tuple[PopulationRecordExpectation, ...]


@dataclass(frozen=True, slots=True)
class PopulationGoldStandard:
    """Versioned pilot selection and its reviewed semantic assertions."""

    document_number: str
    cases: tuple[PopulationGoldCase, ...]

    @classmethod
    def from_path(cls, path: Path) -> PopulationGoldStandard:
        """Load and validate one population gold standard."""
        document = load_yaml(path)
        _reject_unknown_fields(document, {"schema_version", "document_number", "cases"}, "gold standard")
        if document["schema_version"] != POPULATION_GOLD_SCHEMA_VERSION:
            msg = f"Unsupported ontology population gold schema: {document['schema_version']}"
            raise ValueError(msg)
        cases = tuple(_gold_case_from_dict(item) for item in document["cases"])
        _validate_gold_cases(cases)
        return cls(document_number=document["document_number"], cases=cases)


@dataclass(frozen=True, slots=True)
class PopulationEvaluationIssue:
    """One failed deterministic population check."""

    check_id: str
    source_chunk_id: str | None
    assertion_id: str | None
    summary: str

    def to_dict(self) -> dict[str, str | None]:
        """Return a JSON-serializable issue."""
        return {
            "check_id": self.check_id,
            "source_chunk_id": self.source_chunk_id,
            "assertion_id": self.assertion_id,
            "summary": self.summary,
        }


@dataclass(frozen=True, slots=True)
class PopulationEvaluationReport:
    """Machine-readable population gold-evaluation result."""

    overall_status: EvaluationStatus
    document_number: str
    target_chunk_count: int
    processed_target_count: int
    population_record_count: int
    expected_record_count: int
    required_assertion_count: int
    forbidden_assertion_count: int
    issues: tuple[PopulationEvaluationIssue, ...]

    def to_dict(self) -> dict[str, object]:
        """Return the stable evaluation report schema."""
        return {
            "schema_version": POPULATION_EVALUATION_REPORT_SCHEMA_VERSION,
            "overall_status": self.overall_status,
            "document_number": self.document_number,
            "target_chunk_count": self.target_chunk_count,
            "processed_target_count": self.processed_target_count,
            "population_record_count": self.population_record_count,
            "expected_record_count": self.expected_record_count,
            "required_assertion_count": self.required_assertion_count,
            "forbidden_assertion_count": self.forbidden_assertion_count,
            "issues": [issue.to_dict() for issue in self.issues],
        }


def evaluate_population_records(
    records: tuple[OntologyPopulationRecord, ...],
    progress: tuple[PopulationProgressRecord, ...],
    gold: PopulationGoldStandard,
) -> PopulationEvaluationReport:
    """Evaluate persisted population output against reviewed structured assertions."""
    cases_by_chunk = {case.source_chunk_id: case for case in gold.cases}
    expected_chunk_ids = set(cases_by_chunk)
    processed_chunk_ids = {item.chunk_id for item in progress}
    issues = [
        *_target_issues(progress, expected_chunk_ids),
        *_record_inventory_issues(records, expected_chunk_ids, gold.document_number),
    ]

    records_by_chunk = {
        chunk_id: tuple(record for record in records if record.source_chunk_id == chunk_id)
        for chunk_id in expected_chunk_ids
    }
    for case in gold.cases:
        issues.extend(_case_issues(case, records_by_chunk[case.source_chunk_id]))

    return PopulationEvaluationReport(
        overall_status="failed" if issues else "passed",
        document_number=gold.document_number,
        target_chunk_count=len(gold.cases),
        processed_target_count=len(processed_chunk_ids & expected_chunk_ids),
        population_record_count=len(records),
        expected_record_count=sum(case.expected_record_count for case in gold.cases),
        required_assertion_count=sum(len(case.required_records) for case in gold.cases),
        forbidden_assertion_count=sum(len(case.forbidden_records) for case in gold.cases),
        issues=tuple(issues),
    )


def _target_issues(
    progress: tuple[PopulationProgressRecord, ...], expected_chunk_ids: set[str]
) -> tuple[PopulationEvaluationIssue, ...]:
    """Check exact completion coverage for the reviewed pilot selection."""
    progress_ids = [item.chunk_id for item in progress]
    processed_chunk_ids = set(progress_ids)
    issues = [
        *(
            _issue("EVAL-TARGET-001", chunk_id, None, "Target chunk has not completed population.")
            for chunk_id in sorted(expected_chunk_ids - processed_chunk_ids)
        ),
        *(
            _issue("EVAL-TARGET-002", chunk_id, None, "Progress contains a chunk outside the gold standard.")
            for chunk_id in sorted(processed_chunk_ids - expected_chunk_ids)
        ),
    ]
    if len(progress_ids) != len(processed_chunk_ids):
        issues.append(_issue("EVAL-TARGET-003", None, None, "Progress contains duplicate target chunk IDs."))
    return tuple(issues)


def _record_inventory_issues(
    records: tuple[OntologyPopulationRecord, ...], expected_chunk_ids: set[str], document_number: str
) -> tuple[PopulationEvaluationIssue, ...]:
    """Check record identity and ownership before semantic assertions."""
    population_ids = [record.population_id for record in records]
    issues: list[PopulationEvaluationIssue] = []
    if len(population_ids) != len(set(population_ids)):
        issues.append(_issue("EVAL-RECORD-001", None, None, "Population output contains duplicate population IDs."))
    issues.extend(
        _issue(
            "EVAL-RECORD-002",
            record.source_chunk_id,
            None,
            "Population record belongs to a chunk outside the gold standard.",
        )
        for record in records
        if record.source_chunk_id not in expected_chunk_ids
    )
    issues.extend(
        _issue(
            "EVAL-RECORD-003",
            record.source_chunk_id,
            None,
            f"Population document number is {record.document_number!r}, expected {document_number!r}.",
        )
        for record in records
        if record.document_number != document_number
    )
    return tuple(issues)


def _case_issues(
    case: PopulationGoldCase, records: tuple[OntologyPopulationRecord, ...]
) -> tuple[PopulationEvaluationIssue, ...]:
    """Check exact coverage and required or forbidden signatures for one target."""
    issues: list[PopulationEvaluationIssue] = []
    if len(records) < case.expected_record_count:
        issues.append(
            _issue(
                "EVAL-COVERAGE-001",
                case.source_chunk_id,
                None,
                f"Found {len(records)} records; expected exactly {case.expected_record_count}.",
            )
        )
    elif len(records) > case.expected_record_count:
        issues.append(
            _issue(
                "EVAL-COVERAGE-002",
                case.source_chunk_id,
                None,
                f"Found {len(records)} records; expected exactly {case.expected_record_count}.",
            )
        )
    issues.extend(
        _issue(
            "EVAL-REQUIRED-001",
            case.source_chunk_id,
            assertion.assertion_id,
            "Required population assertion is missing.",
        )
        for assertion in _unmatched_required_assertions(case.required_records, records)
    )
    issues.extend(
        _issue(
            "EVAL-FORBIDDEN-001",
            case.source_chunk_id,
            assertion.assertion_id,
            "Forbidden population assertion is present.",
        )
        for assertion in case.forbidden_records
        if any(assertion.matches(record) for record in records)
    )
    return tuple(issues)


def _unmatched_required_assertions(
    assertions: tuple[PopulationRecordExpectation, ...],
    records: tuple[OntologyPopulationRecord, ...],
) -> tuple[PopulationRecordExpectation, ...]:
    """Match each required signature to a distinct population record."""
    assertion_by_record: dict[int, int] = {}

    def assign(assertion_index: int, visited_records: set[int]) -> bool:
        for record_index, record in enumerate(records):
            if record_index in visited_records or not assertions[assertion_index].matches(record):
                continue
            visited_records.add(record_index)
            previous_assertion = assertion_by_record.get(record_index)
            if previous_assertion is None or assign(previous_assertion, visited_records):
                assertion_by_record[record_index] = assertion_index
                return True
        return False

    return tuple(
        assertion for assertion_index, assertion in enumerate(assertions) if not assign(assertion_index, set())
    )


def _validate_gold_cases(cases: tuple[PopulationGoldCase, ...]) -> None:
    """Reject ambiguous or incomplete gold artifacts at the input boundary."""
    if not cases:
        msg = "Ontology population gold standard must contain at least one case"
        raise ValueError(msg)
    chunk_ids = [case.source_chunk_id for case in cases]
    assertion_ids = [
        assertion.assertion_id for case in cases for assertion in (*case.required_records, *case.forbidden_records)
    ]
    if len(chunk_ids) != len(set(chunk_ids)):
        msg = "Ontology population gold standard contains duplicate source chunk IDs"
        raise ValueError(msg)
    if len(assertion_ids) != len(set(assertion_ids)):
        msg = "Ontology population gold standard contains duplicate assertion IDs"
        raise ValueError(msg)
    if any(case.expected_record_count < 0 for case in cases):
        msg = "Ontology population expected record count cannot be negative"
        raise ValueError(msg)
    if any(len(case.required_records) > case.expected_record_count for case in cases):
        msg = "Ontology population required assertions cannot exceed the expected record count"
        raise ValueError(msg)


def _gold_case_from_dict(payload: dict[str, object]) -> PopulationGoldCase:
    """Parse one target case while rejecting unused schema fields."""
    _reject_unknown_fields(
        payload,
        {"source_chunk_id", "expected_record_count", "required_records", "forbidden_records"},
        "gold case",
    )
    return PopulationGoldCase(
        source_chunk_id=payload["source_chunk_id"],
        expected_record_count=payload["expected_record_count"],
        required_records=tuple(
            PopulationRecordExpectation.from_dict(assertion) for assertion in payload.get("required_records", ())
        ),
        forbidden_records=tuple(
            PopulationRecordExpectation.from_dict(assertion) for assertion in payload.get("forbidden_records", ())
        ),
    )


def _match_distinct_consequences(
    expected: tuple[ConsequenceExpectation, ...], actual: tuple[PopulationConsequence, ...]
) -> bool:
    """Match each expected signature to a distinct populated consequence."""
    if not expected:
        return True

    def match_from(index: int, used: frozenset[int]) -> bool:
        if index == len(expected):
            return True
        return any(
            consequence_index not in used
            and expected[index].matches(consequence)
            and match_from(index + 1, used | {consequence_index})
            for consequence_index, consequence in enumerate(actual)
        )

    return match_from(0, frozenset())


def _matches_optional(expected: object | None, actual: object | None) -> bool:
    """Treat omitted expectation fields as unconstrained."""
    return expected is None or expected == actual


def _optional_decimal(value: object | None) -> Decimal | None:
    """Parse one optional exact decimal value."""
    return Decimal(str(value)) if value is not None else None


def _optional_strings(value: object | None) -> tuple[str, ...] | None:
    """Parse an optional unordered label collection."""
    return tuple(value) if value is not None else None


def _reject_unknown_fields(payload: dict[str, object], allowed: set[str], artifact: str) -> None:
    """Reject gold fields that have no evaluator semantics."""
    unknown = set(payload) - allowed
    if unknown:
        msg = f"Unknown {artifact} fields: {', '.join(sorted(unknown))}"
        raise ValueError(msg)


def _issue(
    check_id: str, source_chunk_id: str | None, assertion_id: str | None, summary: str
) -> PopulationEvaluationIssue:
    """Build one concise failed check."""
    return PopulationEvaluationIssue(check_id, source_chunk_id, assertion_id, summary)
