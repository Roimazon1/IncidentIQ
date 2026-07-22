"""Focused tests for deterministic AI evidence-reference validation."""

import json
from pathlib import Path
from typing import TypeAlias

import pytest
from pydantic import BaseModel

from app.models import ClaimSupportStatus, EvidenceType
from app.schemas.ai_outputs import (
    CriticOutputV1,
    EvidenceReferenceV1,
    HypothesesOutputV1,
    SummaryOutputV1,
    TimelineOutputV1,
)
from app.schemas.evidence import EvidenceManifest, EvidenceManifestSource
from app.services.evidence_manifest_service import EvidenceManifestService
from app.services.validation_service import (
    EvidenceReferenceValidationStatus,
    ValidationService,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "fake_ai_responses.json"
OutputModel: TypeAlias = type[
    SummaryOutputV1 | TimelineOutputV1 | HypothesesOutputV1 | CriticOutputV1
]


def _manifest() -> EvidenceManifest:
    return EvidenceManifestService.build_evidence_manifest(
        "INC-000001",
        (
            EvidenceManifestSource(
                evidence_code="E-001",
                source_name="checkout.log",
                evidence_type=EvidenceType.APPLICATION_LOG,
                original_text=(
                    "Checkout failed  \r\napi_key=local-secret\r\nretry=false\r\n"
                ),
            ),
            EvidenceManifestSource(
                evidence_code="E-002",
                source_name="deployment.txt",
                evidence_type=EvidenceType.DEPLOYMENT_NOTE,
                original_text="Deployment completed",
            ),
        ),
    )


def _replace_first_evidence_id(value: object) -> bool:
    if isinstance(value, dict):
        if "evidence_id" in value:
            value["evidence_id"] = "E-999"
            return True
        return any(_replace_first_evidence_id(item) for item in value.values())
    if isinstance(value, list):
        return any(_replace_first_evidence_id(item) for item in value)
    return False


def _count_evidence_references(value: object) -> int:
    if isinstance(value, dict):
        return int("evidence_id" in value) + sum(
            _count_evidence_references(item) for item in value.values()
        )
    if isinstance(value, list):
        return sum(_count_evidence_references(item) for item in value)
    return 0


def test_unknown_evidence_id_produces_invalid_outcome() -> None:
    outcomes = ValidationService.validate_evidence_ids(
        (EvidenceReferenceV1(evidence_id="E-999", line_range="1"),),
        _manifest(),
    )

    assert len(outcomes) == 1
    assert outcomes[0].is_valid is False
    assert outcomes[0].status is (EvidenceReferenceValidationStatus.UNKNOWN_EVIDENCE_ID)
    assert outcomes[0].message == (
        "Evidence identifier is not present in the analysis manifest."
    )
    assert "local-secret" not in repr(outcomes[0])


def test_valid_references_produce_valid_outcomes() -> None:
    references = (
        EvidenceReferenceV1(evidence_id="E-001", line_range="1"),
        EvidenceReferenceV1(evidence_id="E-002", line_range="1"),
    )

    outcomes = ValidationService.validate_evidence_ids(references, _manifest())

    assert [outcome.evidence_id for outcome in outcomes] == ["E-001", "E-002"]
    assert all(outcome.is_valid for outcome in outcomes)
    assert all(
        outcome.status is EvidenceReferenceValidationStatus.VALID
        for outcome in outcomes
    )


def test_validate_supporting_excerpt_uses_normalized_redacted_evidence() -> None:
    manifest = _manifest()

    outcomes = (
        ValidationService.validate_supporting_excerpt(
            EvidenceReferenceV1(
                evidence_id="E-001",
                line_range="1",
                excerpt="Checkout failed",
            ),
            manifest,
        ),
        ValidationService.validate_supporting_excerpt(
            EvidenceReferenceV1(
                evidence_id="E-001",
                line_range="2",
                excerpt="api_key=[REDACTED_API_KEY]",
            ),
            manifest,
        ),
    )

    assert all(outcome.is_valid for outcome in outcomes)


@pytest.mark.parametrize(
    "excerpt,line_range",
    [
        ("Fabricated failure", "1-3"),
        ("checkout failed", "1"),
        ("retry=false", "1"),
        ("api_key=local-secret", "2"),
    ],
)
def test_unmatched_excerpt_produces_invalid_outcome(
    excerpt: str,
    line_range: str,
) -> None:
    outcome = ValidationService.validate_supporting_excerpt(
        EvidenceReferenceV1(
            evidence_id="E-001",
            line_range=line_range,
            excerpt=excerpt,
        ),
        _manifest(),
    )

    assert outcome.is_valid is False
    assert outcome.status is EvidenceReferenceValidationStatus.EXCERPT_MISMATCH
    assert outcome.message == (
        "Excerpt does not match the referenced normalized redacted evidence."
    )
    assert excerpt not in repr(outcome)


def test_mixed_valid_and_invalid_references_are_all_reported() -> None:
    references = (
        EvidenceReferenceV1(
            evidence_id="E-001",
            line_range="1",
            excerpt="Checkout failed",
        ),
        EvidenceReferenceV1(evidence_id="E-999", line_range="1"),
        EvidenceReferenceV1(
            evidence_id="E-001",
            line_range="1",
            excerpt="fabricated failure",
        ),
    )

    outcomes = tuple(
        ValidationService.validate_supporting_excerpt(reference, _manifest())
        for reference in references
    )

    assert [outcome.status for outcome in outcomes] == [
        EvidenceReferenceValidationStatus.VALID,
        EvidenceReferenceValidationStatus.UNKNOWN_EVIDENCE_ID,
        EvidenceReferenceValidationStatus.EXCERPT_MISMATCH,
    ]


def test_validation_outcomes_do_not_classify_claim_support() -> None:
    outcome = ValidationService.validate_supporting_excerpt(
        EvidenceReferenceV1(evidence_id="E-999", line_range="1"),
        _manifest(),
    )

    assert not hasattr(outcome, "support_status")
    assert outcome.status.value not in {
        support_status.value for support_status in ClaimSupportStatus
    }


@pytest.mark.parametrize(
    ("fixture_name", "output_model"),
    [
        ("valid_summary", SummaryOutputV1),
        ("valid_timeline", TimelineOutputV1),
        ("valid_hypotheses", HypothesesOutputV1),
        ("valid_critic", CriticOutputV1),
    ],
)
def test_nested_stage_outputs_report_unknown_evidence_identifiers(
    fixture_name: str,
    output_model: OutputModel,
) -> None:
    fixture_bank = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    output_data = json.loads(fixture_bank[fixture_name]["raw_response"])
    assert _replace_first_evidence_id(output_data) is True
    output: BaseModel = output_model.model_validate(output_data)

    outcomes = ValidationService.validate_output_references(output, _manifest())

    assert len(outcomes) == _count_evidence_references(output_data)
    assert any(
        outcome.status is EvidenceReferenceValidationStatus.UNKNOWN_EVIDENCE_ID
        for outcome in outcomes
    )
