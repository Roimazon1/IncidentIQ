"""Focused tests for the shared structured AI response boundary."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from app.schemas.ai_outputs import HypothesesOutputV1, SummaryOutputV1
from app.schemas.ai_provider import AIFailureCategory
from app.services.ai_provider import (
    AIProviderConfigurationError,
    BoundedRetryPolicy,
    build_model_facing_output_schema,
    process_structured_response,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "fake_ai_responses.json"


def _valid_summary_response() -> str:
    fixture_bank = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    return fixture_bank["valid_summary"]["raw_response"]


def _valid_hypotheses_data() -> dict[str, object]:
    fixture_bank = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    return json.loads(fixture_bank["valid_hypotheses"]["raw_response"])


def test_shared_response_boundary_returns_locally_validated_output() -> None:
    raw_response = _valid_summary_response()

    outcome = process_structured_response(raw_response, SummaryOutputV1)

    assert isinstance(outcome.output, SummaryOutputV1)
    assert outcome.failure_category is None
    assert raw_response not in repr(outcome)
    assert "raw_response" not in repr(outcome)


@pytest.mark.parametrize(
    ("raw_response", "expected_category"),
    [
        (
            '{"sensitive":"malformed response"',
            AIFailureCategory.MALFORMED_JSON,
        ),
        (
            '{"sensitive":"schema-invalid response"}',
            AIFailureCategory.SCHEMA_VALIDATION,
        ),
    ],
)
def test_shared_response_boundary_contains_invalid_raw_output(
    raw_response: str,
    expected_category: AIFailureCategory,
) -> None:
    outcome = process_structured_response(raw_response, SummaryOutputV1)

    assert outcome.output is None
    assert outcome.failure_category is expected_category
    assert raw_response not in repr(outcome)
    assert "raw_response" not in repr(outcome)


def test_schema_validation_retains_only_sanitized_nested_error_details() -> None:
    secret = "sk-validationsecret123"
    response_data = json.loads(_valid_summary_response())
    response_data["facts"][0]["confidence"] = f"api_key={secret}"
    response_data["facts"][0][f"api_key={secret}"] = "private input"
    raw_response = json.dumps(response_data)

    outcome = process_structured_response(raw_response, SummaryOutputV1)

    assert outcome.failure_category is AIFailureCategory.SCHEMA_VALIDATION
    confidence_error = next(
        error
        for error in outcome.validation_errors
        if error.loc == ("facts", 0, "confidence")
    )
    assert confidence_error.type == "int_type"
    assert confidence_error.message
    serialized_diagnostics = json.dumps(
        [asdict(error) for error in outcome.validation_errors]
    )
    assert all(
        set(asdict(error)) == {"loc", "type", "message"}
        for error in outcome.validation_errors
    )
    assert secret not in serialized_diagnostics
    assert "private input" not in serialized_diagnostics
    assert raw_response not in serialized_diagnostics
    assert '"input"' not in serialized_diagnostics
    assert '"ctx"' not in serialized_diagnostics


@pytest.mark.parametrize(
    "model_ids",
    [
        ["H1", "H-1", "H-3"],
        ["H-001", "H-001", "H-001"],
        [None, None, None],
    ],
    ids=["invalid-shapes", "duplicates", "missing"],
)
def test_hypothesis_ids_are_assigned_deterministically_in_output_order(
    model_ids: list[str | None],
) -> None:
    response_data = _valid_hypotheses_data()
    hypotheses = response_data["hypotheses"]
    assert isinstance(hypotheses, list)
    original_titles = [hypothesis["title"] for hypothesis in hypotheses]
    for hypothesis, model_id in zip(hypotheses, model_ids, strict=True):
        if model_id is None:
            hypothesis.pop("hypothesis_id")
        else:
            hypothesis["hypothesis_id"] = model_id

    outcome = process_structured_response(
        json.dumps(response_data),
        HypothesesOutputV1,
    )

    assert isinstance(outcome.output, HypothesesOutputV1)
    assert [hypothesis.hypothesis_id for hypothesis in outcome.output.hypotheses] == [
        "H-001",
        "H-002",
        "H-003",
    ]
    assert [
        hypothesis.title for hypothesis in outcome.output.hypotheses
    ] == original_titles


def test_hypothesis_id_assignment_does_not_repair_semantic_violations() -> None:
    response_data = _valid_hypotheses_data()
    hypotheses = response_data["hypotheses"]
    assert isinstance(hypotheses, list)
    hypotheses[0]["hypothesis_id"] = "H1"
    hypotheses[0]["confidence"] = "not-an-integer"

    outcome = process_structured_response(
        json.dumps(response_data),
        HypothesesOutputV1,
    )

    assert outcome.output is None
    assert outcome.failure_category is AIFailureCategory.SCHEMA_VALIDATION
    assert any(
        error.loc == ("hypotheses", 0, "confidence") and error.type == "int_type"
        for error in outcome.validation_errors
    )
    assert all(
        error.loc != ("hypotheses", 0, "hypothesis_id")
        for error in outcome.validation_errors
    )


def test_hypotheses_public_schema_retains_strict_identifier_contract() -> None:
    schema = HypothesesOutputV1.model_json_schema()
    hypothesis_schema = schema["$defs"]["HypothesisV1"]

    assert "hypothesis_id" in hypothesis_schema["required"]
    assert hypothesis_schema["properties"]["hypothesis_id"]["pattern"] == r"^H-\d{3}$"
    assert set(hypothesis_schema["properties"]) == {
        "hypothesis_id",
        "rank",
        "title",
        "explanation",
        "confidence",
        "supporting_evidence",
        "contradicting_evidence",
        "missing_evidence",
        "validation_test",
        "risk_of_acting",
    }

    model_facing_schema = build_model_facing_output_schema(HypothesesOutputV1)
    model_facing_hypothesis = model_facing_schema["$defs"]["HypothesisV1"]
    assert "hypothesis_id" not in model_facing_hypothesis["properties"]
    assert "hypothesis_id" not in model_facing_hypothesis["required"]


def test_shared_response_boundary_contains_missing_response() -> None:
    outcome = process_structured_response(None, SummaryOutputV1)

    assert outcome.output is None
    assert outcome.failure_category is AIFailureCategory.TRANSIENT_PROVIDER_FAILURE
    assert "raw_response" not in repr(outcome)


def test_bounded_retry_policy_has_finite_attempts_and_deterministic_delays() -> None:
    policy = BoundedRetryPolicy(max_attempts=3, retry_delay_seconds=0.5)

    assert list(policy.attempt_numbers) == [1, 2, 3]
    assert policy.delay_before_next_attempt(1) == 0.5
    assert policy.delay_before_next_attempt(2) == 1.0
    assert not policy.has_next_attempt(3)
    with pytest.raises(ValueError, match="no AI provider retry remains"):
        policy.delay_before_next_attempt(3)


@pytest.mark.parametrize(
    ("max_attempts", "retry_delay_seconds"),
    [(0, 0.5), (True, 0.5), (3, -0.1), (3, True)],
)
def test_bounded_retry_policy_rejects_invalid_configuration_safely(
    max_attempts: int,
    retry_delay_seconds: float,
) -> None:
    with pytest.raises(AIProviderConfigurationError) as error_info:
        BoundedRetryPolicy(
            max_attempts=max_attempts,
            retry_delay_seconds=retry_delay_seconds,
        )

    assert error_info.value.details.category is AIFailureCategory.CONFIGURATION


@pytest.mark.parametrize(
    "retry_delay_seconds",
    [float("nan"), float("inf"), float("-inf")],
    ids=["nan", "positive-infinity", "negative-infinity"],
)
def test_bounded_retry_policy_rejects_non_finite_delay_safely(
    retry_delay_seconds: float,
) -> None:
    with pytest.raises(AIProviderConfigurationError) as error_info:
        BoundedRetryPolicy(
            max_attempts=3,
            retry_delay_seconds=retry_delay_seconds,
        )

    assert error_info.value.details.category is AIFailureCategory.CONFIGURATION
