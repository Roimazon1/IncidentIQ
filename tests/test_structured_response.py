"""Focused tests for the shared structured AI response boundary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.schemas.ai_outputs import SummaryOutputV1
from app.schemas.ai_provider import AIFailureCategory
from app.services.ai_provider import (
    AIProviderConfigurationError,
    BoundedRetryPolicy,
    process_structured_response,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "fake_ai_responses.json"


def _valid_summary_response() -> str:
    fixture_bank = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    return fixture_bank["valid_summary"]["raw_response"]


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
