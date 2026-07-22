"""Focused tests for deterministic fixture-backed fake AI behavior."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.models.enums import EvidenceType
from app.schemas.ai_outputs import CompleteAnalysisOutputV1, SummaryOutputV1
from app.schemas.ai_provider import (
    AIFailureCategory,
    AIRequest,
    AnalysisStage,
    OutputSchemaIdentifier,
    PromptBundle,
    PromptName,
    PromptReference,
    PromptVersion,
    SafeAIMetadata,
)
from app.schemas.evidence import (
    EvidenceManifest,
    EvidenceManifestChunk,
    EvidenceManifestItem,
    EvidenceManifestTimestamp,
)
from app.services.ai_provider import AIProvider, AIProviderExecutionError
from app.services.providers.fake_provider import FakeAIProvider


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "fake_ai_responses.json"


def _summary_request() -> AIRequest:
    manifest = EvidenceManifest(
        incident_id="INC-000001",
        evidence=(
            EvidenceManifestItem(
                id="E-001",
                type=EvidenceType.APPLICATION_LOG,
                source="checkout.log",
                line_range="1-2",
                timestamps=(
                    EvidenceManifestTimestamp(
                        raw_text=None,
                        value=None,
                        status="unknown",
                        reason="no direct timestamp found",
                    ),
                ),
                chunks=(
                    EvidenceManifestChunk(
                        sequence=1,
                        line_range="1-2",
                        content="[REDACTED_API_KEY] checkout failed",
                    ),
                ),
            ),
        ),
    )
    prompts = PromptBundle(
        system=PromptReference(name=PromptName.SYSTEM, version=PromptVersion.V1),
        task=PromptReference(name=PromptName.SUMMARY, version=PromptVersion.V1),
    )
    return AIRequest(
        evidence_manifest=manifest,
        prompts=prompts,
        output_schema=OutputSchemaIdentifier.SUMMARY_V1,
        metadata=SafeAIMetadata(
            request_identifier="req-001",
            incident_public_identifier="INC-000001",
            analysis_stage=AnalysisStage.SUMMARY,
            evidence_manifest_checksum="a" * 64,
        ),
    )


def test_valid_complete_output_fixture_matches_composition_schema() -> None:
    fixture_bank = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    raw_response = fixture_bank["valid_complete_output"]["raw_response"]

    output = CompleteAnalysisOutputV1.model_validate_json(raw_response)

    assert len(output.hypotheses) == 3


def test_fake_provider_is_deterministic_and_returns_audited_typed_result() -> None:
    provider = FakeAIProvider.from_file(FIXTURE_PATH, "valid_summary")
    request = _summary_request()

    first_result = provider.generate(request)
    second_result = provider.generate(request)

    assert isinstance(provider, AIProvider)
    assert first_result == second_result
    assert isinstance(first_result.output, SummaryOutputV1)
    assert first_result.metadata.provider_name == "fake"
    assert first_result.metadata.model_name == "fixture-v1"
    assert first_result.metadata.system_prompt == request.prompts.system
    assert first_result.metadata.task_prompt == request.prompts.task
    assert first_result.metadata.request_identifier == "req-001"
    assert first_result.metadata.attempt_count == 1
    assert first_result.audit.raw_response
    assert "audit" not in first_result.model_dump()
    assert "raw_response" not in first_result.model_dump_json()


@pytest.mark.parametrize(
    ("fixture_name", "expected_category"),
    [
        ("invalid_json", AIFailureCategory.MALFORMED_JSON),
        ("missing_fields", AIFailureCategory.SCHEMA_VALIDATION),
        ("out_of_range_confidence", AIFailureCategory.SCHEMA_VALIDATION),
        ("schema_invalid_output", AIFailureCategory.SCHEMA_VALIDATION),
    ],
)
def test_fake_provider_validates_response_fixtures_locally(
    fixture_name: str,
    expected_category: AIFailureCategory,
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider = FakeAIProvider.from_file(FIXTURE_PATH, fixture_name)

    with pytest.raises(AIProviderExecutionError) as error_info:
        provider.generate(_summary_request())

    error = error_info.value
    assert error.details.category is expected_category
    assert error.details.audit is not None
    assert error.details.audit.raw_response is not None
    assert error.details.audit.raw_response not in str(error)
    assert error.details.audit.raw_response not in repr(error)
    assert error.details.audit.raw_response not in caplog.text
    assert "audit" not in error.details.model_dump()
    assert "raw_response" not in error.details.model_dump_json()


@pytest.mark.parametrize(
    ("fixture_name", "expected_category"),
    [
        (
            "simulated_recoverable_failure",
            AIFailureCategory.TRANSIENT_PROVIDER_FAILURE,
        ),
        ("simulated_non_recoverable_failure", AIFailureCategory.AUTHENTICATION),
    ],
)
def test_fake_provider_exposes_safe_simulated_failures(
    fixture_name: str,
    expected_category: AIFailureCategory,
) -> None:
    provider = FakeAIProvider.from_file(FIXTURE_PATH, fixture_name)

    with pytest.raises(AIProviderExecutionError) as error_info:
        provider.generate(_summary_request())

    error = error_info.value
    assert error.details.category is expected_category
    assert error.details.request_identifier == "req-001"
    assert error.details.audit is not None
    assert error.details.audit.raw_response is None
    assert "audit" not in repr(error)


def test_fake_provider_rejects_unknown_output_schema_before_fixture_execution() -> None:
    provider = FakeAIProvider.from_file(
        FIXTURE_PATH,
        "simulated_recoverable_failure",
    )
    request = _summary_request().model_copy(
        update={"output_schema": "unregistered-output-v1"}
    )

    with pytest.raises(AIProviderExecutionError) as error_info:
        provider.generate(request)

    error = error_info.value
    assert error.details.category is AIFailureCategory.UNSUPPORTED_OUTPUT_SCHEMA
    assert error.details.audit is not None
    assert error.details.audit.attempt_count == 1
    assert error.details.audit.raw_response is None
    assert "unregistered-output-v1" not in str(error)
    assert "unregistered-output-v1" not in repr(error)


def test_fake_provider_requires_no_gemini_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    provider = FakeAIProvider.from_file(FIXTURE_PATH, "valid_summary")

    result = provider.generate(_summary_request())

    assert isinstance(result.output, SummaryOutputV1)
