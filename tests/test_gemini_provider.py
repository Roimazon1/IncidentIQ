"""Network-free tests for the concrete Gemini Developer API adapter."""

from __future__ import annotations

import json
import socket
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict

import pytest

from app.config import Settings
from app.models.enums import EvidenceType
from app.schemas.ai_outputs import (
    CriticOutputV1,
    HypothesesOutputV1,
    OpenQuestionsOutputV1,
    ReasoningRisksOutputV1,
    SummaryOutputV1,
    TimelineOutputV1,
)
from app.schemas.ai_provider import (
    AIFailureCategory,
    AIRequest,
    AnalysisStage,
    BiasContextV1,
    CriticContextV1,
    OpenQuestionsContextV1,
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
from app.services.ai_provider import (
    AIProvider,
    AIProviderConfigurationError,
    AIProviderExecutionError,
)
from app.services.providers.gemini_provider import (
    GeminiClientProtocol,
    GeminiGenerateConfig,
    GeminiAIProvider,
    GeminiModelsProtocol,
    GeminiResponseProtocol,
)
from app.services.validation_service import ValidationService


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "fake_ai_responses.json"
SYSTEM_PROMPT = "Use only the supplied redacted evidence."
TASK_PROMPT = "Produce a neutral summary with cited facts and explicit uncertainty."
CRITIC_PROMPT = "Challenge the supplied validated initial analysis."
BIAS_PROMPT = "Assess possible reasoning risks in the validated analysis."
OPEN_QUESTIONS_PROMPT = "Identify actionable unresolved investigation questions."


@pytest.fixture(autouse=True)
def _enforce_offline_gemini_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)

    def reject_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Gemini provider tests must not access the network")

    monkeypatch.setattr(socket, "create_connection", reject_network)
    monkeypatch.setattr(socket.socket, "connect", reject_network)
    monkeypatch.setattr(socket.socket, "connect_ex", reject_network)


class _FakeResponse(GeminiResponseProtocol):
    def __init__(self, text: str | None) -> None:
        self._text = text

    @property
    def text(self) -> str | None:
        return self._text


class _RecordedCall(TypedDict):
    model: str
    contents: str
    config: GeminiGenerateConfig


_FakeOutcome = _FakeResponse | Exception


class _FakeModels(GeminiModelsProtocol):
    def __init__(self, outcomes: list[_FakeOutcome]) -> None:
        self._outcomes = outcomes
        self.calls: list[_RecordedCall] = []

    def generate_content(
        self,
        *,
        model: str,
        contents: str,
        config: GeminiGenerateConfig,
    ) -> GeminiResponseProtocol:
        self.calls.append(
            {
                "model": model,
                "contents": contents,
                "config": config,
            }
        )
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _FakeClient(GeminiClientProtocol):
    def __init__(self, outcomes: list[_FakeOutcome]) -> None:
        self._models = _FakeModels(outcomes)

    @property
    def models(self) -> GeminiModelsProtocol:
        return self._models

    @property
    def recorded_models(self) -> _FakeModels:
        return self._models


class _FakeGeminiError(Exception):
    def __init__(self, code: int, sensitive_message: str) -> None:
        self.code = code
        super().__init__(sensitive_message)


def _fixture_response(fixture_name: str) -> str:
    fixture_bank = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    return fixture_bank[fixture_name]["raw_response"]


def _valid_summary_response() -> str:
    return _fixture_response("valid_summary")


def _resolve_prompt(reference: PromptReference) -> str:
    if reference.name is PromptName.SYSTEM:
        return SYSTEM_PROMPT
    if reference.name is PromptName.SUMMARY:
        return TASK_PROMPT
    if reference.name is PromptName.CRITIC:
        return CRITIC_PROMPT
    if reference.name is PromptName.BIAS:
        return BIAS_PROMPT
    if reference.name is PromptName.OPEN_QUESTIONS:
        return OPEN_QUESTIONS_PROMPT
    raise LookupError(reference.name)


def _validate_prompt_bundle(
    bundle: PromptBundle,
    analysis_stage: AnalysisStage,
) -> None:
    expected_task = {
        AnalysisStage.SUMMARY: PromptName.SUMMARY,
        AnalysisStage.CRITIC: PromptName.CRITIC,
        AnalysisStage.BIAS: PromptName.BIAS,
        AnalysisStage.OPEN_QUESTIONS: PromptName.OPEN_QUESTIONS,
    }.get(analysis_stage)
    if (
        bundle.system.name is not PromptName.SYSTEM
        or bundle.system.version is not PromptVersion.V1
        or bundle.task.name is not expected_task
        or bundle.task.version is not PromptVersion.V1
    ):
        raise LookupError("unregistered prompt bundle")


def _summary_request() -> AIRequest:
    return AIRequest(
        evidence_manifest=EvidenceManifest(
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
        ),
        prompts=PromptBundle(
            system=PromptReference(
                name=PromptName.SYSTEM,
                version=PromptVersion.V1,
            ),
            task=PromptReference(
                name=PromptName.SUMMARY,
                version=PromptVersion.V1,
            ),
        ),
        output_schema=OutputSchemaIdentifier.SUMMARY_V1,
        metadata=SafeAIMetadata(
            request_identifier="req-001",
            incident_public_identifier="INC-000001",
            analysis_stage=AnalysisStage.SUMMARY,
            evidence_manifest_checksum="a" * 64,
        ),
    )


def _critic_request() -> AIRequest:
    fixture_bank = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    summary_request = _summary_request()
    return AIRequest(
        evidence_manifest=summary_request.evidence_manifest,
        prompts=PromptBundle(
            system=summary_request.prompts.system,
            task=PromptReference(
                name=PromptName.CRITIC,
                version=PromptVersion.V1,
            ),
        ),
        output_schema=OutputSchemaIdentifier.CRITIC_V1,
        metadata=SafeAIMetadata(
            request_identifier="req-critic",
            incident_public_identifier="INC-000001",
            analysis_stage=AnalysisStage.CRITIC,
            evidence_manifest_checksum="a" * 64,
        ),
        critic_context=CriticContextV1(
            summary=SummaryOutputV1.model_validate_json(
                fixture_bank["valid_summary"]["raw_response"]
            ),
            timeline=TimelineOutputV1.model_validate_json(
                fixture_bank["valid_timeline"]["raw_response"]
            ),
            hypotheses=HypothesesOutputV1.model_validate_json(
                fixture_bank["valid_hypotheses"]["raw_response"]
            ),
        ),
    )


def _bias_request() -> AIRequest:
    fixture_bank = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    critic_request = _critic_request()
    assert critic_request.critic_context is not None
    return AIRequest(
        evidence_manifest=critic_request.evidence_manifest,
        prompts=PromptBundle(
            system=critic_request.prompts.system,
            task=PromptReference(
                name=PromptName.BIAS,
                version=PromptVersion.V1,
            ),
        ),
        output_schema=OutputSchemaIdentifier.REASONING_RISKS_V1,
        metadata=SafeAIMetadata(
            request_identifier="req-bias",
            incident_public_identifier="INC-000001",
            analysis_stage=AnalysisStage.BIAS,
            evidence_manifest_checksum="a" * 64,
        ),
        bias_context=BiasContextV1(
            original_analysis=critic_request.critic_context,
            validated_analysis=ValidationService.build_validated_analysis_view(
                critic_request.critic_context.summary,
                critic_request.critic_context.timeline,
                critic_request.critic_context.hypotheses,
                critic_request.evidence_manifest,
            ),
            critic=CriticOutputV1.model_validate_json(
                fixture_bank["valid_critic"]["raw_response"]
            ),
        ),
    )


def _open_questions_request() -> AIRequest:
    fixture_bank = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    bias_request = _bias_request()
    assert bias_request.bias_context is not None
    return AIRequest(
        evidence_manifest=bias_request.evidence_manifest,
        prompts=PromptBundle(
            system=bias_request.prompts.system,
            task=PromptReference(
                name=PromptName.OPEN_QUESTIONS,
                version=PromptVersion.V1,
            ),
        ),
        output_schema=OutputSchemaIdentifier.OPEN_QUESTIONS_V1,
        metadata=SafeAIMetadata(
            request_identifier="req-open-questions",
            incident_public_identifier="INC-000001",
            analysis_stage=AnalysisStage.OPEN_QUESTIONS,
            evidence_manifest_checksum="a" * 64,
        ),
        open_questions_context=OpenQuestionsContextV1(
            analysis_context=bias_request.bias_context,
            reasoning_risks=ReasoningRisksOutputV1.model_validate_json(
                fixture_bank["valid_bias"]["raw_response"]
            ),
        ),
    )


def _provider(
    client: GeminiClientProtocol,
    *,
    max_attempts: int = 3,
    sleeper: Callable[[float], None] = lambda delay: None,
) -> GeminiAIProvider:
    return GeminiAIProvider(
        model_name="gemini-2.5-flash",
        prompt_resolver=_resolve_prompt,
        prompt_bundle_validator=_validate_prompt_bundle,
        client=client,
        max_attempts=max_attempts,
        retry_delay_seconds=0.5,
        sleeper=sleeper,
        api_error_type=_FakeGeminiError,
    )


def test_injected_fakes_satisfy_the_provider_protocols() -> None:
    response = _FakeResponse(_valid_summary_response())
    response_contract: GeminiResponseProtocol = response
    client = _FakeClient([response])
    client_contract: GeminiClientProtocol = client

    assert response_contract.text == response.text
    assert client_contract.models is client.recorded_models


def test_injected_client_receives_redacted_structured_request_only() -> None:
    raw_response = _valid_summary_response()
    client = _FakeClient([_FakeResponse(raw_response)])
    provider = _provider(client)
    request = _summary_request()

    result = provider.generate(request)

    assert isinstance(provider, AIProvider)
    assert isinstance(result.output, SummaryOutputV1)
    assert result.audit.raw_response == raw_response
    assert result.metadata.provider_name == "gemini"
    assert result.metadata.model_name == "gemini-2.5-flash"
    assert result.metadata.system_prompt == request.prompts.system
    assert result.metadata.task_prompt == request.prompts.task
    assert result.metadata.analysis_stage is AnalysisStage.SUMMARY
    assert result.metadata.output_schema is OutputSchemaIdentifier.SUMMARY_V1
    assert result.metadata.request_identifier == "req-001"
    assert result.metadata.attempt_count == 1
    assert "audit" not in result.model_dump()
    assert '"audit"' not in result.model_dump_json()
    assert '"raw_response"' not in result.model_dump_json()

    call = client.recorded_models.calls[0]
    payload = json.loads(call["contents"])
    assert call["model"] == "gemini-2.5-flash"
    assert set(payload) == {
        "task_prompt",
        "evidence_manifest",
        "output_schema",
        "metadata",
    }
    assert payload["task_prompt"] == TASK_PROMPT
    assert (
        payload["evidence_manifest"]["evidence"][0]["chunks"][0]["content"]
        == "[REDACTED_API_KEY] checkout failed"
    )
    assert payload["metadata"] == {
        "analysis_stage": "summary",
        "evidence_manifest_checksum": "a" * 64,
        "incident_public_identifier": "INC-000001",
        "request_identifier": "req-001",
    }
    serialized_payload = call["contents"]
    assert "original_text" not in serialized_payload
    assert "unredacted" not in serialized_payload
    assert "GEMINI_API_KEY" not in serialized_payload

    config = call["config"]
    assert config["system_instruction"] == SYSTEM_PROMPT
    assert config["response_mime_type"] == "application/json"
    response_schema = config["response_json_schema"]
    assert isinstance(response_schema, dict)
    assert response_schema["type"] == "object"
    assert "properties" in response_schema


def test_critic_payload_contains_validated_initial_analysis_without_audit() -> None:
    client = _FakeClient([_FakeResponse(_fixture_response("valid_critic"))])
    provider = _provider(client)

    result = provider.generate(_critic_request())

    assert isinstance(result.output, CriticOutputV1)
    payload = json.loads(client.recorded_models.calls[0]["contents"])
    assert payload["task_prompt"] == CRITIC_PROMPT
    assert payload["critic_context"]["summary"]["summary"]["text"] == (
        "Checkout requests are failing."
    )
    assert payload["critic_context"]["hypotheses"]["hypotheses"][0]["title"] == (
        "Database connection pool exhaustion"
    )
    assert "critic_context" not in payload["evidence_manifest"]
    assert "raw_response" not in client.recorded_models.calls[0]["contents"]
    assert "audit" not in client.recorded_models.calls[0]["contents"]


def test_bias_payload_contains_validated_analysis_and_critic_without_audit() -> None:
    client = _FakeClient([_FakeResponse(_fixture_response("valid_bias"))])
    provider = _provider(client)

    result = provider.generate(_bias_request())

    assert isinstance(result.output, ReasoningRisksOutputV1)
    payload = json.loads(client.recorded_models.calls[0]["contents"])
    assert payload["task_prompt"] == BIAS_PROMPT
    assert (
        payload["bias_context"]["original_analysis"]["summary"]["summary"]["text"]
        == "Checkout requests are failing."
    )
    assert (
        payload["bias_context"]["validated_analysis"]["facts"][0]["support_status"]
        == "UNSUPPORTED"
    )
    assert payload["bias_context"]["critic"]["findings"][0]["affected_claim"] == (
        "Database connection pool exhaustion"
    )
    assert "bias_context" not in payload["evidence_manifest"]
    assert "raw_response" not in client.recorded_models.calls[0]["contents"]
    assert "audit" not in client.recorded_models.calls[0]["contents"]


def test_open_questions_payload_reuses_validated_reasoning_context() -> None:
    client = _FakeClient([_FakeResponse(_fixture_response("valid_open_questions"))])
    provider = _provider(client)

    result = provider.generate(_open_questions_request())

    assert isinstance(result.output, OpenQuestionsOutputV1)
    payload_text = client.recorded_models.calls[0]["contents"]
    payload = json.loads(payload_text)
    assert payload["task_prompt"] == OPEN_QUESTIONS_PROMPT
    context = payload["open_questions_context"]
    assert (
        context["analysis_context"]["validated_analysis"]["hypotheses"][0][
            "hypothesis_id"
        ]
        == "H-001"
    )
    assert (
        context["analysis_context"]["critic"]["findings"][0]["affected_claim"]
        == "Database connection pool exhaustion"
    )
    assert context["reasoning_risks"]["risks"][0]["name"] == "Confirmation bias"
    assert "raw_response" not in payload_text
    assert "audit" not in payload_text
    assert "GEMINI_API_KEY" not in payload_text


def test_schema_sent_to_gemini_uses_only_supported_keywords() -> None:
    client = _FakeClient([_FakeResponse(_valid_summary_response())])
    provider = _provider(client)

    provider.generate(_summary_request())

    schema_text = json.dumps(
        client.recorded_models.calls[0]["config"]["response_json_schema"]
    )
    assert "minLength" not in schema_text
    assert "maxLength" not in schema_text
    assert "pattern" not in schema_text


def test_schema_invalid_response_retries_without_changing_request() -> None:
    invalid_response = '{"summary":{"unexpected":"sensitive raw response"}}'
    valid_response = _valid_summary_response()
    client = _FakeClient(
        [_FakeResponse(invalid_response), _FakeResponse(valid_response)]
    )
    delays: list[float] = []
    provider = _provider(client, max_attempts=2, sleeper=delays.append)

    result = provider.generate(_summary_request())

    assert result.metadata.attempt_count == 2
    assert delays == [0.5]
    assert len(client.recorded_models.calls) == 2
    assert (
        client.recorded_models.calls[0]["contents"]
        == client.recorded_models.calls[1]["contents"]
    )
    assert invalid_response not in client.recorded_models.calls[1]["contents"]


@pytest.mark.parametrize(
    "fixture_name",
    ["missing_fields", "out_of_range_confidence", "schema_invalid_output"],
    ids=["missing-fields", "out-of-range-confidence", "extra-fields"],
)
def test_schema_validation_failures_exhaust_retries_safely(
    fixture_name: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    raw_response = _fixture_response(fixture_name)
    client = _FakeClient([_FakeResponse(raw_response), _FakeResponse(raw_response)])
    delays: list[float] = []
    provider = _provider(client, max_attempts=2, sleeper=delays.append)

    with pytest.raises(AIProviderExecutionError) as error_info:
        provider.generate(_summary_request())

    error = error_info.value
    public_failure = f"{error!s}\n{error!r}\n{error.details.model_dump_json()}"
    assert len(client.recorded_models.calls) == 2
    assert delays == [0.5]
    assert error.details.category is AIFailureCategory.EXHAUSTED_RETRIES
    assert error.details.audit is not None
    assert error.details.audit.attempt_count == 2
    assert error.details.audit.raw_response == raw_response
    assert "audit" not in error.details.model_dump()
    assert raw_response not in public_failure
    assert raw_response not in caplog.text


def test_timeout_retries_and_then_succeeds() -> None:
    client = _FakeClient(
        [
            TimeoutError("sensitive timeout details"),
            _FakeResponse(_valid_summary_response()),
        ]
    )
    delays: list[float] = []
    provider = _provider(client, max_attempts=2, sleeper=delays.append)

    result = provider.generate(_summary_request())

    assert result.metadata.attempt_count == 2
    assert delays == [0.5]
    assert len(client.recorded_models.calls) == 2


def test_rate_limit_exhaustion_is_bounded_and_secret_safe(
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "secret-provider-error-api-key-fragment"
    client = _FakeClient([_FakeGeminiError(429, secret), _FakeGeminiError(429, secret)])
    delays: list[float] = []
    provider = _provider(client, max_attempts=2, sleeper=delays.append)

    with pytest.raises(AIProviderExecutionError) as error_info:
        provider.generate(_summary_request())

    error = error_info.value
    assert len(client.recorded_models.calls) == 2
    assert delays == [0.5]
    assert error.details.category is AIFailureCategory.EXHAUSTED_RETRIES
    assert error.details.audit is not None
    assert error.details.audit.attempt_count == 2
    assert secret not in str(error)
    assert secret not in repr(error)
    assert secret not in caplog.text
    assert error.__cause__ is None


def test_authentication_failure_is_not_retried() -> None:
    secret = "credential-rejected-secret-value"
    client = _FakeClient([_FakeGeminiError(401, secret)])
    provider = _provider(client, max_attempts=3)

    with pytest.raises(AIProviderExecutionError) as error_info:
        provider.generate(_summary_request())

    error = error_info.value
    assert len(client.recorded_models.calls) == 1
    assert error.details.category is AIFailureCategory.AUTHENTICATION
    assert secret not in str(error)
    assert secret not in repr(error)


def test_other_gemini_client_error_is_not_retried() -> None:
    client = _FakeClient([_FakeGeminiError(400, "sensitive invalid request")])
    delays: list[float] = []
    provider = _provider(client, max_attempts=3, sleeper=delays.append)

    with pytest.raises(AIProviderExecutionError) as error_info:
        provider.generate(_summary_request())

    assert len(client.recorded_models.calls) == 1
    assert delays == []
    assert error_info.value.details.category is (
        AIFailureCategory.UNSUPPORTED_OUTPUT_SCHEMA
    )


def test_unknown_local_output_schema_fails_before_client_call() -> None:
    client = _FakeClient([_FakeResponse(_valid_summary_response())])
    provider = _provider(client)
    request = _summary_request().model_copy(
        update={"output_schema": "unregistered-output-v1"}
    )

    with pytest.raises(AIProviderExecutionError) as error_info:
        provider.generate(request)

    error = error_info.value
    assert client.recorded_models.calls == []
    assert error.details.category is AIFailureCategory.UNSUPPORTED_OUTPUT_SCHEMA
    assert error.details.audit is not None
    assert error.details.audit.attempt_count == 1
    assert error.details.audit.raw_response is None
    assert "unregistered-output-v1" not in str(error)
    assert "unregistered-output-v1" not in repr(error)


def test_task_prompt_for_wrong_stage_fails_before_client_call() -> None:
    client = _FakeClient([_FakeResponse(_valid_summary_response())])
    provider = _provider(client)
    request = _summary_request().model_copy(
        update={
            "prompts": PromptBundle(
                system=PromptReference(
                    name=PromptName.SYSTEM,
                    version=PromptVersion.V1,
                ),
                task=PromptReference(
                    name=PromptName.TIMELINE,
                    version=PromptVersion.V1,
                ),
            )
        }
    )

    with pytest.raises(AIProviderExecutionError) as error_info:
        provider.generate(request)

    error = error_info.value
    assert client.recorded_models.calls == []
    assert error.details.category is AIFailureCategory.UNKNOWN_PROMPT
    assert error.details.audit is not None
    assert error.details.audit.attempt_count == 1
    assert error.details.audit.raw_response is None


def test_gemini_server_error_retries_and_then_succeeds() -> None:
    client = _FakeClient(
        [
            _FakeGeminiError(503, "sensitive server details"),
            _FakeResponse(_valid_summary_response()),
        ]
    )
    delays: list[float] = []
    provider = _provider(client, max_attempts=2, sleeper=delays.append)

    result = provider.generate(_summary_request())

    assert result.metadata.attempt_count == 2
    assert len(client.recorded_models.calls) == 2
    assert delays == [0.5]


def test_unknown_provider_exception_fails_once_without_exposing_details(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sensitive_text = "secret exception payload and api-key-fragment"
    client = _FakeClient([RuntimeError(sensitive_text)])
    delays: list[float] = []
    provider = _provider(client, max_attempts=3, sleeper=delays.append)

    with pytest.raises(AIProviderExecutionError) as error_info:
        provider.generate(_summary_request())

    error = error_info.value
    exposed_text = f"{error!s}\n{error!r}\n{error.details.model_dump_json()}"
    assert len(client.recorded_models.calls) == 1
    assert delays == []
    assert error.details.category is AIFailureCategory.TRANSIENT_PROVIDER_FAILURE
    assert error.details.audit is not None
    assert error.details.audit.attempt_count == 1
    assert error.details.audit.raw_response is None
    assert sensitive_text not in exposed_text
    assert SYSTEM_PROMPT not in exposed_text
    assert TASK_PROMPT not in exposed_text
    assert "[REDACTED_API_KEY]" not in exposed_text
    assert sensitive_text not in caplog.text
    assert error.__cause__ is None


def test_malformed_response_exhaustion_retains_internal_raw_audit(
    caplog: pytest.LogCaptureFixture,
) -> None:
    latest_raw_response = '{"second":"sensitive malformed response"'
    client = _FakeClient([_FakeResponse("{"), _FakeResponse(latest_raw_response)])
    provider = _provider(client, max_attempts=2)

    with pytest.raises(AIProviderExecutionError) as error_info:
        provider.generate(_summary_request())

    error = error_info.value
    assert error.details.category is AIFailureCategory.EXHAUSTED_RETRIES
    assert error.details.audit is not None
    assert error.details.audit.raw_response == latest_raw_response
    assert latest_raw_response not in str(error)
    assert latest_raw_response not in repr(error)
    assert latest_raw_response not in caplog.text
    assert "audit" not in error.details.model_dump()
    assert "raw_response" not in error.details.model_dump_json()


def test_injected_client_does_not_require_or_read_api_key() -> None:
    client = _FakeClient([_FakeResponse(_valid_summary_response())])
    settings = Settings.model_validate(
        {
            "ai_provider": "gemini",
            "gemini_api_key": None,
            "gemini_model": "gemini-2.5-flash",
        }
    )

    provider = GeminiAIProvider.from_settings(
        settings,
        prompt_resolver=_resolve_prompt,
        prompt_bundle_validator=_validate_prompt_bundle,
        client=client,
    )

    assert provider.generate(_summary_request()).metadata.model_name == (
        "gemini-2.5-flash"
    )


def test_real_client_is_constructed_only_without_injected_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeClient([_FakeResponse(_valid_summary_response())])
    received_keys: list[str | None] = []

    def create_client(api_key: str | None) -> _FakeClient:
        received_keys.append(api_key)
        return client

    monkeypatch.setattr(
        GeminiAIProvider,
        "_create_real_client",
        staticmethod(create_client),
    )
    settings = Settings.model_validate(
        {
            "ai_provider": "gemini",
            "gemini_api_key": "test-only-key",
            "gemini_model": "gemini-2.5-flash",
        }
    )

    provider = GeminiAIProvider.from_settings(
        settings,
        prompt_resolver=_resolve_prompt,
        prompt_bundle_validator=_validate_prompt_bundle,
    )

    assert received_keys == ["test-only-key"]
    assert provider.generate(_summary_request()).metadata.provider_name == "gemini"


@pytest.mark.parametrize("model_name", [None, "", "gemini\nforged-log-entry"])
def test_invalid_model_configuration_fails_before_client_call(
    model_name: str | None,
) -> None:
    client = _FakeClient([_FakeResponse(_valid_summary_response())])

    with pytest.raises(AIProviderConfigurationError) as error_info:
        GeminiAIProvider(
            model_name=model_name,
            prompt_resolver=_resolve_prompt,
            prompt_bundle_validator=_validate_prompt_bundle,
            client=client,
        )

    assert client.recorded_models.calls == []
    assert "forged-log-entry" not in str(error_info.value)
    assert "forged-log-entry" not in repr(error_info.value)


def test_missing_api_key_fails_before_real_client_construction() -> None:
    with pytest.raises(
        AIProviderConfigurationError,
        match="requires GEMINI_API_KEY",
    ):
        GeminiAIProvider(
            model_name="gemini-2.5-flash",
            prompt_resolver=_resolve_prompt,
            prompt_bundle_validator=_validate_prompt_bundle,
        )


def test_sdk_imports_remain_isolated_to_concrete_adapter() -> None:
    forbidden_imports = ("google.genai", "from google import genai")

    for python_path in Path("app").rglob("*.py"):
        if python_path.name == "gemini_provider.py":
            continue
        source = python_path.read_text(encoding="utf-8")
        assert all(term not in source for term in forbidden_imports), python_path
