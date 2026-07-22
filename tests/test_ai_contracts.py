"""Focused tests for provider-neutral AI request and output contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.models.enums import EvidenceType
from app.schemas.ai_outputs import (
    ActionPriority,
    ActionsOutputV1,
    AssumptionItemV1,
    CompleteAnalysisOutputV1,
    ContradictingEvidenceV1,
    CriticFindingV1,
    CriticOutputV1,
    EvidenceReferenceV1,
    FactItemV1,
    HypothesesOutputV1,
    HypothesisV1,
    HypothesisValidationTestV1,
    ReasoningRiskV1,
    RecommendedActionV1,
    SummaryAndImpactV1,
    SummaryOutputV1,
    SupportingEvidenceV1,
    TimelineEventV1,
    TimelineOutputV1,
)
from app.schemas.ai_provider import (
    AIFailureCategory,
    AIFailureDetails,
    AIRequest,
    AIResult,
    AIResultMetadata,
    AnalysisStage,
    BiasContextV1,
    CriticContextV1,
    FailureAuditData,
    OutputSchemaIdentifier,
    PromptBundle,
    PromptName,
    PromptReference,
    PromptVersion,
    SafeAIMetadata,
    SuccessAuditData,
)
from app.schemas.evidence import (
    EvidenceManifest,
    EvidenceManifestChunk,
    EvidenceManifestItem,
    EvidenceManifestTimestamp,
)
from app.services.validation_service import ValidationService


def _manifest() -> EvidenceManifest:
    return EvidenceManifest(
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
                        content=(
                            "L0001: checkout failed\nL0002: api_key=[REDACTED_API_KEY]"
                        ),
                    ),
                ),
            ),
        ),
    )


def _prompt_bundle() -> PromptBundle:
    return PromptBundle(
        system=PromptReference(
            name=PromptName.SYSTEM,
            version=PromptVersion.V1,
        ),
        task=PromptReference(
            name=PromptName.SUMMARY,
            version=PromptVersion.V1,
        ),
    )


def _metadata() -> SafeAIMetadata:
    return SafeAIMetadata(
        request_identifier="req-001",
        incident_public_identifier="INC-000001",
        analysis_stage=AnalysisStage.SUMMARY,
        evidence_manifest_checksum="a" * 64,
    )


def _reference() -> EvidenceReferenceV1:
    return EvidenceReferenceV1(
        evidence_id="E-001",
        line_range="1-2",
        excerpt="checkout failed",
    )


def _summary() -> SummaryOutputV1:
    return SummaryOutputV1(
        summary=SummaryAndImpactV1(
            text="Checkout requests are failing.",
            impact="Some customers cannot complete checkout.",
            uncertainty="The frequency and root cause are not yet verified.",
        ),
        facts=(
            FactItemV1(
                claim="The checkout log contains a failure.",
                evidence=(_reference(),),
                confidence=90,
            ),
        ),
        assumptions=(
            AssumptionItemV1(
                claim="The deployment may be related.",
                reason="The available evidence is incomplete.",
                required_evidence=("Compare pre-deployment behavior.",),
            ),
        ),
        unknowns=("The failure rate is unknown.",),
    )


def _hypothesis(index: int) -> HypothesisV1:
    return HypothesisV1(
        hypothesis_id=f"H-{index:03d}",
        rank=index,
        title=f"Possible cause {index}",
        explanation="This explanation is plausible but unconfirmed.",
        confidence=70 - index,
        supporting_evidence=(
            SupportingEvidenceV1(
                reference=_reference(),
                relevance="The failure is consistent with this explanation.",
            ),
        ),
        contradicting_evidence=(
            ContradictingEvidenceV1(
                reference=_reference(),
                relevance="The same evidence does not establish causation.",
            ),
        ),
        missing_evidence=("A controlled comparison is missing.",),
        validation_test=HypothesisValidationTestV1(
            description="Compare the configuration before and after deployment.",
            expected_if_true="The changed value reproduces the failure.",
            expected_if_false="The failure persists without the changed value.",
        ),
        risk_of_acting="A premature rollback could disrupt recovery.",
    )


def _critic_context() -> CriticContextV1:
    return CriticContextV1(
        summary=_summary(),
        timeline=TimelineOutputV1(
            events=(
                TimelineEventV1(
                    timestamp="time unknown",
                    description="Checkout failed.",
                    evidence=(_reference(),),
                    is_inferred=False,
                    confidence=90,
                ),
            ),
        ),
        hypotheses=HypothesesOutputV1(
            hypotheses=(_hypothesis(1), _hypothesis(2), _hypothesis(3)),
        ),
    )


def _result_metadata() -> AIResultMetadata:
    return AIResultMetadata(
        provider_name="fake",
        model_name="fixture-v1",
        system_prompt=_prompt_bundle().system,
        task_prompt=_prompt_bundle().task,
        analysis_stage=AnalysisStage.SUMMARY,
        output_schema=OutputSchemaIdentifier.SUMMARY_V1,
        request_identifier="req-001",
        attempt_count=1,
    )


def _complete_analysis_for_ranking(
    hypotheses: tuple[HypothesisV1, ...],
) -> CompleteAnalysisOutputV1:
    critic = CriticOutputV1(
        findings=(
            CriticFindingV1(
                concern="A claim may be overstated.",
                affected_claim="Top hypothesis",
                evidence=(),
                impact="The investigation may anchor too early.",
                recommendation="Run the validation test.",
            ),
        ),
        ignored_evidence=(),
        alternative_hypothesis=None,
        ranking_rationale="Multiple explanations remain possible.",
    )
    return CompleteAnalysisOutputV1(
        summary=_summary().summary,
        facts=(),
        assumptions=(),
        timeline=(),
        hypotheses=hypotheses,
        actions=(),
        open_questions=(),
        reasoning_risks=(),
        critic=critic,
    )


def _validate_hypothesis_container(
    container: str,
    hypotheses: tuple[HypothesisV1, ...],
) -> object:
    if container == "stage":
        return HypothesesOutputV1(hypotheses=hypotheses)
    return _complete_analysis_for_ranking(hypotheses)


def test_ai_request_accepts_only_typed_redacted_input() -> None:
    request = AIRequest(
        evidence_manifest=_manifest(),
        prompts=_prompt_bundle(),
        output_schema=OutputSchemaIdentifier.SUMMARY_V1,
        metadata=_metadata(),
    )

    serialized = request.model_dump_json()

    assert "[REDACTED_API_KEY]" in serialized
    assert "original_text" not in serialized
    assert "user_prompt" not in serialized


def test_critic_request_requires_typed_initial_analysis_context() -> None:
    request = AIRequest(
        evidence_manifest=_manifest(),
        prompts=PromptBundle(
            system=_prompt_bundle().system,
            task=PromptReference(name=PromptName.CRITIC, version=PromptVersion.V1),
        ),
        output_schema=OutputSchemaIdentifier.CRITIC_V1,
        metadata=SafeAIMetadata(
            request_identifier="req-critic",
            incident_public_identifier="INC-000001",
            analysis_stage=AnalysisStage.CRITIC,
            evidence_manifest_checksum="a" * 64,
        ),
        critic_context=_critic_context(),
    )

    serialized = request.model_dump_json()

    assert request.critic_context.hypotheses.hypotheses[0].title == "Possible cause 1"
    assert '"critic_context"' in serialized
    assert '"raw_response"' not in serialized
    assert '"audit"' not in serialized


def test_critic_request_rejects_missing_initial_analysis_context() -> None:
    with pytest.raises(ValidationError, match="require validated initial analysis"):
        AIRequest(
            evidence_manifest=_manifest(),
            prompts=PromptBundle(
                system=_prompt_bundle().system,
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
            ),
        )


def test_non_critic_request_rejects_critic_context() -> None:
    with pytest.raises(ValidationError, match="only accepted for critic requests"):
        AIRequest(
            evidence_manifest=_manifest(),
            prompts=_prompt_bundle(),
            output_schema=OutputSchemaIdentifier.SUMMARY_V1,
            metadata=_metadata(),
            critic_context=_critic_context(),
        )


def test_bias_request_requires_typed_analysis_and_critic_context() -> None:
    hypotheses = (_hypothesis(1), _hypothesis(2), _hypothesis(3))
    critic = _complete_analysis_for_ranking(hypotheses).critic
    original_analysis = _critic_context()
    bias_context = BiasContextV1(
        original_analysis=original_analysis,
        validated_analysis=ValidationService.build_validated_analysis_view(
            original_analysis.summary,
            original_analysis.timeline,
            original_analysis.hypotheses,
            _manifest(),
        ),
        critic=critic,
    )

    request = AIRequest(
        evidence_manifest=_manifest(),
        prompts=PromptBundle(
            system=_prompt_bundle().system,
            task=PromptReference(name=PromptName.BIAS, version=PromptVersion.V1),
        ),
        output_schema=OutputSchemaIdentifier.REASONING_RISKS_V1,
        metadata=SafeAIMetadata(
            request_identifier="req-bias",
            incident_public_identifier="INC-000001",
            analysis_stage=AnalysisStage.BIAS,
        ),
        bias_context=bias_context,
    )

    serialized = request.model_dump_json()

    assert request.bias_context == bias_context
    assert '"bias_context"' in serialized
    assert '"raw_response"' not in serialized
    assert '"audit"' not in serialized


def test_bias_request_rejects_missing_analysis_context() -> None:
    with pytest.raises(ValidationError, match="bias requests require"):
        AIRequest(
            evidence_manifest=_manifest(),
            prompts=PromptBundle(
                system=_prompt_bundle().system,
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
            ),
        )


@pytest.mark.parametrize(
    "forbidden_field",
    ["original_text", "unredacted_evidence", "user_prompt", "api_key"],
)
def test_ai_request_rejects_forbidden_or_extra_fields(
    forbidden_field: str,
) -> None:
    request_data = {
        "evidence_manifest": _manifest(),
        "prompts": _prompt_bundle(),
        "output_schema": OutputSchemaIdentifier.SUMMARY_V1,
        "metadata": _metadata(),
        forbidden_field: "must not cross the provider boundary",
    }

    with pytest.raises(ValidationError):
        AIRequest.model_validate(request_data)


def test_prompt_reference_is_allowlisted_and_rejects_paths() -> None:
    with pytest.raises(ValidationError):
        PromptReference(name="arbitrary", version="v1")

    with pytest.raises(ValidationError):
        PromptReference.model_validate(
            {
                "name": "summary",
                "version": "v1",
                "file_path": "C:/private/prompt.txt",
            }
        )


def test_prompt_bundle_rejects_reversed_system_and_task_references() -> None:
    with pytest.raises(ValidationError, match="system prompt must reference"):
        PromptBundle(
            system=PromptReference(name=PromptName.SUMMARY, version="v1"),
            task=PromptReference(name=PromptName.SYSTEM, version="v1"),
        )


def test_prompt_bundle_rejects_system_prompt_as_task() -> None:
    with pytest.raises(ValidationError, match="task prompt must not reference"):
        PromptBundle(
            system=PromptReference(name=PromptName.SYSTEM, version="v1"),
            task=PromptReference(name=PromptName.SYSTEM, version="v1"),
        )


def test_safe_metadata_rejects_arbitrary_or_sensitive_values() -> None:
    with pytest.raises(ValidationError):
        SafeAIMetadata.model_validate(
            {
                **_metadata().model_dump(),
                "arbitrary_user_data": "secret",
            }
        )


def test_safe_metadata_rejects_system_only_request_stage() -> None:
    with pytest.raises(ValidationError, match="not a requestable analysis stage"):
        SafeAIMetadata(
            request_identifier="req-001",
            incident_public_identifier="INC-000001",
            analysis_stage=AnalysisStage.SYSTEM,
        )


def test_result_metadata_rejects_system_only_analysis_stage() -> None:
    metadata = _result_metadata().model_dump()
    metadata["analysis_stage"] = AnalysisStage.SYSTEM

    with pytest.raises(ValidationError, match="not a requestable analysis stage"):
        AIResultMetadata.model_validate(metadata)


def test_provider_call_output_schema_identifiers_are_stage_only() -> None:
    assert {identifier.value for identifier in OutputSchemaIdentifier} == {
        "summary_v1",
        "timeline_v1",
        "hypotheses_v1",
        "critic_v1",
        "reasoning_risks_v1",
    }


@pytest.mark.parametrize("confidence", [-1, 101, "90"])
def test_confidence_is_strictly_bounded(confidence: object) -> None:
    with pytest.raises(ValidationError):
        FactItemV1(
            claim="A claim.",
            evidence=(_reference(),),
            confidence=confidence,
        )


def test_outputs_reject_blank_required_text_and_extra_fields() -> None:
    with pytest.raises(ValidationError):
        SummaryAndImpactV1(
            text="   ",
            impact="Known impact.",
            uncertainty="Known uncertainty.",
        )

    with pytest.raises(ValidationError):
        SummaryAndImpactV1.model_validate(
            {
                "text": "Summary.",
                "impact": "Impact.",
                "uncertainty": "Uncertainty.",
                "provider_field": "not allowed",
            }
        )


def test_inferred_timeline_event_requires_uncertainty_explanation() -> None:
    with pytest.raises(ValidationError):
        TimelineEventV1(
            timestamp="time unknown",
            description="A failure may have started.",
            evidence=(_reference(),),
            is_inferred=True,
            confidence=60,
        )


def test_inferred_timeline_event_accepts_confidence_of_70() -> None:
    event = TimelineEventV1(
        timestamp="time unknown",
        description="A failure may have started.",
        evidence=(_reference(),),
        is_inferred=True,
        confidence=70,
        uncertainty_explanation="The evidence has no direct timestamp.",
    )

    assert event.confidence == 70


def test_inferred_timeline_event_retains_provider_confidence_above_cap() -> None:
    event = TimelineEventV1(
        timestamp="time unknown",
        description="A failure may have started.",
        evidence=(_reference(),),
        is_inferred=True,
        confidence=95,
        uncertainty_explanation="The evidence has no direct timestamp.",
    )

    assert event.confidence == 95


def test_hypothesis_requires_a_typed_validation_test() -> None:
    hypothesis_data = _hypothesis(1).model_dump()
    hypothesis_data.pop("validation_test")

    with pytest.raises(ValidationError):
        HypothesisV1.model_validate(hypothesis_data)


def test_hypotheses_output_requires_three_candidates() -> None:
    with pytest.raises(ValidationError):
        HypothesesOutputV1(hypotheses=(_hypothesis(1), _hypothesis(2)))


@pytest.mark.parametrize("container", ["stage", "complete"])
def test_hypothesis_collections_reject_duplicate_identifiers(
    container: str,
) -> None:
    hypotheses = (
        _hypothesis(1),
        _hypothesis(2).model_copy(update={"hypothesis_id": "H-001"}),
        _hypothesis(3),
    )

    with pytest.raises(ValidationError, match="identifiers must be unique"):
        _validate_hypothesis_container(container, hypotheses)


@pytest.mark.parametrize("container", ["stage", "complete"])
def test_hypothesis_collections_reject_duplicate_ranks(container: str) -> None:
    hypotheses = (
        _hypothesis(1),
        _hypothesis(2).model_copy(update={"rank": 1}),
        _hypothesis(3),
    )

    with pytest.raises(ValidationError, match="ranks must be unique"):
        _validate_hypothesis_container(container, hypotheses)


@pytest.mark.parametrize("container", ["stage", "complete"])
def test_hypothesis_collections_reject_non_contiguous_ranks(
    container: str,
) -> None:
    hypotheses = (
        _hypothesis(1),
        _hypothesis(2),
        _hypothesis(3).model_copy(update={"rank": 4}),
    )

    with pytest.raises(ValidationError, match="contiguous sequence 1..N"):
        _validate_hypothesis_container(container, hypotheses)


def test_hypothesis_ranks_do_not_require_confidence_ordering() -> None:
    hypotheses = (
        _hypothesis(1).model_copy(update={"confidence": 20}),
        _hypothesis(2).model_copy(update={"confidence": 90}),
        _hypothesis(3).model_copy(update={"confidence": 50}),
    )

    output = HypothesesOutputV1(hypotheses=hypotheses)

    assert [hypothesis.confidence for hypothesis in output.hypotheses] == [
        20,
        90,
        50,
    ]


def test_complete_analysis_composes_smaller_typed_outputs() -> None:
    hypotheses = tuple(_hypothesis(index) for index in range(1, 4))
    critic = CriticOutputV1(
        findings=(
            CriticFindingV1(
                concern="The strongest claim overstates causation.",
                affected_claim=hypotheses[0].title,
                evidence=(_reference(),),
                impact="The ranking may anchor the investigation.",
                recommendation="Run the declared validation test first.",
            ),
        ),
        ignored_evidence=(_reference(),),
        alternative_hypothesis=hypotheses[1],
        ranking_rationale="The alternative explains the same evidence.",
    )
    complete = CompleteAnalysisOutputV1(
        summary=_summary().summary,
        facts=_summary().facts,
        assumptions=_summary().assumptions,
        timeline=(
            TimelineEventV1(
                timestamp="time unknown",
                description="Checkout failures were reported.",
                evidence=(_reference(),),
                is_inferred=False,
                confidence=80,
            ),
        ),
        hypotheses=hypotheses,
        actions=(
            RecommendedActionV1(
                description="Compare deployment configuration values.",
                priority=ActionPriority.HIGH,
                linked_hypothesis_ids=("H-001",),
                evidence=(_reference(),),
                owner_role="Incident investigator",
                expected_information="Whether configuration changed.",
                operational_risk="Read-only comparison has low risk.",
            ),
        ),
        open_questions=("What changed in the deployment?",),
        reasoning_risks=(
            ReasoningRiskV1(
                name="Anchoring bias",
                location="Top hypothesis",
                trigger="The deployment occurred before the report.",
                potential_effect="Other causes may receive too little attention.",
                mitigation="Compare multiple materially different hypotheses.",
                confidence=60,
            ),
        ),
        critic=critic,
    )

    assert isinstance(complete.facts[0], FactItemV1)
    assert isinstance(
        complete.hypotheses[0].validation_test, HypothesisValidationTestV1
    )
    assert isinstance(complete.critic, CriticOutputV1)
    assert isinstance(ActionsOutputV1(actions=complete.actions), ActionsOutputV1)


@pytest.mark.parametrize("field_name", ["provider_name", "model_name"])
def test_result_metadata_rejects_names_with_newlines(field_name: str) -> None:
    metadata = _result_metadata().model_dump()
    metadata[field_name] = "gemini\nforged-log-entry"

    with pytest.raises(ValidationError, match="log-safe characters"):
        AIResultMetadata.model_validate(metadata)


def test_result_metadata_accepts_expected_log_safe_names() -> None:
    metadata_data = _result_metadata().model_dump()
    metadata_data.update(
        provider_name="gemini",
        model_name="gemini-2.5-flash",
    )
    metadata = AIResultMetadata.model_validate(metadata_data)

    assert metadata.provider_name == "gemini"
    assert metadata.model_name == "gemini-2.5-flash"


def test_ai_result_keeps_success_raw_response_internal() -> None:
    raw_response = '{"summary":"sensitive provider response"}'
    result = AIResult[SummaryOutputV1](
        output=_summary(),
        metadata=_result_metadata(),
        audit=SuccessAuditData(raw_response=raw_response),
    )

    assert result.audit.raw_response == raw_response
    assert raw_response not in repr(result)
    assert raw_response not in repr(result.audit)
    assert "raw_response" not in result.audit.model_dump()
    assert "audit" not in result.model_dump()
    assert raw_response not in result.model_dump_json()


def test_ai_result_parses_output_into_declared_pydantic_type() -> None:
    result = AIResult[SummaryOutputV1].model_validate(
        {
            "output": _summary().model_dump(),
            "metadata": _result_metadata().model_dump(),
            "audit": {"raw_response": "raw"},
        }
    )

    assert isinstance(result.output, SummaryOutputV1)
    assert not isinstance(result.output, dict)


def test_failure_details_keep_raw_response_out_of_public_surfaces() -> None:
    raw_response = "sensitive malformed response"
    audit = FailureAuditData(
        request_identifier="req-001",
        attempt_count=2,
        raw_response=raw_response,
    )
    failure = AIFailureDetails(
        category=AIFailureCategory.SCHEMA_VALIDATION,
        request_identifier="req-001",
        explanation="The provider response did not match the requested schema.",
        audit=audit,
    )

    assert failure.audit is not None
    assert failure.audit.raw_response == raw_response
    assert raw_response not in str(failure)
    assert raw_response not in repr(failure)
    assert raw_response not in repr(audit)
    assert "raw_response" not in audit.model_dump()
    assert "audit" not in failure.model_dump()
    assert raw_response not in failure.model_dump_json()
