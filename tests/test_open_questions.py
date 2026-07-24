"""Focused tests for deterministic open-question source traceability."""

import json
from pathlib import Path

import pytest

from app.models import EvidenceType
from app.schemas.ai_outputs import (
    CriticOutputV1,
    HypothesesOutputV1,
    OpenQuestionSourceKind,
    OpenQuestionsOutputV1,
    OpenQuestionV1,
    ReasoningRisksOutputV1,
    SummaryOutputV1,
    TimelineOutputV1,
)
from app.schemas.ai_provider import (
    AIFailureCategory,
    BiasContextV1,
    CriticContextV1,
    OpenQuestionsContextV1,
)
from app.schemas.evidence import EvidenceManifest, EvidenceManifestSource
from app.services.analysis_stage_runner import (
    AnalysisStageOutputError,
    AnalysisStageRunner,
)
from app.services.ai_provider import process_structured_response
from app.services.evidence_manifest_service import EvidenceManifestService
from app.services.open_question_source_service import OpenQuestionSourceService
from app.services.validation_service import ValidationService


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "fake_ai_responses.json"


def _manifest() -> EvidenceManifest:
    return EvidenceManifestService.build_evidence_manifest(
        "INC-000001",
        (
            EvidenceManifestSource(
                evidence_code="E-001",
                source_name="checkout.log",
                evidence_type=EvidenceType.APPLICATION_LOG,
                original_text="api_key=local-secret\ncheckout failed",
            ),
        ),
    )


def _open_questions_context() -> OpenQuestionsContextV1:
    fixture_bank = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    summary = SummaryOutputV1.model_validate_json(
        fixture_bank["valid_summary"]["raw_response"]
    )
    timeline = TimelineOutputV1.model_validate_json(
        fixture_bank["valid_timeline"]["raw_response"]
    )
    hypothesis_data = json.loads(fixture_bank["valid_hypotheses"]["raw_response"])
    hypothesis_data["hypotheses"][0]["contradicting_evidence"] = [
        {
            "reference": {"evidence_id": "E-001", "line_range": "1"},
            "relevance": "The first line weakens the hypothesis.",
        },
        {
            "reference": {"evidence_id": "E-001", "line_range": "2"},
            "relevance": "The second line independently weakens the hypothesis.",
        },
    ]
    hypotheses = HypothesesOutputV1.model_validate(hypothesis_data)
    original_analysis = CriticContextV1(
        summary=summary,
        timeline=timeline,
        hypotheses=hypotheses,
    )
    analysis_context = BiasContextV1(
        original_analysis=original_analysis,
        validated_analysis=ValidationService.build_validated_analysis_view(
            summary,
            timeline,
            hypotheses,
            _manifest(),
        ),
        critic=CriticOutputV1.model_validate_json(
            fixture_bank["valid_critic"]["raw_response"]
        ),
    )
    return OpenQuestionsContextV1(
        analysis_context=analysis_context,
        reasoning_risks=ReasoningRisksOutputV1.model_validate_json(
            fixture_bank["valid_bias"]["raw_response"]
        ),
    )


def _contradiction_question(source_reference: str) -> OpenQuestionV1:
    return OpenQuestionV1(
        question="What explains the contradicting observation?",
        source_kind=OpenQuestionSourceKind.CONTRADICTION,
        source_reference=source_reference,
        rationale="The contradiction must be resolved before confidence increases.",
        evidence_needed=("A time-aligned comparison of the conflicting signals",),
        resolution_criteria="The comparison explains whether the contradiction holds.",
    )


def test_real_contradictions_have_distinct_accepted_source_references() -> None:
    context = _open_questions_context()
    hypothesis = context.analysis_context.validated_analysis.hypotheses[0]
    source_references = tuple(
        AnalysisStageRunner.build_contradiction_source_reference(
            hypothesis.hypothesis_id,
            evidence.reference.reference.evidence_id,
            evidence.reference.reference.line_range,
        )
        for evidence in hypothesis.contradicting_evidence
    )

    AnalysisStageRunner.require_traceable_open_questions(
        OpenQuestionsOutputV1(
            questions=tuple(
                _contradiction_question(source_reference)
                for source_reference in source_references
            )
        ),
        context,
    )

    assert source_references == ("H-001|E-001|1", "H-001|E-001|2")
    assert len(set(source_references)) == 2


def test_stable_source_identifier_maps_to_exact_typed_source() -> None:
    context = _open_questions_context()
    source_options = OpenQuestionSourceService.build_source_options(context)
    selected_source = next(
        source
        for source in source_options
        if source.source_kind is OpenQuestionSourceKind.ASSUMPTION
    )
    raw_response = json.dumps(
        {
            "questions": [
                {
                    "question": "Did the possible deployment relationship occur?",
                    "source_id": selected_source.source_id,
                    "rationale": "The relationship remains an assumption.",
                    "evidence_needed": ["Deployment history"],
                    "resolution_criteria": (
                        "The deployment record establishes or rules out the "
                        "relationship."
                    ),
                }
            ]
        }
    )

    outcome = process_structured_response(
        raw_response,
        OpenQuestionsOutputV1,
        open_question_sources=source_options,
    )

    assert isinstance(outcome.output, OpenQuestionsOutputV1)
    question = outcome.output.questions[0]
    assert question.source_kind is selected_source.source_kind
    assert question.source_reference == selected_source.source_reference
    assert "source_id" not in outcome.output.model_dump_json()
    assert [source.source_id for source in source_options] == [
        f"S-{index:03d}" for index in range(1, len(source_options) + 1)
    ]
    AnalysisStageRunner.require_traceable_open_questions(outcome.output, context)


def test_invented_source_identifier_fails_before_public_validation() -> None:
    context = _open_questions_context()
    raw_response = json.dumps(
        {
            "questions": [
                {
                    "question": "What invented source should be investigated?",
                    "source_id": "S-999",
                    "rationale": "This source is not in the allowlist.",
                    "evidence_needed": ["A real traceable source"],
                    "resolution_criteria": "A valid source is selected.",
                }
            ]
        }
    )

    outcome = process_structured_response(
        raw_response,
        OpenQuestionsOutputV1,
        open_question_sources=OpenQuestionSourceService.build_source_options(context),
    )

    assert outcome.output is None
    assert outcome.failure_category is AIFailureCategory.SCHEMA_VALIDATION
    assert raw_response not in repr(outcome)


@pytest.mark.parametrize(
    "source_fields",
    (
        {"source_id": []},
        {"source_id": {}},
        {
            "source_kind": OpenQuestionSourceKind.ASSUMPTION.value,
            "source_reference": [],
        },
    ),
    ids=("list-source-id", "object-source-id", "unhashable-legacy-reference"),
)
def test_malformed_source_values_fail_schema_validation_safely(
    source_fields: dict[str, object],
) -> None:
    context = _open_questions_context()
    response_data = {
        "questions": [
            {
                "question": "What malformed source should be investigated?",
                **source_fields,
                "rationale": "This malformed source must fail closed.",
                "evidence_needed": ["A valid traceable source"],
                "resolution_criteria": "A valid source is selected.",
            }
        ]
    }
    raw_response = json.dumps(response_data)

    outcome = process_structured_response(
        raw_response,
        OpenQuestionsOutputV1,
        open_question_sources=OpenQuestionSourceService.build_source_options(context),
    )

    assert outcome.output is None
    assert outcome.failure_category is AIFailureCategory.SCHEMA_VALIDATION
    assert raw_response not in repr(outcome)


def test_paraphrased_assumption_source_is_rejected_safely() -> None:
    context = _open_questions_context()
    output = OpenQuestionsOutputV1(
        questions=(
            OpenQuestionV1(
                question="Was a deployment related?",
                source_kind=OpenQuestionSourceKind.ASSUMPTION,
                source_reference="A deployment might be related.",
                rationale="The relationship remains unverified.",
                evidence_needed=("Deployment history",),
                resolution_criteria="The history confirms or rules out a relationship.",
            ),
        )
    )

    with pytest.raises(
        AnalysisStageOutputError,
        match="untraceable analysis source",
    ):
        AnalysisStageRunner.require_traceable_open_questions(output, context)


@pytest.mark.parametrize(
    "source_reference",
    ("H-001|E-999|1", "H-001|E-001|99"),
)
def test_invented_contradiction_reference_is_rejected_safely(
    source_reference: str,
) -> None:
    raw_response = "internal raw response containing checkout failed"

    with pytest.raises(
        AnalysisStageOutputError,
        match="untraceable analysis source",
    ) as error_info:
        AnalysisStageRunner.require_traceable_open_questions(
            OpenQuestionsOutputV1(
                questions=(_contradiction_question(source_reference),)
            ),
            _open_questions_context(),
            raw_response=raw_response,
        )

    assert raw_response not in str(error_info.value)
    assert raw_response not in repr(error_info.value)
    assert "checkout failed" not in str(error_info.value)
    assert "checkout failed" not in repr(error_info.value)
    assert source_reference not in str(error_info.value)
    assert source_reference not in repr(error_info.value)
    assert error_info.value.audit_raw_response == raw_response
