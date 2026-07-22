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
    BiasContextV1,
    CriticContextV1,
    OpenQuestionsContextV1,
)
from app.schemas.evidence import EvidenceManifest, EvidenceManifestSource
from app.services.analysis_stage_runner import (
    AnalysisStageOutputError,
    AnalysisStageRunner,
)
from app.services.evidence_manifest_service import EvidenceManifestService
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
