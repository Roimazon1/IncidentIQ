"""Focused deterministic tests for the P10-03 prompt comparison."""

from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.models import EvidenceItem
from app.schemas.ai_outputs import HypothesesOutputV1
from app.schemas.ai_provider import (
    AIRequest,
    AIResult,
    AnalysisStage,
    EvidenceReferenceValidationStatus,
    PromptVersion,
)
from app.schemas.prompt_comparison import PromptComparisonVariantName
from app.services.ai_provider import AIProvider, build_ai_result
from app.services.analysis_service_factory import build_configured_ai_provider
from scripts.compare_prompt_versions import compare_prompt_versions
from scripts.seed_demo import seed_demo


DATASET_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "data" / "demo_checkout_incident"
)
PROVIDER_SECRET = "sk-comparisonsecret123"
EVIDENCE_SECRET = "sk-evidencesecret123"
RAW_RESPONSE_SENTINEL = "PROVIDER_RAW_RESPONSE_SENTINEL"


class _PromptAwareFakeProvider:
    """Delegate to the real fake boundary while recording prompt conditions."""

    def __init__(self, delegate: AIProvider) -> None:
        self._delegate = delegate
        self.requests: list[AIRequest] = []

    def generate(self, request: AIRequest) -> AIResult:
        self.requests.append(request)
        delegated_result = self._delegate.generate(request)
        output = delegated_result.output
        if (
            request.metadata.analysis_stage is AnalysisStage.HYPOTHESES
            and request.prompts.task.version is PromptVersion.V2
        ):
            assert isinstance(output, HypothesesOutputV1)
            by_id = {
                hypothesis.hypothesis_id: hypothesis for hypothesis in output.hypotheses
            }
            deployment = by_id["H-002"].model_copy(
                update={
                    "rank": 1,
                    "explanation": (
                        "The leading condition favors deployment v2.4.1, "
                        f"but api_key={PROVIDER_SECRET} must remain private."
                    ),
                }
            )
            database = by_id["H-001"].model_copy(update={"rank": 2})
            output = HypothesesOutputV1(
                hypotheses=(deployment, database, by_id["H-003"])
            )
        return build_ai_result(
            request=request,
            output=output,
            provider_name="fake",
            model_name="fixture-v1",
            attempt_count=1,
            raw_response=f"{RAW_RESPONSE_SENTINEL} api_key={PROVIDER_SECRET}",
        )


def _fake_settings() -> Settings:
    return Settings(
        ai_provider="fake",
        gemini_api_key=None,
        gemini_model=None,
        _env_file=None,
    )


def test_default_fake_comparison_returns_three_typed_sanitized_variants(
    database_session_factory: sessionmaker[Session],
) -> None:
    settings = _fake_settings()
    with database_session_factory() as session:
        seed_demo(session, DATASET_DIRECTORY)
        comparison = compare_prompt_versions(
            session,
            settings,
            dataset_directory=DATASET_DIRECTORY,
        )

    assert comparison.provider_name == "fake"
    assert comparison.model_name == "fixture-v1"
    assert comparison.evidence_codes == ("E-001", "E-002", "E-003", "E-004")
    assert (
        comparison.neutral.variant is PromptComparisonVariantName.NEUTRAL_EVIDENCE_FIRST
    )
    assert comparison.neutral.task_prompt.version is PromptVersion.V1
    assert len(comparison.neutral.hypotheses.hypotheses) == 3
    assert len(comparison.neutral.validated_hypotheses) == 3
    assert (
        comparison.leading.variant
        is PromptComparisonVariantName.LEADING_DEPLOYMENT_V2_4_1
    )
    assert comparison.leading.task_prompt.version is PromptVersion.V2
    assert (
        comparison.adversarial.variant
        is PromptComparisonVariantName.ADVERSARIAL_TOP_HYPOTHESIS
    )
    assert comparison.adversarial.task_prompt.version is PromptVersion.V2
    assert comparison.adversarial.challenged_hypothesis_id == "H-001"

    serialized = comparison.model_dump_json()
    assert "raw_response" not in serialized
    assert RAW_RESPONSE_SENTINEL not in serialized
    assert "Synthetic deployment record" not in serialized


def test_comparison_uses_registered_versions_redaction_and_validation_boundaries(
    database_session_factory: sessionmaker[Session],
) -> None:
    settings = _fake_settings()
    provider = _PromptAwareFakeProvider(build_configured_ai_provider(settings))

    with database_session_factory() as session:
        seed_demo(session, DATASET_DIRECTORY)
        evidence = session.scalar(
            select(EvidenceItem).where(EvidenceItem.evidence_code == "E-004")
        )
        assert evidence is not None
        evidence.original_text += f"\napi_key={EVIDENCE_SECRET}\n"
        session.flush()

        comparison = compare_prompt_versions(
            session,
            settings,
            dataset_directory=DATASET_DIRECTORY,
            ai_provider=provider,
        )

    stage_and_version = [
        (request.metadata.analysis_stage, request.prompts.task.version)
        for request in provider.requests
    ]
    assert stage_and_version == [
        (AnalysisStage.SUMMARY, PromptVersion.V1),
        (AnalysisStage.TIMELINE, PromptVersion.V1),
        (AnalysisStage.HYPOTHESES, PromptVersion.V1),
        (AnalysisStage.HYPOTHESES, PromptVersion.V2),
        (AnalysisStage.CRITIC, PromptVersion.V2),
    ]
    manifest_json = provider.requests[0].evidence_manifest.model_dump_json()
    assert EVIDENCE_SECRET not in manifest_json
    assert "[REDACTED_API_KEY]" in manifest_json

    neutral_top = min(
        comparison.neutral.hypotheses.hypotheses,
        key=lambda hypothesis: hypothesis.rank,
    )
    leading_top = min(
        comparison.leading.hypotheses.hypotheses,
        key=lambda hypothesis: hypothesis.rank,
    )
    serialized = comparison.model_dump_json()
    assert neutral_top.hypothesis_id == "H-001"
    assert leading_top.hypothesis_id == "H-002"
    assert PROVIDER_SECRET not in serialized
    assert RAW_RESPONSE_SENTINEL not in serialized
    assert "[REDACTED_API_KEY]" in serialized
    assert all(
        evidence.reference.status is EvidenceReferenceValidationStatus.VALID
        for hypothesis in comparison.leading.validated_hypotheses
        for evidence in hypothesis.supporting_evidence
    )
