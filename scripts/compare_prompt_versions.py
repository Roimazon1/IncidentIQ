"""Compare three prompt conditions against the synthetic checkout incident."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from app.config import Settings, get_settings  # noqa: E402
from app.models import Incident  # noqa: E402
from app.schemas.ai_outputs import (  # noqa: E402
    CriticOutputV1,
    HypothesesOutputV1,
    SummaryOutputV1,
    TimelineOutputV1,
)
from app.schemas.ai_provider import (  # noqa: E402
    AIResult,
    AnalysisStage,
    CriticContextV1,
    OutputSchemaIdentifier,
    PromptName,
    PromptVersion,
)
from app.schemas.evidence import EvidenceManifest  # noqa: E402
from app.schemas.prompt_comparison import (  # noqa: E402
    AdversarialComparisonVariant,
    ComparisonEvidenceValidation,
    HypothesisComparisonVariant,
    PromptComparisonResult,
    PromptComparisonVariantName,
)
from app.services.ai_provider import AIProvider  # noqa: E402
from app.services.analysis_service_factory import (  # noqa: E402
    build_configured_ai_provider,
)
from app.services.analysis_stage_runner import AnalysisStageRunner  # noqa: E402
from app.services.redaction_service import RedactionService  # noqa: E402
from app.services.validation_service import ValidationService  # noqa: E402
from scripts.seed_demo import (  # noqa: E402
    DEFAULT_DATASET_DIRECTORY,
    load_demo_definition,
)


LOGGER = logging.getLogger(__name__)
StructuredOutputT = TypeVar("StructuredOutputT", bound=BaseModel)


class PromptComparisonError(RuntimeError):
    """Raised when the configured demo comparison cannot be completed safely."""


@dataclass(frozen=True, slots=True)
class _EvaluationRunContext:
    """Non-persisted identity used by the shared analysis-stage boundary."""

    id: int
    incident_id: int
    incident: Incident


def compare_prompt_versions(
    session: Session,
    settings: Settings,
    *,
    dataset_directory: Path = DEFAULT_DATASET_DIRECTORY,
    ai_provider: AIProvider | None = None,
) -> PromptComparisonResult:
    """Run and return the three sanitized P10-03 comparison variants."""
    incident = _load_demo_incident(session, dataset_directory)
    provider = ai_provider or build_configured_ai_provider(settings)
    run_context = _EvaluationRunContext(
        id=incident.id,
        incident_id=incident.id,
        incident=incident,
    )
    stage_runner = AnalysisStageRunner(session, provider)
    evidence_manifest = stage_runner.build_evidence_manifest(run_context)

    summary_result = stage_runner.execute_stage(
        run_context,
        evidence_manifest,
        task_prompt=PromptName.SUMMARY,
        analysis_stage=AnalysisStage.SUMMARY,
        output_schema=OutputSchemaIdentifier.SUMMARY_V1,
        output_type=SummaryOutputV1,
    )
    timeline_result = stage_runner.execute_stage(
        run_context,
        evidence_manifest,
        task_prompt=PromptName.TIMELINE,
        analysis_stage=AnalysisStage.TIMELINE,
        output_schema=OutputSchemaIdentifier.TIMELINE_V1,
        output_type=TimelineOutputV1,
    )
    summary = _sanitize_output(summary_result.output, SummaryOutputV1)
    timeline = _sanitize_output(timeline_result.output, TimelineOutputV1)

    neutral_result = _execute_hypothesis_variant(
        stage_runner,
        run_context,
        evidence_manifest,
        prompt_version=PromptVersion.V1,
    )
    leading_result = _execute_hypothesis_variant(
        stage_runner,
        run_context,
        evidence_manifest,
        prompt_version=PromptVersion.V2,
    )
    neutral_hypotheses = _sanitize_output(
        neutral_result.output,
        HypothesesOutputV1,
    )
    leading_hypotheses = _sanitize_output(
        leading_result.output,
        HypothesesOutputV1,
    )

    critic_result = stage_runner.execute_stage(
        run_context,
        evidence_manifest,
        task_prompt=PromptName.CRITIC,
        task_prompt_version=PromptVersion.V2,
        analysis_stage=AnalysisStage.CRITIC,
        output_schema=OutputSchemaIdentifier.CRITIC_V1,
        output_type=CriticOutputV1,
        critic_context=CriticContextV1(
            summary=summary,
            timeline=timeline,
            hypotheses=neutral_hypotheses,
        ),
    )
    critique = _sanitize_output(critic_result.output, CriticOutputV1)
    neutral_validation = ValidationService.build_validated_analysis_view(
        summary,
        timeline,
        neutral_hypotheses,
        evidence_manifest,
    )
    leading_validation = ValidationService.build_validated_analysis_view(
        summary,
        timeline,
        leading_hypotheses,
        evidence_manifest,
    )
    critic_reference_validation = tuple(
        ComparisonEvidenceValidation(
            evidence_id=outcome.evidence_id,
            line_range=outcome.line_range,
            status=outcome.status,
            message=outcome.message,
        )
        for outcome in ValidationService.validate_output_references(
            critique,
            evidence_manifest,
        )
    )
    top_hypothesis = min(
        neutral_hypotheses.hypotheses,
        key=lambda hypothesis: hypothesis.rank,
    )

    return PromptComparisonResult(
        incident_public_id=incident.public_id,
        provider_name=neutral_result.metadata.provider_name,
        model_name=neutral_result.metadata.model_name,
        evidence_codes=tuple(item.id for item in evidence_manifest.evidence),
        neutral=HypothesisComparisonVariant(
            variant=PromptComparisonVariantName.NEUTRAL_EVIDENCE_FIRST,
            task_prompt=neutral_result.metadata.task_prompt,
            output_schema=neutral_result.metadata.output_schema,
            hypotheses=neutral_hypotheses,
            validated_hypotheses=neutral_validation.hypotheses,
        ),
        leading=HypothesisComparisonVariant(
            variant=PromptComparisonVariantName.LEADING_DEPLOYMENT_V2_4_1,
            task_prompt=leading_result.metadata.task_prompt,
            output_schema=leading_result.metadata.output_schema,
            hypotheses=leading_hypotheses,
            validated_hypotheses=leading_validation.hypotheses,
        ),
        adversarial=AdversarialComparisonVariant(
            variant=PromptComparisonVariantName.ADVERSARIAL_TOP_HYPOTHESIS,
            task_prompt=critic_result.metadata.task_prompt,
            output_schema=critic_result.metadata.output_schema,
            challenged_hypothesis_id=top_hypothesis.hypothesis_id,
            critique=critique,
            evidence_validation=critic_reference_validation,
        ),
    )


def _load_demo_incident(
    session: Session,
    dataset_directory: Path,
) -> Incident:
    definition = load_demo_definition(dataset_directory)
    incidents = list(
        session.scalars(select(Incident).where(Incident.name == definition.name))
    )
    if not incidents:
        raise PromptComparisonError(
            "Seed the synthetic checkout incident before comparing prompts."
        )
    if len(incidents) > 1:
        raise PromptComparisonError(
            "Multiple incidents match the synthetic prompt-comparison dataset."
        )
    return incidents[0]


def _execute_hypothesis_variant(
    stage_runner: AnalysisStageRunner,
    run_context: _EvaluationRunContext,
    evidence_manifest: EvidenceManifest,
    *,
    prompt_version: PromptVersion,
) -> AIResult[HypothesesOutputV1]:
    result = stage_runner.execute_stage(
        run_context,
        evidence_manifest,
        task_prompt=PromptName.HYPOTHESES,
        task_prompt_version=prompt_version,
        analysis_stage=AnalysisStage.HYPOTHESES,
        output_schema=OutputSchemaIdentifier.HYPOTHESES_V1,
        output_type=HypothesesOutputV1,
    )
    stage_runner.require_materially_distinct_hypotheses(
        result.output,
        raw_response=result.audit.raw_response,
    )
    return result


def _sanitize_output(
    output: StructuredOutputT,
    output_type: type[StructuredOutputT],
) -> StructuredOutputT:
    sanitized_data = _redact_strings(output.model_dump(mode="json"))
    return output_type.model_validate(sanitized_data)


def _redact_strings(value: object) -> object:
    if isinstance(value, str):
        return RedactionService.redact_text(value).redacted_text
    if isinstance(value, list):
        return [_redact_strings(item) for item in value]
    if isinstance(value, dict):
        return {key: _redact_strings(item) for key, item in value.items()}
    return value


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare sanitized prompt results for the seeded demo incident."
    )
    parser.add_argument(
        "--use-configured-provider",
        action="store_true",
        help=(
            "Use the provider configured in the environment. "
            "The default always uses the offline fake provider."
        ),
    )
    return parser.parse_args()


def _default_evaluation_settings(
    configured_settings: Settings,
    *,
    use_configured_provider: bool,
) -> Settings:
    if use_configured_provider:
        return configured_settings
    return configured_settings.model_copy(
        update={
            "ai_provider": "fake",
            "gemini_api_key": None,
            "gemini_model": None,
        }
    )


def main() -> None:
    """Print only sanitized structured comparison data as JSON."""
    arguments = _parse_arguments()
    settings = _default_evaluation_settings(
        get_settings(),
        use_configured_provider=arguments.use_configured_provider,
    )

    from app.database import SessionLocal

    try:
        with SessionLocal() as session:
            comparison = compare_prompt_versions(session, settings)
    except Exception:
        LOGGER.error("Prompt comparison failed safely.")
        raise SystemExit(1) from None

    sys.stdout.write(comparison.model_dump_json(indent=2))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
