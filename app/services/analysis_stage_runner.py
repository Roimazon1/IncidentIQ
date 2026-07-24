"""Provider-neutral execution boundary for Phase 6 analysis stages."""

from hashlib import sha256
from typing import Protocol, TypeVar

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import AnalysisRun, EvidenceItem, Incident
from app.schemas.ai_outputs import (
    AIOutput,
    HypothesesOutputV1,
    OpenQuestionsOutputV1,
    ReasoningRisksOutputV1,
)
from app.schemas.ai_provider import (
    AIRequest,
    AIResult,
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
from app.schemas.evidence import EvidenceManifest, EvidenceManifestSource
from app.services.ai_provider import AIProvider, ai_result_matches_request
from app.services.bias_service import BiasAnalysisError, BiasService
from app.services.evidence_manifest_service import EvidenceManifestService
from app.services.open_question_source_service import OpenQuestionSourceService
from app.services.validation_service import ValidationService


StageOutputT = TypeVar("StageOutputT", bound=BaseModel)


class AnalysisStageContext(Protocol):
    """Run identity required by the provider-neutral stage boundary."""

    id: int
    incident_id: int
    incident: Incident


class AnalysisEvidenceRequiredError(ValueError):
    """Raised when an incident has no evidence to analyze."""


class AnalysisProviderRequiredError(RuntimeError):
    """Raised when an AI stage is requested without an injected provider."""


class AnalysisStageOutputError(RuntimeError):
    """Raised when a provider violates the requested stage output contract."""

    def __init__(
        self,
        explanation: str,
        *,
        raw_response: str | None = None,
    ) -> None:
        self._raw_response = raw_response
        super().__init__(explanation)

    @property
    def audit_raw_response(self) -> str | None:
        """Return internal raw-response audit data without exposing it in errors."""
        return self._raw_response


class AnalysisStageRunner:
    """Build redacted requests and validate provider-neutral stage results."""

    def __init__(
        self,
        session: Session,
        ai_provider: AIProvider | None,
    ) -> None:
        self._session = session
        self._ai_provider = ai_provider
        self._evaluation_provider_identity: tuple[str, str] | None = None

    def build_evidence_manifest(
        self,
        analysis_run: AnalysisStageContext,
    ) -> EvidenceManifest:
        """Build one provider-safe manifest from locally stored evidence."""
        evidence_items = tuple(
            self._session.scalars(
                select(EvidenceItem)
                .where(EvidenceItem.incident_id == analysis_run.incident_id)
                .order_by(EvidenceItem.evidence_code)
            )
        )
        if not evidence_items:
            raise AnalysisEvidenceRequiredError(
                f"Incident {analysis_run.incident.public_id} requires evidence "
                "before analysis."
            )
        sources = (
            EvidenceManifestSource(
                evidence_code=item.evidence_code,
                source_name=item.source_name,
                evidence_type=item.evidence_type,
                original_text=item.original_text,
            )
            for item in evidence_items
        )
        return EvidenceManifestService.build_evidence_manifest(
            analysis_run.incident.public_id,
            sources,
        )

    def execute_stage(
        self,
        analysis_run: AnalysisStageContext,
        evidence_manifest: EvidenceManifest,
        *,
        task_prompt: PromptName,
        analysis_stage: AnalysisStage,
        output_schema: OutputSchemaIdentifier,
        output_type: type[StageOutputT],
        task_prompt_version: PromptVersion = PromptVersion.V1,
        critic_context: CriticContextV1 | None = None,
        bias_context: BiasContextV1 | None = None,
        open_questions_context: OpenQuestionsContextV1 | None = None,
    ) -> AIResult[StageOutputT]:
        """Execute one typed stage and enforce exact request/result traceability."""
        if critic_context is not None:
            self._validate_critic_context(critic_context, evidence_manifest)
        if bias_context is not None:
            self._validate_bias_context(bias_context, evidence_manifest)
        if open_questions_context is not None:
            self._validate_open_questions_context(
                open_questions_context,
                evidence_manifest,
            )
        request = self._build_stage_request(
            analysis_run,
            evidence_manifest,
            task_prompt=task_prompt,
            task_prompt_version=task_prompt_version,
            analysis_stage=analysis_stage,
            output_schema=output_schema,
            critic_context=critic_context,
            bias_context=bias_context,
            open_questions_context=open_questions_context,
        )
        result = self._require_ai_provider().generate(request)
        typed_result = self._validate_stage_result(
            result,
            request=request,
            analysis_run=analysis_run,
            output_type=output_type,
        )
        ValidationService.validate_output_references(
            typed_result.output,
            evidence_manifest,
        )
        return typed_result

    @staticmethod
    def require_materially_distinct_hypotheses(
        output: HypothesesOutputV1,
        *,
        raw_response: str | None = None,
    ) -> None:
        """Require three hypotheses with materially different normalized titles."""
        normalized_titles = {
            " ".join(hypothesis.title.casefold().split())
            for hypothesis in output.hypotheses
        }
        if len(normalized_titles) < 3:
            raise AnalysisStageOutputError(
                "The AI provider returned fewer than three distinct hypotheses.",
                raw_response=raw_response,
            )

    @staticmethod
    def require_required_reasoning_risks(
        output: ReasoningRisksOutputV1,
        *,
        raw_response: str | None = None,
    ) -> None:
        """Require every locked core reasoning-risk warning category."""
        try:
            BiasService.identify_risks(output)
        except BiasAnalysisError as exc:
            raise AnalysisStageOutputError(
                str(exc),
                raw_response=raw_response,
            ) from exc

    @staticmethod
    def require_traceable_open_questions(
        output: OpenQuestionsOutputV1,
        context: OpenQuestionsContextV1,
        *,
        raw_response: str | None = None,
    ) -> None:
        """Require every question to reference an unresolved typed analysis item."""
        allowed_sources = {
            (source.source_kind, source.source_reference)
            for source in OpenQuestionSourceService.build_source_options(context)
        }
        if any(
            (question.source_kind, question.source_reference) not in allowed_sources
            for question in output.questions
        ):
            raise AnalysisStageOutputError(
                "The open-question output contains an untraceable analysis source.",
                raw_response=raw_response,
            )

    @staticmethod
    def build_contradiction_source_reference(
        hypothesis_id: str,
        evidence_id: str,
        line_range: str,
    ) -> str:
        """Return the stable reference for one typed contradicting evidence item."""
        return OpenQuestionSourceService.build_contradiction_source_reference(
            hypothesis_id,
            evidence_id,
            line_range,
        )

    def _require_ai_provider(self) -> AIProvider:
        if self._ai_provider is None:
            raise AnalysisProviderRequiredError(
                "An AI provider is required to run analysis stages."
            )
        return self._ai_provider

    @staticmethod
    def _validate_critic_context(
        critic_context: CriticContextV1,
        evidence_manifest: EvidenceManifest,
    ) -> None:
        for initial_output in (
            critic_context.summary,
            critic_context.timeline,
            critic_context.hypotheses,
        ):
            ValidationService.validate_output_references(
                initial_output,
                evidence_manifest,
            )

    @classmethod
    def _validate_bias_context(
        cls,
        bias_context: BiasContextV1,
        evidence_manifest: EvidenceManifest,
    ) -> None:
        cls._validate_critic_context(
            bias_context.original_analysis,
            evidence_manifest,
        )
        expected_validated_analysis = ValidationService.build_validated_analysis_view(
            bias_context.original_analysis.summary,
            bias_context.original_analysis.timeline,
            bias_context.original_analysis.hypotheses,
            evidence_manifest,
        )
        if bias_context.validated_analysis != expected_validated_analysis:
            raise AnalysisStageOutputError(
                "The bias request contains inconsistent deterministic validation data."
            )
        ValidationService.validate_output_references(
            bias_context.critic,
            evidence_manifest,
        )

    @classmethod
    def _validate_open_questions_context(
        cls,
        context: OpenQuestionsContextV1,
        evidence_manifest: EvidenceManifest,
    ) -> None:
        cls._validate_bias_context(
            context.analysis_context,
            evidence_manifest,
        )
        try:
            BiasService.identify_risks(context.reasoning_risks)
        except BiasAnalysisError as exc:
            raise AnalysisStageOutputError(str(exc)) from exc

    def _validate_stage_result(
        self,
        result: AIResult[AIOutput],
        *,
        request: AIRequest,
        analysis_run: AnalysisStageContext,
        output_type: type[StageOutputT],
    ) -> AIResult[StageOutputT]:
        if isinstance(analysis_run, AnalysisRun):
            provider_name = analysis_run.provider_name
            model_name = analysis_run.model_name
        elif self._evaluation_provider_identity is None:
            provider_name = result.metadata.provider_name
            model_name = result.metadata.model_name
        else:
            provider_name, model_name = self._evaluation_provider_identity

        if not ai_result_matches_request(
            result,
            request=request,
            output_type=output_type,
            provider_name=provider_name,
            model_name=model_name,
        ):
            raise AnalysisStageOutputError(
                "The AI provider returned an invalid analysis-stage output.",
                raw_response=result.audit.raw_response,
            )
        if (
            not isinstance(analysis_run, AnalysisRun)
            and self._evaluation_provider_identity is None
        ):
            self._evaluation_provider_identity = (
                result.metadata.provider_name,
                result.metadata.model_name,
            )
        return AIResult[StageOutputT](
            output=result.output,
            metadata=result.metadata,
            audit=result.audit,
        )

    @staticmethod
    def _build_stage_request(
        analysis_run: AnalysisStageContext,
        evidence_manifest: EvidenceManifest,
        *,
        task_prompt: PromptName,
        task_prompt_version: PromptVersion = PromptVersion.V1,
        analysis_stage: AnalysisStage,
        output_schema: OutputSchemaIdentifier,
        critic_context: CriticContextV1 | None = None,
        bias_context: BiasContextV1 | None = None,
        open_questions_context: OpenQuestionsContextV1 | None = None,
    ) -> AIRequest:
        manifest_checksum = sha256(
            evidence_manifest.model_dump_json().encode("utf-8")
        ).hexdigest()
        return AIRequest(
            evidence_manifest=evidence_manifest,
            prompts=PromptBundle(
                system=PromptReference(
                    name=PromptName.SYSTEM,
                    version=PromptVersion.V1,
                ),
                task=PromptReference(
                    name=task_prompt,
                    version=task_prompt_version,
                ),
            ),
            output_schema=output_schema,
            critic_context=critic_context,
            bias_context=bias_context,
            open_questions_context=open_questions_context,
            metadata=SafeAIMetadata(
                request_identifier=(
                    f"analysis-run-{analysis_run.id}-{analysis_stage.value}"
                ),
                incident_public_identifier=analysis_run.incident.public_id,
                analysis_stage=analysis_stage,
                evidence_manifest_checksum=manifest_checksum,
            ),
        )
