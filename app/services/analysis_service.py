"""Lifecycle and provider-neutral stages for auditable analysis runs."""

from hashlib import sha256
from typing import TypeVar

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.models import (
    AnalysisRun,
    AnalysisRunStatus,
    EvidenceItem,
    Incident,
    IncidentStatus,
    utc_now,
)
from app.schemas.ai_outputs import (
    AIOutput,
    HypothesesOutputV1,
    SummaryOutputV1,
    TimelineOutputV1,
)
from app.schemas.ai_provider import (
    AIRequest,
    AIResult,
    AnalysisStage,
    OutputSchemaIdentifier,
    PromptBundle,
    PromptName,
    PromptReference,
    PromptVersion,
    SafeAIMetadata,
)
from app.schemas.evidence import EvidenceManifest, EvidenceManifestSource
from app.services.ai_provider import AIProvider
from app.services.evidence_manifest_service import EvidenceManifestService
from app.services.incident_service import IncidentService


StageOutputT = TypeVar("StageOutputT", bound=BaseModel)


class AnalysisEvidenceRequiredError(ValueError):
    """Raised when an incident has no evidence to analyze."""


class AnalysisAlreadyRunningError(RuntimeError):
    """Raised when an incident already has an active analysis run."""


class AnalysisRunNotFoundError(LookupError):
    """Raised when an analysis run identifier does not exist."""


class AnalysisRunTransitionError(RuntimeError):
    """Raised when a terminal or otherwise invalid transition is requested."""


class AnalysisPersistenceError(RuntimeError):
    """Raised when an analysis lifecycle write cannot be completed safely."""


class AnalysisProviderRequiredError(RuntimeError):
    """Raised when an AI stage is requested without an injected provider."""


class AnalysisStageOutputError(RuntimeError):
    """Raised when a provider violates the requested stage output contract."""


class AnalysisService:
    """Create analysis runs and persist their legal lifecycle transitions."""

    def __init__(
        self,
        session: Session,
        *,
        ai_provider: AIProvider | None = None,
    ) -> None:
        self.session = session
        self._incident_service = IncidentService(session)
        self._ai_provider = ai_provider

    def start_analysis_run(
        self,
        incident_public_id: str,
        *,
        provider_name: str,
        model_name: str,
    ) -> AnalysisRun:
        """Create one running analysis for an incident that has evidence."""
        incident = self._incident_service.get_incident_or_raise(incident_public_id)
        self._require_evidence(incident)
        self._require_no_running_analysis(incident)

        analysis_run = AnalysisRun(
            incident=incident,
            provider_name=provider_name,
            model_name=model_name,
            status=AnalysisRunStatus.RUNNING,
        )
        incident.status = IncidentStatus.ANALYZING
        self.session.add(analysis_run)
        self._commit(
            analysis_run,
            failure_message="The analysis run could not be started.",
        )
        return analysis_run

    def mark_analysis_run_completed(self, run_id: int) -> AnalysisRun:
        """Move a running analysis to its successful terminal state."""
        analysis_run = self._get_analysis_run_or_raise(run_id)
        self._require_running(analysis_run, target_status=AnalysisRunStatus.COMPLETED)

        analysis_run.status = AnalysisRunStatus.COMPLETED
        analysis_run.completed_at = utc_now()
        analysis_run.error_message = None
        analysis_run.incident.status = IncidentStatus.COMPLETED
        self._commit(
            analysis_run,
            failure_message="The completed analysis run could not be saved.",
        )
        return analysis_run

    def mark_analysis_run_failed(
        self,
        run_id: int,
        *,
        error_message: str,
    ) -> AnalysisRun:
        """Retain a running analysis as failed with a safe explanation."""
        safe_error_message = error_message.strip()
        if not safe_error_message:
            raise ValueError("analysis failure explanation must not be empty")

        analysis_run = self._get_analysis_run_or_raise(run_id)
        self._require_running(analysis_run, target_status=AnalysisRunStatus.FAILED)

        analysis_run.status = AnalysisRunStatus.FAILED
        analysis_run.completed_at = utc_now()
        analysis_run.error_message = safe_error_message
        analysis_run.incident.status = IncidentStatus.FAILED
        self._commit(
            analysis_run,
            failure_message="The failed analysis run could not be saved.",
        )
        return analysis_run

    def run_summary_stage(self, run_id: int) -> AIResult[SummaryOutputV1]:
        """Run typed summary, fact, and assumption extraction on redacted evidence."""
        return self._run_stage(
            run_id,
            operation="run summary extraction",
            task_prompt=PromptName.SUMMARY,
            analysis_stage=AnalysisStage.SUMMARY,
            output_schema=OutputSchemaIdentifier.SUMMARY_V1,
            output_type=SummaryOutputV1,
        )

    def run_timeline_stage(self, run_id: int) -> AIResult[TimelineOutputV1]:
        """Reconstruct a typed timeline with direct and inferred event labels."""
        return self._run_stage(
            run_id,
            operation="run timeline reconstruction",
            task_prompt=PromptName.TIMELINE,
            analysis_stage=AnalysisStage.TIMELINE,
            output_schema=OutputSchemaIdentifier.TIMELINE_V1,
            output_type=TimelineOutputV1,
        )

    def run_hypotheses_stage(self, run_id: int) -> AIResult[HypothesesOutputV1]:
        """Generate at least three ranked and materially distinct hypotheses."""
        result = self._run_stage(
            run_id,
            operation="run hypothesis generation",
            task_prompt=PromptName.HYPOTHESES,
            analysis_stage=AnalysisStage.HYPOTHESES,
            output_schema=OutputSchemaIdentifier.HYPOTHESES_V1,
            output_type=HypothesesOutputV1,
        )
        self._require_materially_distinct_hypotheses(result.output)
        return result

    def _get_analysis_run_or_raise(self, run_id: int) -> AnalysisRun:
        analysis_run = self.session.scalar(
            select(AnalysisRun).where(AnalysisRun.id == run_id)
        )
        if analysis_run is None:
            raise AnalysisRunNotFoundError(f"Analysis run {run_id} was not found.")
        return analysis_run

    def _require_evidence(self, incident: Incident) -> None:
        evidence_id = self.session.scalar(
            select(EvidenceItem.id)
            .where(EvidenceItem.incident_id == incident.id)
            .limit(1)
        )
        if evidence_id is None:
            raise AnalysisEvidenceRequiredError(
                f"Incident {incident.public_id} requires evidence before analysis."
            )

    def _require_no_running_analysis(self, incident: Incident) -> None:
        running_run_id = self.session.scalar(
            select(AnalysisRun.id)
            .where(
                AnalysisRun.incident_id == incident.id,
                AnalysisRun.status == AnalysisRunStatus.RUNNING,
            )
            .limit(1)
        )
        if running_run_id is not None:
            raise AnalysisAlreadyRunningError(
                f"Incident {incident.public_id} already has a running analysis."
            )

    @staticmethod
    def _require_running(
        analysis_run: AnalysisRun,
        *,
        operation: str | None = None,
        target_status: AnalysisRunStatus | None = None,
    ) -> None:
        if analysis_run.status is not AnalysisRunStatus.RUNNING:
            requested_operation = operation
            if requested_operation is None:
                if target_status is None:
                    raise ValueError("a requested analysis operation is required")
                requested_operation = f"transition to {target_status.value}"
            raise AnalysisRunTransitionError(
                f"Analysis run {analysis_run.id} cannot {requested_operation} while "
                f"its status is {analysis_run.status.value}."
            )

    def _require_ai_provider(self) -> AIProvider:
        if self._ai_provider is None:
            raise AnalysisProviderRequiredError(
                "An AI provider is required to run analysis stages."
            )
        return self._ai_provider

    def _build_redacted_evidence_manifest(
        self,
        analysis_run: AnalysisRun,
    ) -> EvidenceManifest:
        evidence_items = tuple(
            self.session.scalars(
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

    def _run_stage(
        self,
        run_id: int,
        *,
        operation: str,
        task_prompt: PromptName,
        analysis_stage: AnalysisStage,
        output_schema: OutputSchemaIdentifier,
        output_type: type[StageOutputT],
    ) -> AIResult[StageOutputT]:
        analysis_run = self._get_analysis_run_or_raise(run_id)
        self._require_running(analysis_run, operation=operation)
        provider = self._require_ai_provider()
        evidence_manifest = self._build_redacted_evidence_manifest(analysis_run)
        request = self._build_stage_request(
            analysis_run,
            evidence_manifest,
            task_prompt=task_prompt,
            analysis_stage=analysis_stage,
            output_schema=output_schema,
        )
        result = provider.generate(request)
        return self._validate_stage_result(
            result,
            request=request,
            analysis_run=analysis_run,
            output_type=output_type,
        )

    @staticmethod
    def _validate_stage_result(
        result: AIResult[AIOutput],
        *,
        request: AIRequest,
        analysis_run: AnalysisRun,
        output_type: type[StageOutputT],
    ) -> AIResult[StageOutputT]:
        metadata = result.metadata
        output = result.output
        if (
            not isinstance(output, output_type)
            or metadata.analysis_stage is not request.metadata.analysis_stage
            or metadata.output_schema is not request.output_schema
            or metadata.system_prompt != request.prompts.system
            or metadata.task_prompt != request.prompts.task
            or metadata.request_identifier != request.metadata.request_identifier
            or metadata.provider_name != analysis_run.provider_name
            or metadata.model_name != analysis_run.model_name
        ):
            raise AnalysisStageOutputError(
                "The AI provider returned an invalid analysis-stage output."
            )
        return AIResult[StageOutputT](
            output=output,
            metadata=result.metadata,
            audit=result.audit,
        )

    @staticmethod
    def _require_materially_distinct_hypotheses(
        output: HypothesesOutputV1,
    ) -> None:
        normalized_titles = {
            " ".join(hypothesis.title.casefold().split())
            for hypothesis in output.hypotheses
        }
        if len(normalized_titles) < 3:
            raise AnalysisStageOutputError(
                "The AI provider returned fewer than three distinct hypotheses."
            )

    @staticmethod
    def _build_stage_request(
        analysis_run: AnalysisRun,
        evidence_manifest: EvidenceManifest,
        *,
        task_prompt: PromptName,
        analysis_stage: AnalysisStage,
        output_schema: OutputSchemaIdentifier,
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
                    version=PromptVersion.V1,
                ),
            ),
            output_schema=output_schema,
            metadata=SafeAIMetadata(
                request_identifier=(
                    f"analysis-run-{analysis_run.id}-{analysis_stage.value}"
                ),
                incident_public_identifier=analysis_run.incident.public_id,
                analysis_stage=analysis_stage,
                evidence_manifest_checksum=manifest_checksum,
            ),
        )

    def _commit(
        self,
        analysis_run: AnalysisRun,
        *,
        failure_message: str,
    ) -> None:
        try:
            self.session.flush()
            self.session.refresh(analysis_run)
            self.session.commit()
        except SQLAlchemyError as exc:
            self.session.rollback()
            raise AnalysisPersistenceError(failure_message) from exc
