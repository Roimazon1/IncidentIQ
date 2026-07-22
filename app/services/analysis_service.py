"""Lifecycle and provider-neutral stages for auditable analysis runs."""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, joinedload, selectinload

from app.config import Settings
from app.models import (
    AnalysisRun,
    AnalysisRunStatus,
    ClaimSupportStatus,
    EvidenceItem,
    Fact,
    Hypothesis,
    Incident,
    IncidentStatus,
    TimelineEvent,
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
from app.services.ai_provider import (
    AIProvider,
    AIProviderExecutionError,
    AIProviderFactory,
)
from app.services.evidence_manifest_service import EvidenceManifestService
from app.services.incident_service import IncidentService
from app.services.prompt_registry import PromptRegistry
from app.services.providers.fake_provider import FakeAIProvider
from app.services.providers.gemini_provider import GeminiAIProvider


StageOutputT = TypeVar("StageOutputT", bound=BaseModel)
_FAKE_RESPONSE_FIXTURE_PATH = (
    Path(__file__).resolve().parents[1] / "resources" / "fake_ai_core_responses.json"
)
_CORE_FAKE_FIXTURES = (
    "valid_summary",
    "valid_timeline",
    "valid_hypotheses",
)


@dataclass(frozen=True, slots=True)
class AnalysisPageData:
    """Safe structured data needed to render one saved analysis run."""

    analysis_run: AnalysisRun
    summary_output: SummaryOutputV1 | None


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

    def __init__(
        self,
        explanation: str,
        *,
        raw_response: str | None = None,
    ) -> None:
        self._raw_response = raw_response
        super().__init__(explanation)


class AnalysisService:
    """Create analysis runs and persist their legal lifecycle transitions."""

    def __init__(
        self,
        session: Session,
        *,
        ai_provider: AIProvider | None = None,
        configured_provider_name: str | None = None,
        configured_model_name: str | None = None,
    ) -> None:
        self.session = session
        self._incident_service = IncidentService(session)
        self._ai_provider = ai_provider
        self._configured_provider_name = configured_provider_name
        self._configured_model_name = configured_model_name

    def start_configured_analysis_run(
        self,
        incident_public_id: str,
    ) -> AnalysisRun:
        """Start a run using the provider identity validated at service creation."""
        if (
            self._configured_provider_name is None
            or self._configured_model_name is None
        ):
            raise AnalysisProviderRequiredError(
                "A configured AI provider is required to start analysis."
            )
        return self.start_analysis_run(
            incident_public_id,
            provider_name=self._configured_provider_name,
            model_name=self._configured_model_name,
        )

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
        self._require_complete_core_results(analysis_run)

        self._apply_completed_state(analysis_run)
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

        self._apply_failed_state(analysis_run, error_message=safe_error_message)
        self._commit(
            analysis_run,
            failure_message="The failed analysis run could not be saved.",
        )
        return analysis_run

    def run_core_analysis(self, run_id: int) -> AnalysisRun:
        """Run and atomically persist all required core analysis stages."""
        analysis_run = self._get_analysis_run_or_raise(run_id)
        self._require_running(analysis_run, operation="run core analysis")
        evidence_manifest = self._build_redacted_evidence_manifest(analysis_run)
        prompt_versions = {PromptName.SYSTEM.value: PromptVersion.V1.value}
        input_evidence_codes = [item.id for item in evidence_manifest.evidence]
        stage_records: dict[str, dict[str, object]] = {}
        current_stage = AnalysisStage.SUMMARY

        try:
            prompt_versions[PromptName.SUMMARY.value] = PromptVersion.V1.value
            summary_result = self._execute_stage(
                analysis_run,
                evidence_manifest,
                task_prompt=PromptName.SUMMARY,
                analysis_stage=AnalysisStage.SUMMARY,
                output_schema=OutputSchemaIdentifier.SUMMARY_V1,
                output_type=SummaryOutputV1,
            )
            stage_records[AnalysisStage.SUMMARY.value] = (
                self._build_success_stage_record(summary_result)
            )

            current_stage = AnalysisStage.TIMELINE
            prompt_versions[PromptName.TIMELINE.value] = PromptVersion.V1.value
            timeline_result = self._execute_stage(
                analysis_run,
                evidence_manifest,
                task_prompt=PromptName.TIMELINE,
                analysis_stage=AnalysisStage.TIMELINE,
                output_schema=OutputSchemaIdentifier.TIMELINE_V1,
                output_type=TimelineOutputV1,
            )
            stage_records[AnalysisStage.TIMELINE.value] = (
                self._build_success_stage_record(timeline_result)
            )

            current_stage = AnalysisStage.HYPOTHESES
            prompt_versions[PromptName.HYPOTHESES.value] = PromptVersion.V1.value
            hypotheses_result = self._execute_stage(
                analysis_run,
                evidence_manifest,
                task_prompt=PromptName.HYPOTHESES,
                analysis_stage=AnalysisStage.HYPOTHESES,
                output_schema=OutputSchemaIdentifier.HYPOTHESES_V1,
                output_type=HypothesesOutputV1,
            )
            self._require_materially_distinct_hypotheses(
                hypotheses_result.output,
                raw_response=hypotheses_result.audit.raw_response,
            )
            stage_records[AnalysisStage.HYPOTHESES.value] = (
                self._build_success_stage_record(hypotheses_result)
            )
        except AIProviderExecutionError as exc:
            audit = exc.details.audit
            stage_records[current_stage.value] = {
                "failure_category": exc.details.category.value,
                "raw_response": None if audit is None else audit.raw_response,
            }
            self._persist_failed_analysis(
                analysis_run,
                error_message=exc.details.explanation,
                prompt_versions=prompt_versions,
                input_evidence_codes=input_evidence_codes,
                stage_records=stage_records,
            )
            raise
        except AnalysisStageOutputError as exc:
            stage_records[current_stage.value] = {
                "failure_category": "stage_output_validation",
                "raw_response": exc._raw_response,
            }
            self._persist_failed_analysis(
                analysis_run,
                error_message=str(exc),
                prompt_versions=prompt_versions,
                input_evidence_codes=input_evidence_codes,
                stage_records=stage_records,
            )
            raise

        try:
            self._persist_completed_analysis(
                analysis_run,
                summary_result=summary_result,
                timeline_result=timeline_result,
                hypotheses_result=hypotheses_result,
                prompt_versions=prompt_versions,
                input_evidence_codes=input_evidence_codes,
                stage_records=stage_records,
            )
        except AnalysisPersistenceError as exc:
            analysis_run = self._get_analysis_run_or_raise(run_id)
            self._persist_failed_analysis(
                analysis_run,
                error_message=str(exc),
                prompt_versions=prompt_versions,
                input_evidence_codes=input_evidence_codes,
                stage_records=stage_records,
            )
            raise
        return analysis_run

    def run_core_analysis_to_terminal(self, run_id: int) -> AnalysisRun:
        """Return a completed or safely retained failed run for an application flow."""
        try:
            return self.run_core_analysis(run_id)
        except (
            AIProviderExecutionError,
            AnalysisStageOutputError,
            AnalysisPersistenceError,
        ):
            analysis_run = self._get_analysis_run_or_raise(run_id)
            if analysis_run.status is AnalysisRunStatus.FAILED:
                return analysis_run
            raise

    def get_analysis_page_data(
        self,
        incident_public_id: str,
        run_id: int,
    ) -> AnalysisPageData:
        """Load one incident-scoped run and its basic display relationships."""
        analysis_run = self.session.scalar(
            select(AnalysisRun)
            .join(AnalysisRun.incident)
            .options(
                joinedload(AnalysisRun.incident),
                selectinload(AnalysisRun.facts),
                selectinload(AnalysisRun.timeline_events),
                selectinload(AnalysisRun.hypotheses),
            )
            .where(
                AnalysisRun.id == run_id,
                Incident.public_id == incident_public_id,
            )
        )
        if analysis_run is None:
            raise AnalysisRunNotFoundError(
                f"Analysis run {run_id} was not found for incident "
                f"{incident_public_id}."
            )
        return AnalysisPageData(
            analysis_run=analysis_run,
            summary_output=self._extract_summary_output(analysis_run.raw_response),
        )

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
        evidence_manifest = self._build_redacted_evidence_manifest(analysis_run)
        return self._execute_stage(
            analysis_run,
            evidence_manifest,
            task_prompt=task_prompt,
            analysis_stage=analysis_stage,
            output_schema=output_schema,
            output_type=output_type,
        )

    def _execute_stage(
        self,
        analysis_run: AnalysisRun,
        evidence_manifest: EvidenceManifest,
        *,
        task_prompt: PromptName,
        analysis_stage: AnalysisStage,
        output_schema: OutputSchemaIdentifier,
        output_type: type[StageOutputT],
    ) -> AIResult[StageOutputT]:
        provider = self._require_ai_provider()
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
                "The AI provider returned an invalid analysis-stage output.",
                raw_response=result.audit.raw_response,
            )
        return AIResult[StageOutputT](
            output=output,
            metadata=result.metadata,
            audit=result.audit,
        )

    @staticmethod
    def _require_materially_distinct_hypotheses(
        output: HypothesesOutputV1,
        *,
        raw_response: str | None = None,
    ) -> None:
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
    def _build_success_stage_record(
        result: AIResult[StageOutputT],
    ) -> dict[str, object]:
        return {
            "metadata": result.metadata.model_dump(mode="json"),
            "parsed_output": result.output.model_dump(mode="json"),
            "raw_response": result.audit.raw_response,
        }

    def _persist_completed_analysis(
        self,
        analysis_run: AnalysisRun,
        *,
        summary_result: AIResult[SummaryOutputV1],
        timeline_result: AIResult[TimelineOutputV1],
        hypotheses_result: AIResult[HypothesesOutputV1],
        prompt_versions: dict[str, str],
        input_evidence_codes: list[str],
        stage_records: dict[str, dict[str, object]],
    ) -> None:
        analysis_run.prompt_versions = dict(prompt_versions)
        analysis_run.input_evidence_codes = list(input_evidence_codes)
        analysis_run.raw_response = self._serialize_stage_records(stage_records)
        analysis_run.facts = [
            Fact(
                claim=fact.claim,
                support_status=ClaimSupportStatus.UNSUPPORTED,
                confidence=fact.confidence,
                evidence_codes=list(
                    dict.fromkeys(reference.evidence_id for reference in fact.evidence)
                ),
                supporting_excerpt=next(
                    (
                        reference.excerpt
                        for reference in fact.evidence
                        if reference.excerpt is not None
                    ),
                    None,
                ),
            )
            for fact in summary_result.output.facts
        ]
        analysis_run.timeline_events = [
            TimelineEvent(
                event_time=self._parse_timeline_instant(event.timestamp),
                description=event.description,
                evidence_codes=list(
                    dict.fromkeys(reference.evidence_id for reference in event.evidence)
                ),
                is_inferred=event.is_inferred,
                confidence=event.confidence,
            )
            for event in timeline_result.output.events
        ]
        analysis_run.hypotheses = [
            Hypothesis(
                rank=hypothesis.rank,
                title=hypothesis.title,
                explanation=hypothesis.explanation,
                confidence=hypothesis.confidence,
                supporting_evidence_codes=list(
                    dict.fromkeys(
                        evidence.reference.evidence_id
                        for evidence in hypothesis.supporting_evidence
                    )
                ),
                contradicting_evidence_codes=list(
                    dict.fromkeys(
                        evidence.reference.evidence_id
                        for evidence in hypothesis.contradicting_evidence
                    )
                ),
                missing_evidence=list(hypothesis.missing_evidence),
                recommended_test=hypothesis.validation_test.description,
                expected_true_result=hypothesis.validation_test.expected_if_true,
                expected_false_result=hypothesis.validation_test.expected_if_false,
            )
            for hypothesis in hypotheses_result.output.hypotheses
        ]
        self._require_complete_core_results(analysis_run)
        self._apply_completed_state(analysis_run)
        self._commit(
            analysis_run,
            failure_message="The completed analysis results could not be saved.",
        )

    def _persist_failed_analysis(
        self,
        analysis_run: AnalysisRun,
        *,
        error_message: str,
        prompt_versions: dict[str, str],
        input_evidence_codes: list[str],
        stage_records: dict[str, dict[str, object]],
    ) -> None:
        self._require_running(analysis_run, target_status=AnalysisRunStatus.FAILED)
        analysis_run.prompt_versions = dict(prompt_versions)
        analysis_run.input_evidence_codes = list(input_evidence_codes)
        analysis_run.raw_response = self._serialize_stage_records(stage_records)
        self._apply_failed_state(analysis_run, error_message=error_message)
        self._commit(
            analysis_run,
            failure_message="The failed analysis run could not be saved.",
        )

    @staticmethod
    def _serialize_stage_records(
        stage_records: dict[str, dict[str, object]],
    ) -> str:
        return json.dumps(
            {"stages": stage_records},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    @staticmethod
    def _extract_summary_output(raw_response: str | None) -> SummaryOutputV1 | None:
        if raw_response is None:
            return None
        try:
            audit_envelope = json.loads(raw_response)
            parsed_output = audit_envelope["stages"][AnalysisStage.SUMMARY.value][
                "parsed_output"
            ]
            return SummaryOutputV1.model_validate(parsed_output)
        except (json.JSONDecodeError, KeyError, TypeError, ValidationError):
            return None

    @staticmethod
    def _parse_timeline_instant(value: str) -> datetime | None:
        candidate = f"{value[:-1]}+00:00" if value.endswith("Z") else value
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError:
            return None
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            return None
        return parsed.astimezone(UTC)

    @staticmethod
    def _require_complete_core_results(analysis_run: AnalysisRun) -> None:
        required_stages = {
            AnalysisStage.SUMMARY.value,
            AnalysisStage.TIMELINE.value,
            AnalysisStage.HYPOTHESES.value,
        }
        required_prompts = {
            PromptName.SYSTEM.value,
            PromptName.SUMMARY.value,
            PromptName.TIMELINE.value,
            PromptName.HYPOTHESES.value,
        }
        try:
            audit_envelope = json.loads(analysis_run.raw_response or "")
            stages = audit_envelope["stages"]
            stage_records_are_complete = all(
                isinstance(stages[stage], dict)
                and "metadata" in stages[stage]
                and "parsed_output" in stages[stage]
                and "raw_response" in stages[stage]
                for stage in required_stages
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            stage_records_are_complete = False
        if (
            set(analysis_run.prompt_versions) != required_prompts
            or not analysis_run.input_evidence_codes
            or not stage_records_are_complete
            or len(analysis_run.hypotheses) < 3
        ):
            raise AnalysisRunTransitionError(
                f"Analysis run {analysis_run.id} cannot transition to COMPLETED "
                "before all required stage results are available."
            )

    @staticmethod
    def _apply_completed_state(analysis_run: AnalysisRun) -> None:
        analysis_run.status = AnalysisRunStatus.COMPLETED
        analysis_run.completed_at = utc_now()
        analysis_run.error_message = None
        analysis_run.incident.status = IncidentStatus.COMPLETED

    @staticmethod
    def _apply_failed_state(
        analysis_run: AnalysisRun,
        *,
        error_message: str,
    ) -> None:
        analysis_run.status = AnalysisRunStatus.FAILED
        analysis_run.completed_at = utc_now()
        analysis_run.error_message = error_message
        analysis_run.incident.status = IncidentStatus.FAILED

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


def build_configured_analysis_service(
    session: Session,
    settings: Settings,
) -> AnalysisService:
    """Build an analysis service with the settings-selected concrete provider."""
    prompt_registry = PromptRegistry()

    def build_fake_provider(configured_settings: Settings) -> AIProvider:
        del configured_settings
        return FakeAIProvider.from_file_set(
            _FAKE_RESPONSE_FIXTURE_PATH,
            _CORE_FAKE_FIXTURES,
            prompt_resolver=prompt_registry.resolve_content,
            prompt_bundle_validator=prompt_registry.validate_bundle,
        )

    def build_gemini_provider(configured_settings: Settings) -> AIProvider:
        return GeminiAIProvider.from_settings(
            configured_settings,
            prompt_resolver=prompt_registry.resolve_content,
            prompt_bundle_validator=prompt_registry.validate_bundle,
        )

    provider = AIProviderFactory(
        fake_builder=build_fake_provider,
        gemini_builder=build_gemini_provider,
    ).create(settings)
    model_name = (
        FakeAIProvider.model_name
        if settings.ai_provider == FakeAIProvider.provider_name
        else settings.gemini_model
    )
    if model_name is None:
        raise AnalysisProviderRequiredError(
            "A configured AI provider model is required to start analysis."
        )
    return AnalysisService(
        session,
        ai_provider=provider,
        configured_provider_name=settings.ai_provider,
        configured_model_name=model_name,
    )
