"""Public lifecycle and orchestration facade for analysis runs."""

from dataclasses import dataclass
from typing import TypeVar

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, joinedload, selectinload

from app.models import (
    AnalysisRun,
    AnalysisRunStatus,
    ClaimSupportStatus,
    EvidenceItem,
    Fact,
    Incident,
    IncidentStatus,
)
from app.schemas.ai_outputs import (
    CriticOutputV1,
    HypothesesOutputV1,
    ReasoningRisksOutputV1,
    SummaryOutputV1,
    TimelineOutputV1,
)
from app.schemas.ai_provider import (
    AIResult,
    AnalysisStage,
    BiasContextV1,
    CriticContextV1,
    OutputSchemaIdentifier,
    PromptName,
    PromptVersion,
)
from app.services.ai_provider import AIProvider, AIProviderExecutionError
from app.services.analysis_persistence import (
    AnalysisPersistenceError,
    AnalysisResultPersistence,
    AnalysisRunTransitionError,
)
from app.services.analysis_stage_runner import (
    AnalysisEvidenceRequiredError,
    AnalysisProviderRequiredError,
    AnalysisStageOutputError,
    AnalysisStageRunner,
)
from app.services.incident_service import IncidentService
from app.services.validation_service import ValidationService


StageOutputT = TypeVar("StageOutputT", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class AnalysisPageData:
    """Safe structured data needed to render one saved analysis run."""

    analysis_run: AnalysisRun
    summary_output: SummaryOutputV1 | None
    timeline_output: TimelineOutputV1 | None
    critic_output: CriticOutputV1 | None
    confirmed_facts: tuple[Fact, ...]
    unconfirmed_claims: tuple[Fact, ...]


class AnalysisAlreadyRunningError(RuntimeError):
    """Raised when an incident already has an active analysis run."""


class AnalysisRunNotFoundError(LookupError):
    """Raised when an analysis run identifier does not exist."""


class AnalysisService:
    """Create, orchestrate, persist, and reopen Phase 6 analysis runs."""

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
        self._stage_runner = AnalysisStageRunner(session, ai_provider)
        self._result_persistence = AnalysisResultPersistence(session)
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
        self._begin_analysis_start_transaction()
        incident = self._incident_service.get_incident_or_raise(incident_public_id)
        self._require_evidence(incident)
        try:
            self._require_no_running_analysis(incident)
        except AnalysisAlreadyRunningError:
            self.session.rollback()
            raise

        analysis_run = AnalysisRun(
            incident=incident,
            provider_name=provider_name,
            model_name=model_name,
            status=AnalysisRunStatus.RUNNING,
        )
        incident.status = IncidentStatus.ANALYZING
        self.session.add(analysis_run)
        try:
            self._result_persistence.commit(
                analysis_run,
                failure_message="The analysis run could not be started.",
            )
        except AnalysisPersistenceError as exc:
            if isinstance(exc.__cause__, IntegrityError) and self._has_running_analysis(
                incident.id
            ):
                raise AnalysisAlreadyRunningError(
                    f"Incident {incident.public_id} already has a running analysis."
                ) from exc
            raise
        return analysis_run

    def mark_analysis_run_completed(self, run_id: int) -> AnalysisRun:
        """Move a running analysis to its successful terminal state."""
        analysis_run = self._get_analysis_run_or_raise(run_id)
        self._require_running(analysis_run, target_status=AnalysisRunStatus.COMPLETED)
        self._result_persistence.require_complete_core_results(analysis_run)

        self._result_persistence.apply_completed_state(analysis_run)
        self._result_persistence.commit(
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

        self._result_persistence.apply_failed_state(
            analysis_run,
            error_message=safe_error_message,
        )
        self._result_persistence.commit(
            analysis_run,
            failure_message="The failed analysis run could not be saved.",
        )
        return analysis_run

    def run_core_analysis(self, run_id: int) -> AnalysisRun:
        """Run and atomically persist all required core analysis stages."""
        analysis_run = self._get_analysis_run_or_raise(run_id)
        self._require_running(analysis_run, operation="run core analysis")
        evidence_manifest = self._stage_runner.build_evidence_manifest(analysis_run)
        prompt_versions = {PromptName.SYSTEM.value: PromptVersion.V1.value}
        input_evidence_codes = [item.id for item in evidence_manifest.evidence]
        stage_records: dict[str, dict[str, object]] = {}
        current_stage = AnalysisStage.SUMMARY

        try:
            prompt_versions[PromptName.SUMMARY.value] = PromptVersion.V1.value
            summary_result = self._stage_runner.execute_stage(
                analysis_run,
                evidence_manifest,
                task_prompt=PromptName.SUMMARY,
                analysis_stage=AnalysisStage.SUMMARY,
                output_schema=OutputSchemaIdentifier.SUMMARY_V1,
                output_type=SummaryOutputV1,
            )
            stage_records[AnalysisStage.SUMMARY.value] = (
                self._result_persistence.build_success_stage_record(summary_result)
            )

            current_stage = AnalysisStage.TIMELINE
            prompt_versions[PromptName.TIMELINE.value] = PromptVersion.V1.value
            timeline_result = self._stage_runner.execute_stage(
                analysis_run,
                evidence_manifest,
                task_prompt=PromptName.TIMELINE,
                analysis_stage=AnalysisStage.TIMELINE,
                output_schema=OutputSchemaIdentifier.TIMELINE_V1,
                output_type=TimelineOutputV1,
            )
            stage_records[AnalysisStage.TIMELINE.value] = (
                self._result_persistence.build_success_stage_record(timeline_result)
            )

            current_stage = AnalysisStage.HYPOTHESES
            prompt_versions[PromptName.HYPOTHESES.value] = PromptVersion.V1.value
            hypotheses_result = self._stage_runner.execute_stage(
                analysis_run,
                evidence_manifest,
                task_prompt=PromptName.HYPOTHESES,
                analysis_stage=AnalysisStage.HYPOTHESES,
                output_schema=OutputSchemaIdentifier.HYPOTHESES_V1,
                output_type=HypothesesOutputV1,
            )
            self._stage_runner.require_materially_distinct_hypotheses(
                hypotheses_result.output,
                raw_response=hypotheses_result.audit.raw_response,
            )
            stage_records[AnalysisStage.HYPOTHESES.value] = (
                self._result_persistence.build_success_stage_record(hypotheses_result)
            )

            current_stage = AnalysisStage.CRITIC
            prompt_versions[PromptName.CRITIC.value] = PromptVersion.V1.value
            initial_analysis_context = CriticContextV1(
                summary=summary_result.output,
                timeline=timeline_result.output,
                hypotheses=hypotheses_result.output,
            )
            validated_analysis = ValidationService.build_validated_analysis_view(
                summary_result.output,
                timeline_result.output,
                hypotheses_result.output,
                evidence_manifest,
            )
            critic_result = self._stage_runner.execute_stage(
                analysis_run,
                evidence_manifest,
                task_prompt=PromptName.CRITIC,
                analysis_stage=AnalysisStage.CRITIC,
                output_schema=OutputSchemaIdentifier.CRITIC_V1,
                output_type=CriticOutputV1,
                critic_context=initial_analysis_context,
            )
            stage_records[AnalysisStage.CRITIC.value] = (
                self._result_persistence.build_success_stage_record(critic_result)
            )

            current_stage = AnalysisStage.BIAS
            prompt_versions[PromptName.BIAS.value] = PromptVersion.V1.value
            bias_result = self._stage_runner.execute_stage(
                analysis_run,
                evidence_manifest,
                task_prompt=PromptName.BIAS,
                analysis_stage=AnalysisStage.BIAS,
                output_schema=OutputSchemaIdentifier.REASONING_RISKS_V1,
                output_type=ReasoningRisksOutputV1,
                bias_context=BiasContextV1(
                    original_analysis=initial_analysis_context,
                    validated_analysis=validated_analysis,
                    critic=critic_result.output,
                ),
            )
            self._stage_runner.require_required_reasoning_risks(
                bias_result.output,
                raw_response=bias_result.audit.raw_response,
            )
            stage_records[AnalysisStage.BIAS.value] = (
                self._result_persistence.build_success_stage_record(bias_result)
            )
        except AIProviderExecutionError as exc:
            stage_records[current_stage.value] = (
                self._result_persistence.build_provider_failure_stage_record(exc)
            )
            self._persist_failed_analysis(
                analysis_run,
                error_message=exc.details.explanation,
                prompt_versions=prompt_versions,
                input_evidence_codes=input_evidence_codes,
                stage_records=stage_records,
            )
            raise
        except AnalysisStageOutputError as exc:
            stage_records[current_stage.value] = (
                self._result_persistence.build_stage_output_failure_record(exc)
            )
            self._persist_failed_analysis(
                analysis_run,
                error_message=str(exc),
                prompt_versions=prompt_versions,
                input_evidence_codes=input_evidence_codes,
                stage_records=stage_records,
            )
            raise

        try:
            self._result_persistence.persist_completed_analysis(
                analysis_run,
                bias_result=bias_result,
                validated_analysis=validated_analysis,
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
                selectinload(AnalysisRun.bias_flags),
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
        confirmed_facts = tuple(
            fact
            for fact in analysis_run.facts
            if fact.support_status is ClaimSupportStatus.SUPPORTED
        )
        unconfirmed_claims = tuple(
            fact
            for fact in analysis_run.facts
            if fact.support_status is not ClaimSupportStatus.SUPPORTED
        )
        return AnalysisPageData(
            analysis_run=analysis_run,
            summary_output=self._result_persistence.extract_summary_output(
                analysis_run.raw_response
            ),
            timeline_output=self._result_persistence.extract_timeline_output(
                analysis_run.raw_response
            ),
            critic_output=self._result_persistence.extract_critic_output(
                analysis_run.raw_response
            ),
            confirmed_facts=confirmed_facts,
            unconfirmed_claims=unconfirmed_claims,
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
        self._stage_runner.require_materially_distinct_hypotheses(result.output)
        return result

    def run_critic_stage(
        self,
        run_id: int,
        *,
        critic_context: CriticContextV1,
    ) -> AIResult[CriticOutputV1]:
        """Run a typed adversarial pass without changing original stage outputs."""
        return self._run_stage(
            run_id,
            operation="run adversarial critique",
            task_prompt=PromptName.CRITIC,
            analysis_stage=AnalysisStage.CRITIC,
            output_schema=OutputSchemaIdentifier.CRITIC_V1,
            output_type=CriticOutputV1,
            critic_context=critic_context,
        )

    def run_bias_stage(
        self,
        run_id: int,
        *,
        bias_context: BiasContextV1,
    ) -> AIResult[ReasoningRisksOutputV1]:
        """Generate typed possible reasoning risks from validated analysis context."""
        result = self._run_stage(
            run_id,
            operation="run reasoning-risk analysis",
            task_prompt=PromptName.BIAS,
            analysis_stage=AnalysisStage.BIAS,
            output_schema=OutputSchemaIdentifier.REASONING_RISKS_V1,
            output_type=ReasoningRisksOutputV1,
            bias_context=bias_context,
        )
        self._stage_runner.require_required_reasoning_risks(
            result.output,
            raw_response=result.audit.raw_response,
        )
        return result

    def _run_stage(
        self,
        run_id: int,
        *,
        operation: str,
        task_prompt: PromptName,
        analysis_stage: AnalysisStage,
        output_schema: OutputSchemaIdentifier,
        output_type: type[StageOutputT],
        critic_context: CriticContextV1 | None = None,
        bias_context: BiasContextV1 | None = None,
    ) -> AIResult[StageOutputT]:
        analysis_run = self._get_analysis_run_or_raise(run_id)
        self._require_running(analysis_run, operation=operation)
        evidence_manifest = self._stage_runner.build_evidence_manifest(analysis_run)
        return self._stage_runner.execute_stage(
            analysis_run,
            evidence_manifest,
            task_prompt=task_prompt,
            analysis_stage=analysis_stage,
            output_schema=output_schema,
            output_type=output_type,
            critic_context=critic_context,
            bias_context=bias_context,
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
        self._result_persistence.persist_failed_analysis(
            analysis_run,
            error_message=error_message,
            prompt_versions=prompt_versions,
            input_evidence_codes=input_evidence_codes,
            stage_records=stage_records,
        )

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
        if self._has_running_analysis(incident.id):
            raise AnalysisAlreadyRunningError(
                f"Incident {incident.public_id} already has a running analysis."
            )

    def _begin_analysis_start_transaction(self) -> None:
        if self.session.get_bind().dialect.name == "sqlite":
            self.session.connection().exec_driver_sql("BEGIN IMMEDIATE")

    def _has_running_analysis(self, incident_id: int) -> bool:
        running_run_id = self.session.scalar(
            select(AnalysisRun.id)
            .where(
                AnalysisRun.incident_id == incident_id,
                AnalysisRun.status == AnalysisRunStatus.RUNNING,
            )
            .limit(1)
        )
        return running_run_id is not None

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
