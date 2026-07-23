"""Structured result, audit, and transaction persistence for Phase 6 analysis."""

import json
from datetime import UTC, datetime
from typing import TypeVar

from pydantic import BaseModel, ValidationError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.models import (
    AnalysisRun,
    AnalysisRunStatus,
    BiasFlag,
    Fact,
    Hypothesis,
    IncidentStatus,
    TimelineEvent,
    utc_now,
)
from app.schemas.ai_outputs import (
    CriticOutputV1,
    HypothesesOutputV1,
    OpenQuestionsOutputV1,
    ReasoningRisksOutputV1,
    SummaryOutputV1,
    TimelineOutputV1,
)
from app.schemas.ai_provider import (
    AIResult,
    AnalysisStage,
    EvidenceReferenceValidationStatus,
    PromptName,
    ValidatedAnalysisViewV1,
    ValidatedFactV1,
    ValidatedHypothesisV1,
)
from app.services.ai_provider import AIProviderExecutionError
from app.services.analysis_stage_runner import AnalysisStageOutputError


StageOutputT = TypeVar("StageOutputT", bound=BaseModel)


class AnalysisPersistenceError(RuntimeError):
    """Raised when an analysis lifecycle write cannot be completed safely."""


class AnalysisRunTransitionError(RuntimeError):
    """Raised when a terminal or otherwise invalid transition is requested."""


class AnalysisResultPersistence:
    """Persist exact audit envelopes and validated structured analysis results."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def commit(
        self,
        analysis_run: AnalysisRun,
        *,
        failure_message: str,
    ) -> None:
        """Flush and refresh before committing one rollback-safe lifecycle unit."""
        try:
            self._session.flush()
            self._session.refresh(analysis_run)
            self._session.commit()
        except SQLAlchemyError as exc:
            self._session.rollback()
            raise AnalysisPersistenceError(failure_message) from exc

    @staticmethod
    def build_success_stage_record(
        result: AIResult[StageOutputT],
    ) -> dict[str, object]:
        """Build the exact internal audit record for one successful stage."""
        return {
            "metadata": result.metadata.model_dump(mode="json"),
            "parsed_output": result.output.model_dump(mode="json"),
            "raw_response": result.audit.raw_response,
        }

    @staticmethod
    def build_provider_failure_stage_record(
        error: AIProviderExecutionError,
    ) -> dict[str, object]:
        """Build the exact internal audit record for a provider failure."""
        audit = error.details.audit
        return {
            "failure_category": error.details.category.value,
            "raw_response": None if audit is None else audit.raw_response,
        }

    @staticmethod
    def build_stage_output_failure_record(
        error: AnalysisStageOutputError,
    ) -> dict[str, object]:
        """Build the exact internal audit record for a result-contract failure."""
        return {
            "failure_category": "stage_output_validation",
            "raw_response": error.audit_raw_response,
        }

    def persist_completed_analysis(
        self,
        analysis_run: AnalysisRun,
        *,
        bias_result: AIResult[ReasoningRisksOutputV1],
        validated_analysis: ValidatedAnalysisViewV1,
        prompt_versions: dict[str, str],
        input_evidence_codes: list[str],
        stage_records: dict[str, dict[str, object]],
    ) -> None:
        """Atomically persist structured results, audit data, and completion state."""
        analysis_run.prompt_versions = dict(prompt_versions)
        analysis_run.input_evidence_codes = list(input_evidence_codes)
        analysis_run.raw_response = self._serialize_stage_records(stage_records)
        analysis_run.facts = [
            self._build_fact(fact) for fact in validated_analysis.facts
        ]
        analysis_run.timeline_events = [
            TimelineEvent(
                event_time=self._parse_timeline_instant(event.timestamp),
                description=event.description,
                evidence_codes=list(
                    dict.fromkeys(
                        evidence.reference.evidence_id for evidence in event.evidence
                    )
                ),
                is_inferred=event.is_inferred,
                confidence=event.persisted_confidence,
            )
            for event in validated_analysis.timeline
        ]
        analysis_run.hypotheses = [
            self._build_hypothesis(hypothesis)
            for hypothesis in validated_analysis.hypotheses
        ]
        analysis_run.bias_flags = [
            BiasFlag(
                bias_type=risk.name,
                explanation=(
                    f"Location: {risk.location}\n"
                    f"Potential effect: {risk.potential_effect}"
                ),
                trigger=risk.trigger,
                mitigation=risk.mitigation,
                confidence=risk.confidence,
            )
            for risk in bias_result.output.risks
        ]
        self.require_complete_core_results(analysis_run)
        self.apply_completed_state(analysis_run)
        self.commit(
            analysis_run,
            failure_message="The completed analysis results could not be saved.",
        )

    @staticmethod
    def _build_fact(
        fact: ValidatedFactV1,
    ) -> Fact:
        supporting_excerpt = next(
            (
                evidence.reference.excerpt
                for evidence in fact.evidence
                if evidence.status is EvidenceReferenceValidationStatus.VALID
                and evidence.reference.excerpt is not None
            ),
            None,
        )
        return Fact(
            claim=fact.claim,
            support_status=fact.support_status,
            confidence=fact.confidence,
            evidence_codes=list(
                dict.fromkeys(
                    evidence.reference.evidence_id for evidence in fact.evidence
                )
            ),
            supporting_excerpt=supporting_excerpt,
        )

    @staticmethod
    def _build_hypothesis(
        hypothesis: ValidatedHypothesisV1,
    ) -> Hypothesis:
        return Hypothesis(
            rank=hypothesis.rank,
            title=hypothesis.title,
            explanation=hypothesis.explanation,
            confidence=hypothesis.adjusted_confidence,
            supporting_evidence_codes=list(
                dict.fromkeys(
                    evidence.reference.reference.evidence_id
                    for evidence in hypothesis.supporting_evidence
                )
            ),
            contradicting_evidence_codes=list(
                dict.fromkeys(
                    evidence.reference.reference.evidence_id
                    for evidence in hypothesis.contradicting_evidence
                )
            ),
            missing_evidence=list(hypothesis.missing_evidence),
            recommended_test=hypothesis.validation_test.description,
            expected_true_result=hypothesis.validation_test.expected_if_true,
            expected_false_result=hypothesis.validation_test.expected_if_false,
        )

    def persist_failed_analysis(
        self,
        analysis_run: AnalysisRun,
        *,
        error_message: str,
        prompt_versions: dict[str, str],
        input_evidence_codes: list[str],
        stage_records: dict[str, dict[str, object]],
    ) -> None:
        """Persist a failed terminal run while retaining all available audit data."""
        analysis_run.prompt_versions = dict(prompt_versions)
        analysis_run.input_evidence_codes = list(input_evidence_codes)
        analysis_run.raw_response = self._serialize_stage_records(stage_records)
        self.apply_failed_state(analysis_run, error_message=error_message)
        self.commit(
            analysis_run,
            failure_message="The failed analysis run could not be saved.",
        )

    @staticmethod
    def extract_summary_output(raw_response: str | None) -> SummaryOutputV1 | None:
        """Read only the validated summary from an internal audit envelope."""
        return AnalysisResultPersistence._extract_stage_output(
            raw_response,
            analysis_stage=AnalysisStage.SUMMARY,
            output_type=SummaryOutputV1,
        )

    @staticmethod
    def extract_critic_output(raw_response: str | None) -> CriticOutputV1 | None:
        """Read only the separate validated critic output from internal audit."""
        return AnalysisResultPersistence._extract_stage_output(
            raw_response,
            analysis_stage=AnalysisStage.CRITIC,
            output_type=CriticOutputV1,
        )

    @staticmethod
    def extract_timeline_output(raw_response: str | None) -> TimelineOutputV1 | None:
        """Read only the validated timeline from an internal audit envelope."""
        return AnalysisResultPersistence._extract_stage_output(
            raw_response,
            analysis_stage=AnalysisStage.TIMELINE,
            output_type=TimelineOutputV1,
        )

    @staticmethod
    def extract_hypotheses_output(
        raw_response: str | None,
    ) -> HypothesesOutputV1 | None:
        """Read only validated hypotheses from an internal audit envelope."""
        return AnalysisResultPersistence._extract_stage_output(
            raw_response,
            analysis_stage=AnalysisStage.HYPOTHESES,
            output_type=HypothesesOutputV1,
        )

    @staticmethod
    def extract_open_questions_output(
        raw_response: str | None,
    ) -> OpenQuestionsOutputV1 | None:
        """Read actionable open questions from the internal audit envelope."""
        return AnalysisResultPersistence._extract_stage_output(
            raw_response,
            analysis_stage=AnalysisStage.OPEN_QUESTIONS,
            output_type=OpenQuestionsOutputV1,
        )

    @staticmethod
    def _extract_stage_output(
        raw_response: str | None,
        *,
        analysis_stage: AnalysisStage,
        output_type: type[StageOutputT],
    ) -> StageOutputT | None:
        if raw_response is None:
            return None
        try:
            audit_envelope = json.loads(raw_response)
            parsed_output = audit_envelope["stages"][analysis_stage.value][
                "parsed_output"
            ]
            return output_type.model_validate(parsed_output)
        except (json.JSONDecodeError, KeyError, TypeError, ValidationError):
            return None

    @staticmethod
    def require_complete_core_results(analysis_run: AnalysisRun) -> None:
        """Require all core prompts, audits, evidence, and hypotheses before completion."""
        required_stages = {
            AnalysisStage.BIAS.value,
            AnalysisStage.CRITIC.value,
            AnalysisStage.SUMMARY.value,
            AnalysisStage.TIMELINE.value,
            AnalysisStage.HYPOTHESES.value,
            AnalysisStage.OPEN_QUESTIONS.value,
        }
        required_prompts = {
            PromptName.BIAS.value,
            PromptName.CRITIC.value,
            PromptName.SYSTEM.value,
            PromptName.SUMMARY.value,
            PromptName.TIMELINE.value,
            PromptName.HYPOTHESES.value,
            PromptName.OPEN_QUESTIONS.value,
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
            or len(analysis_run.bias_flags) < 5
        ):
            raise AnalysisRunTransitionError(
                f"Analysis run {analysis_run.id} cannot transition to COMPLETED "
                "before all required stage results are available."
            )

    @staticmethod
    def apply_completed_state(analysis_run: AnalysisRun) -> None:
        """Apply the successful terminal state to a run and its incident."""
        analysis_run.status = AnalysisRunStatus.COMPLETED
        analysis_run.completed_at = utc_now()
        analysis_run.error_message = None
        analysis_run.incident.status = IncidentStatus.COMPLETED

    @staticmethod
    def apply_failed_state(
        analysis_run: AnalysisRun,
        *,
        error_message: str,
    ) -> None:
        """Apply the failed terminal state to a run and its incident."""
        analysis_run.status = AnalysisRunStatus.FAILED
        analysis_run.completed_at = utc_now()
        analysis_run.error_message = error_message
        analysis_run.incident.status = IncidentStatus.FAILED

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
    def _parse_timeline_instant(value: str) -> datetime | None:
        candidate = f"{value[:-1]}+00:00" if value.endswith("Z") else value
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError:
            return None
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            return None
        return parsed.astimezone(UTC)
