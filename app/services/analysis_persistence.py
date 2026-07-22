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
    Fact,
    Hypothesis,
    IncidentStatus,
    TimelineEvent,
    utc_now,
)
from app.schemas.ai_outputs import (
    CriticOutputV1,
    FactItemV1,
    HypothesisV1,
    HypothesesOutputV1,
    SummaryOutputV1,
    TimelineOutputV1,
)
from app.schemas.ai_provider import AIResult, AnalysisStage, PromptName
from app.schemas.evidence import EvidenceManifest
from app.services.ai_provider import AIProviderExecutionError
from app.services.analysis_stage_runner import AnalysisStageOutputError
from app.services.validation_service import ValidationService


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
        summary_result: AIResult[SummaryOutputV1],
        timeline_result: AIResult[TimelineOutputV1],
        hypotheses_result: AIResult[HypothesesOutputV1],
        evidence_manifest: EvidenceManifest,
        prompt_versions: dict[str, str],
        input_evidence_codes: list[str],
        stage_records: dict[str, dict[str, object]],
    ) -> None:
        """Atomically persist structured results, audit data, and completion state."""
        analysis_run.prompt_versions = dict(prompt_versions)
        analysis_run.input_evidence_codes = list(input_evidence_codes)
        analysis_run.raw_response = self._serialize_stage_records(stage_records)
        analysis_run.facts = [
            self._build_fact(fact, evidence_manifest)
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
                confidence=(
                    ValidationService.apply_inferred_timeline_confidence_cap(
                        event.confidence,
                        event.is_inferred,
                    )
                ),
            )
            for event in timeline_result.output.events
        ]
        analysis_run.hypotheses = [
            self._build_hypothesis(hypothesis, evidence_manifest)
            for hypothesis in hypotheses_result.output.hypotheses
        ]
        self.require_complete_core_results(analysis_run)
        self.apply_completed_state(analysis_run)
        self.commit(
            analysis_run,
            failure_message="The completed analysis results could not be saved.",
        )

    @staticmethod
    def _build_fact(
        fact: FactItemV1,
        evidence_manifest: EvidenceManifest,
    ) -> Fact:
        outcomes = ValidationService.validate_output_references(
            fact,
            evidence_manifest,
        )
        supporting_excerpt = next(
            (
                reference.excerpt
                for reference, outcome in zip(
                    fact.evidence,
                    outcomes,
                    strict=True,
                )
                if outcome.is_valid and reference.excerpt is not None
            ),
            None,
        )
        return Fact(
            claim=fact.claim,
            support_status=ValidationService.classify_claim_support(outcomes),
            confidence=fact.confidence,
            evidence_codes=list(
                dict.fromkeys(reference.evidence_id for reference in fact.evidence)
            ),
            supporting_excerpt=supporting_excerpt,
        )

    @staticmethod
    def _build_hypothesis(
        hypothesis: HypothesisV1,
        evidence_manifest: EvidenceManifest,
    ) -> Hypothesis:
        contradiction_outcomes = tuple(
            ValidationService.validate_supporting_excerpt(
                evidence.reference,
                evidence_manifest,
            )
            for evidence in hypothesis.contradicting_evidence
        )
        return Hypothesis(
            rank=hypothesis.rank,
            title=hypothesis.title,
            explanation=hypothesis.explanation,
            confidence=(
                ValidationService.adjust_hypothesis_confidence_for_contradictions(
                    hypothesis.confidence,
                    contradiction_outcomes,
                )
            ),
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
            AnalysisStage.CRITIC.value,
            AnalysisStage.SUMMARY.value,
            AnalysisStage.TIMELINE.value,
            AnalysisStage.HYPOTHESES.value,
        }
        required_prompts = {
            PromptName.CRITIC.value,
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
