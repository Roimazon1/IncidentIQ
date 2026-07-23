"""Build secret-safe report input from one reviewed analysis run."""

from __future__ import annotations

import html
import json
from collections.abc import Mapping, Sequence
from hashlib import sha256

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session, joinedload

from app.models import (
    AnalysisRun,
    AnalysisRunStatus,
    ClaimSupportStatus,
    EvidenceItem,
    Fact,
    Incident,
    RecommendedAction,
    Report,
)
from app.schemas.ai_outputs import (
    CriticOutputV1,
    HypothesesOutputV1,
    OpenQuestionsOutputV1,
    PostmortemOutputV1,
    SummaryOutputV1,
    TimelineOutputV1,
)
from app.schemas.ai_provider import (
    AIRequest,
    AIResult,
    AnalysisStage,
    EvidenceReferenceValidationStatus,
    OutputSchemaIdentifier,
    PromptBundle,
    PromptName,
    PromptReference,
    PromptVersion,
    SafeAIMetadata,
    ValidatedHypothesisEvidenceV1,
    ValidatedHypothesisV1,
)
from app.schemas.evidence import EvidenceManifestSource
from app.schemas.report import (
    ReportActionInput,
    ReportAnalysisRunInput,
    ReportAssumptionAIValues,
    ReportAssumptionInput,
    ReportAssumptionSource,
    ReportCriticFindingInput,
    ReportEvidenceReference,
    ReportFactAIValues,
    ReportFactCategory,
    ReportFactInput,
    ReportHumanNoteInput,
    ReportHypothesisAIValues,
    ReportHypothesisAIReferenceInput,
    ReportHypothesisEvidenceInput,
    ReportHypothesisInput,
    ReportIncidentInput,
    ReportDraftUpdate,
    ReportInput,
    ReportOpenQuestionInput,
    ReportReasoningRiskInput,
    ReportSummaryInput,
    ReportTimelineAIValues,
    ReportTimelineEventInput,
    ReportValidationInput,
)
from app.services.analysis_persistence import AnalysisResultPersistence
from app.services.analysis_service import AnalysisPageData, AnalysisService
from app.services.ai_provider import (
    AIProvider,
    AIProviderExecutionError,
    ai_result_matches_request,
)
from app.services.evidence_manifest_service import EvidenceManifestService
from app.services.redaction_service import RedactionService
from app.services.validation_service import ValidationService


POSTMORTEM_SECTIONS = (
    ("executive_summary", "Executive summary"),
    ("incident_impact", "Incident impact"),
    ("detection", "Detection"),
    ("evidence_reviewed", "Evidence reviewed"),
    ("timeline", "Timeline"),
    ("confirmed_facts", "Confirmed facts"),
    (
        "assumptions_and_unresolved_questions",
        "Assumptions and unresolved questions",
    ),
    (
        "root_cause_hypotheses_and_confidence",
        "Root-cause hypotheses and confidence",
    ),
    (
        "supporting_and_contradicting_evidence",
        "Supporting and contradicting evidence",
    ),
    ("investigation_actions", "Investigation actions"),
    ("mitigation_and_recovery", "Mitigation and recovery"),
    ("biases_and_reasoning_risks", "Biases and reasoning risks"),
    (
        "ai_limitations_and_unsupported_claims",
        "AI limitations and unsupported claims detected",
    ),
    ("lessons_learned", "Lessons learned"),
    ("follow_up_actions", "Follow-up actions"),
)


class ReportInputUnavailableError(RuntimeError):
    """Raised when a run cannot supply complete validated report input."""


class ReportProviderRequiredError(RuntimeError):
    """Raised when postmortem generation has no injected AI provider."""


class ReportProviderExecutionError(RuntimeError):
    """Raised when the configured provider fails report generation safely."""


class ReportGenerationError(RuntimeError):
    """Raised when a provider result violates the postmortem contract."""


class ReportPersistenceError(RuntimeError):
    """Raised when a generated report cannot be saved atomically."""


class ReportNotFoundError(LookupError):
    """Raised when a report is outside the requested incident scope."""


class ReportService:
    """Compose reviewed analysis values without exposing audit or evidence bodies."""

    def __init__(
        self,
        session: Session,
        *,
        ai_provider: AIProvider | None = None,
    ) -> None:
        self._session = session
        self._analysis_service = AnalysisService(session)
        self._ai_provider = ai_provider

    def generate_draft_report(
        self,
        incident_public_id: str,
        run_id: int,
    ) -> Report:
        """Generate and persist one idempotent editable postmortem draft."""
        existing_report = self._find_existing_report(incident_public_id, run_id)
        if existing_report is not None:
            return existing_report

        report_input = self.build_report_input(incident_public_id, run_id)
        request = self._build_postmortem_request(report_input)
        provider = self._require_provider()
        try:
            result = provider.generate(request)
        except AIProviderExecutionError as exc:
            self._session.rollback()
            raise ReportProviderExecutionError(
                "The postmortem provider request failed safely."
            ) from exc
        if not ai_result_matches_request(
            result,
            request=request,
            output_type=PostmortemOutputV1,
            provider_name=report_input.analysis_run.provider_name,
            model_name=report_input.analysis_run.model_name,
        ):
            self._session.rollback()
            raise ReportGenerationError(
                "The AI provider returned an invalid postmortem output."
            )

        typed_result = AIResult[PostmortemOutputV1](
            output=result.output,
            metadata=result.metadata,
            audit=result.audit,
        )
        generated_text = self._render_postmortem(typed_result.output)
        analysis_run = self._get_scoped_analysis_run(
            incident_public_id,
            report_input.analysis_run.id,
        )
        report = Report(
            incident_id=analysis_run.incident_id,
            analysis_run_id=analysis_run.id,
            generated_text=generated_text,
            editable_text=generated_text,
            final_text=None,
            export_metadata={
                "generation": self._build_generation_metadata(
                    typed_result,
                    report_input,
                )
            },
        )
        self._session.add(report)
        try:
            self._session.commit()
            self._session.refresh(report)
        except IntegrityError as exc:
            self._session.rollback()
            existing_report = self._find_existing_report(
                incident_public_id,
                run_id,
            )
            if existing_report is not None:
                return existing_report
            raise ReportPersistenceError(
                "The generated report could not be saved."
            ) from exc
        except SQLAlchemyError as exc:
            self._session.rollback()
            raise ReportPersistenceError(
                "The generated report could not be saved."
            ) from exc
        return report

    def get_report(
        self,
        incident_public_id: str,
        report_id: int,
    ) -> Report:
        """Return one report with its incident and analysis run loaded."""
        report = self._find_scoped_report(incident_public_id, report_id)
        if report is None:
            raise ReportNotFoundError(
                f"Report {report_id} was not found for incident {incident_public_id}."
            )
        return report

    def save_report_edit(
        self,
        incident_public_id: str,
        report_id: int,
        update: ReportDraftUpdate,
    ) -> Report:
        """Persist a sanitized human draft without changing generated content."""
        report = self.get_report(incident_public_id, report_id)
        report.editable_text = self._sanitize_editable_text(update.editable_text)
        try:
            self._session.commit()
            self._session.refresh(report)
        except SQLAlchemyError as exc:
            self._session.rollback()
            raise ReportPersistenceError(
                "The edited report could not be saved."
            ) from exc
        return report

    def build_report_input(
        self,
        incident_public_id: str,
        run_id: int,
    ) -> ReportInput:
        """Return one incident-scoped, typed input for later report generation."""
        page_data = self._analysis_service.get_analysis_page_data(
            incident_public_id,
            run_id,
        )
        if page_data.analysis_run.status is not AnalysisRunStatus.COMPLETED:
            raise ReportInputUnavailableError(
                f"Analysis run {run_id} must be completed before report generation."
            )

        summary, timeline, hypotheses, critic, open_questions = (
            self._require_stage_outputs(page_data)
        )
        evidence_by_code = page_data.evidence_by_code
        analysis_run = page_data.analysis_run
        incident = analysis_run.incident
        validated_hypotheses = self._build_validated_hypotheses(
            page_data,
            summary,
            timeline,
            hypotheses,
        )

        return ReportInput(
            incident=ReportIncidentInput(
                public_id=incident.public_id,
                name=incident.name,
                affected_service=incident.affected_service,
                reported_start_time=incident.reported_start_time,
            ),
            analysis_run=ReportAnalysisRunInput(
                id=analysis_run.id,
                status=analysis_run.status,
                provider_name=analysis_run.provider_name,
                model_name=analysis_run.model_name,
                prompt_versions=dict(analysis_run.prompt_versions),
                started_at=analysis_run.started_at,
                completed_at=analysis_run.completed_at,
            ),
            summary=ReportSummaryInput(
                text=summary.summary.text,
                impact=summary.summary.impact,
                uncertainty=summary.summary.uncertainty,
                unknowns=summary.unknowns,
            ),
            evidence=self._build_evidence_inventory(
                analysis_run.input_evidence_codes,
                evidence_by_code,
            ),
            confirmed_facts=tuple(
                self._build_fact(
                    fact,
                    category=ReportFactCategory.CONFIRMED,
                    evidence_by_code=evidence_by_code,
                )
                for fact in sorted(page_data.confirmed_facts, key=lambda item: item.id)
            ),
            unconfirmed_facts=tuple(
                self._build_fact(
                    fact,
                    category=ReportFactCategory.UNCONFIRMED,
                    evidence_by_code=evidence_by_code,
                )
                for fact in sorted(
                    page_data.unconfirmed_claims,
                    key=lambda item: item.id,
                )
            ),
            assumptions=self._build_assumptions(
                page_data,
                summary,
                evidence_by_code,
            ),
            timeline=self._build_timeline(page_data, timeline, evidence_by_code),
            hypotheses=self._build_hypotheses(
                page_data,
                hypotheses,
                validated_hypotheses,
                evidence_by_code,
            ),
            actions=tuple(
                self._build_action(action, evidence_by_code)
                for action in sorted(analysis_run.actions, key=lambda item: item.id)
            ),
            reasoning_risks=tuple(
                ReportReasoningRiskInput(
                    risk_id=risk.id,
                    name=risk.bias_type,
                    explanation=risk.explanation,
                    trigger=risk.trigger,
                    mitigation=risk.mitigation,
                    confidence=risk.confidence,
                )
                for risk in sorted(analysis_run.bias_flags, key=lambda item: item.id)
            ),
            open_questions=tuple(
                ReportOpenQuestionInput(
                    question=question.question,
                    source_kind=question.source_kind,
                    source_reference=question.source_reference,
                    rationale=question.rationale,
                    evidence_needed=question.evidence_needed,
                    resolution_criteria=question.resolution_criteria,
                )
                for question in open_questions.questions
            ),
            human_notes=tuple(
                ReportHumanNoteInput(
                    note_id=note.id,
                    note=note.note,
                    created_at=note.created_at,
                )
                for note in sorted(
                    analysis_run.human_notes,
                    key=lambda item: (item.created_at, item.id),
                )
            ),
            validation=self._build_validation(
                page_data,
                critic,
                evidence_by_code,
            ),
        )

    def _find_existing_report(
        self,
        incident_public_id: str,
        run_id: int,
    ) -> Report | None:
        return self._session.scalar(
            select(Report)
            .join(Report.analysis_run)
            .join(AnalysisRun.incident)
            .where(
                Report.analysis_run_id == run_id,
                Report.incident_id == AnalysisRun.incident_id,
                Incident.public_id == incident_public_id,
            )
        )

    def _find_scoped_report(
        self,
        incident_public_id: str,
        report_id: int,
    ) -> Report | None:
        return self._session.scalar(
            select(Report)
            .join(Report.incident)
            .join(Report.analysis_run)
            .options(
                joinedload(Report.incident),
                joinedload(Report.analysis_run),
            )
            .where(
                Report.id == report_id,
                Incident.public_id == incident_public_id,
                Report.incident_id == AnalysisRun.incident_id,
            )
        )

    def _get_scoped_analysis_run(
        self,
        incident_public_id: str,
        run_id: int,
    ) -> AnalysisRun:
        analysis_run = self._session.scalar(
            select(AnalysisRun)
            .join(AnalysisRun.incident)
            .where(
                AnalysisRun.id == run_id,
                Incident.public_id == incident_public_id,
            )
        )
        if analysis_run is None:
            raise ReportInputUnavailableError(
                f"Analysis run {run_id} was not found for incident "
                f"{incident_public_id}."
            )
        return analysis_run

    def _require_provider(self) -> AIProvider:
        if self._ai_provider is None:
            raise ReportProviderRequiredError(
                "A configured AI provider is required to generate a report."
            )
        return self._ai_provider

    @staticmethod
    def _build_postmortem_request(report_input: ReportInput) -> AIRequest:
        return AIRequest(
            report_input=report_input,
            prompts=PromptBundle(
                system=PromptReference(
                    name=PromptName.SYSTEM,
                    version=PromptVersion.V1,
                ),
                task=PromptReference(
                    name=PromptName.POSTMORTEM,
                    version=PromptVersion.V2,
                ),
            ),
            output_schema=OutputSchemaIdentifier.POSTMORTEM_V1,
            metadata=SafeAIMetadata(
                request_identifier=(
                    f"analysis-run-{report_input.analysis_run.id}-postmortem"
                ),
                incident_public_identifier=report_input.incident.public_id,
                analysis_stage=AnalysisStage.POSTMORTEM,
            ),
        )

    @classmethod
    def _render_postmortem(cls, output: PostmortemOutputV1) -> str:
        lines = ["# Incident Postmortem"]
        for position, (field_name, heading) in enumerate(
            POSTMORTEM_SECTIONS,
            start=1,
        ):
            section_text = cls._sanitize_generated_text(getattr(output, field_name))
            lines.extend(("", f"## {position}. {heading}", "", section_text))
        return "\n".join(lines).strip() + "\n"

    @staticmethod
    def _sanitize_generated_text(value: str) -> str:
        normalized = value.replace("\r\n", "\n").replace("\r", "\n")
        without_controls = "".join(
            character
            for character in normalized
            if character in {"\n", "\t"} or ord(character) >= 32
        )
        redacted = RedactionService.redact_text(without_controls).redacted_text
        return html.escape(redacted.strip(), quote=False)

    @classmethod
    def _sanitize_editable_text(cls, value: str) -> str:
        return cls._sanitize_generated_text(html.unescape(value))

    @staticmethod
    def _build_generation_metadata(
        result: AIResult[PostmortemOutputV1],
        report_input: ReportInput,
    ) -> dict[str, object]:
        input_payload = json.dumps(
            report_input.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        metadata = result.metadata
        return {
            "provider_name": metadata.provider_name,
            "model_name": metadata.model_name,
            "system_prompt": {
                "name": metadata.system_prompt.name.value,
                "version": metadata.system_prompt.version.value,
            },
            "task_prompt": {
                "name": metadata.task_prompt.name.value,
                "version": metadata.task_prompt.version.value,
            },
            "output_schema": metadata.output_schema.value,
            "request_identifier": metadata.request_identifier,
            "attempt_count": metadata.attempt_count,
            "report_input_sha256": sha256(input_payload.encode("utf-8")).hexdigest(),
        }

    @staticmethod
    def _require_stage_outputs(
        page_data: AnalysisPageData,
    ) -> tuple[
        SummaryOutputV1,
        TimelineOutputV1,
        HypothesesOutputV1,
        CriticOutputV1,
        OpenQuestionsOutputV1,
    ]:
        hypotheses_output = AnalysisResultPersistence.extract_hypotheses_output(
            page_data.analysis_run.raw_response
        )
        outputs = (
            page_data.summary_output,
            page_data.timeline_output,
            hypotheses_output,
            page_data.critic_output,
            page_data.open_questions_output,
        )
        if any(output is None for output in outputs):
            raise ReportInputUnavailableError(
                "The completed analysis run is missing validated structured report data."
            )
        summary, timeline, hypotheses, critic, open_questions = outputs
        return summary, timeline, hypotheses, critic, open_questions

    @staticmethod
    def _build_validated_hypotheses(
        page_data: AnalysisPageData,
        summary: SummaryOutputV1,
        timeline: TimelineOutputV1,
        hypotheses: HypothesesOutputV1,
    ) -> tuple[ValidatedHypothesisV1, ...]:
        evidence_sources = tuple(
            EvidenceManifestSource(
                evidence_code=evidence.evidence_code,
                source_name=evidence.source_name,
                evidence_type=evidence.evidence_type,
                original_text=evidence.original_text,
            )
            for evidence_code in page_data.analysis_run.input_evidence_codes
            if (evidence := page_data.evidence_by_code.get(evidence_code)) is not None
        )
        if not evidence_sources:
            raise ReportInputUnavailableError(
                "The analysis run evidence snapshot is unavailable for validation."
            )
        evidence_manifest = EvidenceManifestService.build_evidence_manifest(
            page_data.analysis_run.incident.public_id,
            evidence_sources,
        )
        return ValidationService.build_validated_analysis_view(
            summary,
            timeline,
            hypotheses,
            evidence_manifest,
        ).hypotheses

    @classmethod
    def _build_evidence_inventory(
        cls,
        evidence_codes: Sequence[str],
        evidence_by_code: Mapping[str, EvidenceItem],
    ) -> tuple[ReportEvidenceReference, ...]:
        return cls._build_evidence_references(evidence_codes, evidence_by_code)

    @staticmethod
    def _build_evidence_reference(
        evidence_code: str,
        evidence_by_code: Mapping[str, EvidenceItem],
    ) -> ReportEvidenceReference:
        evidence = evidence_by_code.get(evidence_code)
        return ReportEvidenceReference(
            evidence_code=evidence_code,
            available=evidence is not None,
            source_name=None if evidence is None else evidence.source_name,
            evidence_type=None if evidence is None else evidence.evidence_type,
        )

    @classmethod
    def _build_evidence_references(
        cls,
        evidence_codes: Sequence[str],
        evidence_by_code: Mapping[str, EvidenceItem],
    ) -> tuple[ReportEvidenceReference, ...]:
        return tuple(
            cls._build_evidence_reference(evidence_code, evidence_by_code)
            for evidence_code in dict.fromkeys(evidence_codes)
        )

    @classmethod
    def _build_fact(
        cls,
        fact: Fact,
        *,
        category: ReportFactCategory,
        evidence_by_code: Mapping[str, EvidenceItem],
    ) -> ReportFactInput:
        ai_original = cls._build_fact_ai_values(fact)
        return ReportFactInput(
            fact_id=fact.id,
            claim=fact.claim,
            confidence=fact.confidence,
            category=category,
            human_status=fact.human_status,
            evidence=cls._build_evidence_references(
                fact.evidence_codes,
                evidence_by_code,
            ),
            ai_original=ai_original,
        )

    @staticmethod
    def _build_fact_ai_values(fact: Fact) -> ReportFactAIValues:
        return ReportFactAIValues(
            claim=fact.claim,
            support_status=fact.support_status,
            confidence=fact.confidence,
            evidence_codes=tuple(fact.evidence_codes),
        )

    @classmethod
    def _build_assumptions(
        cls,
        page_data: AnalysisPageData,
        summary: SummaryOutputV1,
        evidence_by_code: Mapping[str, EvidenceItem],
    ) -> tuple[ReportAssumptionInput, ...]:
        ai_assumptions = tuple(
            ReportAssumptionInput(
                claim=assumption.claim,
                reason=assumption.reason,
                required_evidence=assumption.required_evidence,
                source=ReportAssumptionSource.AI_IDENTIFIED,
                originating_fact_id=None,
                evidence=(),
                ai_original=ReportAssumptionAIValues(
                    claim=assumption.claim,
                    reason=assumption.reason,
                    required_evidence=assumption.required_evidence,
                    fact_support_status=None,
                    fact_confidence=None,
                    fact_evidence_codes=(),
                ),
            )
            for assumption in summary.assumptions
        )
        reclassified_assumptions = tuple(
            ReportAssumptionInput(
                claim=fact.claim,
                reason="Reclassified by an explicit human review decision.",
                required_evidence=(),
                source=ReportAssumptionSource.HUMAN_RECLASSIFIED_FACT,
                originating_fact_id=fact.id,
                evidence=cls._build_evidence_references(
                    fact.evidence_codes,
                    evidence_by_code,
                ),
                ai_original=ReportAssumptionAIValues(
                    claim=fact.claim,
                    reason=None,
                    required_evidence=(),
                    fact_support_status=fact.support_status,
                    fact_confidence=fact.confidence,
                    fact_evidence_codes=tuple(fact.evidence_codes),
                ),
            )
            for fact in sorted(page_data.reclassified_facts, key=lambda item: item.id)
        )
        return ai_assumptions + reclassified_assumptions

    @classmethod
    def _build_timeline(
        cls,
        page_data: AnalysisPageData,
        timeline: TimelineOutputV1,
        evidence_by_code: Mapping[str, EvidenceItem],
    ) -> tuple[ReportTimelineEventInput, ...]:
        persisted_events = sorted(
            page_data.analysis_run.timeline_events,
            key=lambda item: item.id,
        )
        if len(persisted_events) != len(timeline.events):
            raise ReportInputUnavailableError(
                "The persisted timeline does not match its validated AI output."
            )

        report_events: list[ReportTimelineEventInput] = []
        for event, ai_event in zip(persisted_events, timeline.events, strict=True):
            if (
                event.description != ai_event.description
                or event.is_inferred is not ai_event.is_inferred
            ):
                raise ReportInputUnavailableError(
                    "The persisted timeline cannot be aligned with its AI original."
                )
            human_description = (
                None if event.human_review is None else event.human_review.description
            )
            report_events.append(
                ReportTimelineEventInput(
                    event_id=event.id,
                    event_time=event.event_time,
                    description=human_description or event.description,
                    confidence=event.confidence,
                    is_inferred=event.is_inferred,
                    uncertainty=ai_event.uncertainty_explanation,
                    evidence=cls._build_evidence_references(
                        event.evidence_codes,
                        evidence_by_code,
                    ),
                    has_human_override=human_description is not None,
                    ai_original=ReportTimelineAIValues(
                        timestamp=ai_event.timestamp,
                        description=ai_event.description,
                        confidence=ai_event.confidence,
                    ),
                )
            )
        return tuple(report_events)

    @classmethod
    def _build_hypotheses(
        cls,
        page_data: AnalysisPageData,
        hypotheses: HypothesesOutputV1,
        validated_hypotheses: tuple[ValidatedHypothesisV1, ...],
        evidence_by_code: Mapping[str, EvidenceItem],
    ) -> tuple[ReportHypothesisInput, ...]:
        original_by_rank = {
            hypothesis.rank: hypothesis for hypothesis in hypotheses.hypotheses
        }
        validated_by_rank = {
            hypothesis.rank: hypothesis for hypothesis in validated_hypotheses
        }
        if len(original_by_rank) != len(page_data.analysis_run.hypotheses) or len(
            validated_by_rank
        ) != len(page_data.analysis_run.hypotheses):
            raise ReportInputUnavailableError(
                "The persisted hypotheses do not match their validated AI output."
            )

        report_hypotheses: list[ReportHypothesisInput] = []
        for hypothesis in sorted(
            page_data.analysis_run.hypotheses,
            key=lambda item: item.rank,
        ):
            ai_original = original_by_rank.get(hypothesis.rank)
            validated = validated_by_rank.get(hypothesis.rank)
            if (
                ai_original is None
                or validated is None
                or ai_original.title != hypothesis.title
                or ai_original.explanation != hypothesis.explanation
            ):
                raise ReportInputUnavailableError(
                    "A persisted hypothesis cannot be aligned with its AI original."
                )
            confidence_override = hypothesis.confidence_override
            report_hypotheses.append(
                ReportHypothesisInput(
                    hypothesis_id=hypothesis.id,
                    rank=hypothesis.rank,
                    title=hypothesis.title,
                    explanation=hypothesis.explanation,
                    confidence=(
                        hypothesis.confidence
                        if confidence_override is None
                        else confidence_override.confidence
                    ),
                    validated_ai_confidence=hypothesis.confidence,
                    has_human_confidence_override=confidence_override is not None,
                    human_status=hypothesis.status,
                    supporting_evidence=cls._build_effective_hypothesis_evidence(
                        validated.supporting_evidence,
                        evidence_by_code,
                    ),
                    contradicting_evidence=(
                        cls._build_effective_hypothesis_evidence(
                            validated.contradicting_evidence,
                            evidence_by_code,
                        )
                    ),
                    missing_evidence=tuple(hypothesis.missing_evidence),
                    validation_test=hypothesis.recommended_test,
                    expected_if_true=hypothesis.expected_true_result,
                    expected_if_false=hypothesis.expected_false_result,
                    ai_original=ReportHypothesisAIValues(
                        title=ai_original.title,
                        explanation=ai_original.explanation,
                        confidence=ai_original.confidence,
                        risk_of_acting=ai_original.risk_of_acting,
                        supporting_evidence=(
                            cls._build_original_hypothesis_evidence(
                                validated.supporting_evidence
                            )
                        ),
                        contradicting_evidence=(
                            cls._build_original_hypothesis_evidence(
                                validated.contradicting_evidence
                            )
                        ),
                    ),
                )
            )
        return tuple(report_hypotheses)

    @classmethod
    def _build_effective_hypothesis_evidence(
        cls,
        evidence_items: tuple[ValidatedHypothesisEvidenceV1, ...],
        evidence_by_code: Mapping[str, EvidenceItem],
    ) -> tuple[ReportHypothesisEvidenceInput, ...]:
        return tuple(
            ReportHypothesisEvidenceInput(
                reference=cls._build_evidence_reference(
                    evidence.reference.reference.evidence_id,
                    evidence_by_code,
                ),
                line_range=evidence.reference.reference.line_range,
                relevance=evidence.relevance,
            )
            for evidence in evidence_items
            if evidence.reference.status is EvidenceReferenceValidationStatus.VALID
        )

    @staticmethod
    def _build_original_hypothesis_evidence(
        evidence_items: tuple[ValidatedHypothesisEvidenceV1, ...],
    ) -> tuple[ReportHypothesisAIReferenceInput, ...]:
        return tuple(
            ReportHypothesisAIReferenceInput(
                evidence_code=evidence.reference.reference.evidence_id,
                line_range=evidence.reference.reference.line_range,
                relevance=evidence.relevance,
                validation_status=evidence.reference.status,
                validation_message=evidence.reference.message,
            )
            for evidence in evidence_items
        )

    @classmethod
    def _build_action(
        cls,
        action: RecommendedAction,
        evidence_by_code: Mapping[str, EvidenceItem],
    ) -> ReportActionInput:
        return ReportActionInput(
            action_id=action.id,
            description=action.description,
            priority=action.priority,
            owner_role=action.owner_role,
            expected_information=action.expected_information,
            operational_risk=action.operational_risk,
            evidence=cls._build_evidence_references(
                action.evidence_codes,
                evidence_by_code,
            ),
            linked_hypothesis_ranks=tuple(
                sorted(hypothesis.rank for hypothesis in action.hypotheses)
            ),
        )

    @classmethod
    def _build_validation(
        cls,
        page_data: AnalysisPageData,
        critic: CriticOutputV1,
        evidence_by_code: Mapping[str, EvidenceItem],
    ) -> ReportValidationInput:
        return ReportValidationInput(
            claim_support_counts=dict(page_data.validation_summary.claim_status_counts),
            inferred_timeline_events=(
                page_data.validation_summary.inferred_timeline_events
            ),
            hypotheses_with_contradictions=(
                page_data.validation_summary.hypotheses_with_contradictions
            ),
            unavailable_evidence_codes=(
                page_data.validation_summary.unavailable_evidence_codes
            ),
            unsupported_fact_ids=tuple(
                fact.id
                for fact in sorted(
                    page_data.analysis_run.facts,
                    key=lambda item: item.id,
                )
                if fact.support_status is ClaimSupportStatus.UNSUPPORTED
            ),
            critic_findings=tuple(
                ReportCriticFindingInput(
                    concern=finding.concern,
                    affected_claim=finding.affected_claim,
                    evidence=tuple(
                        cls._build_evidence_reference(
                            reference.evidence_id,
                            evidence_by_code,
                        )
                        for reference in finding.evidence
                    ),
                    impact=finding.impact,
                    recommendation=finding.recommendation,
                )
                for finding in critic.findings
            ),
            critic_ignored_evidence=tuple(
                cls._build_evidence_reference(
                    reference.evidence_id,
                    evidence_by_code,
                )
                for reference in critic.ignored_evidence
            ),
            hypothesis_ranking_rationale=critic.ranking_rationale,
        )
