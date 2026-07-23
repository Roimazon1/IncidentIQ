"""Transactional human-review operations for persisted analysis results."""

from typing import TypeVar

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.models import (
    AnalysisRun,
    AnalysisRunStatus,
    Fact,
    HumanNote,
    Hypothesis,
    HypothesisConfidenceOverride,
    Incident,
    TimelineEvent,
    TimelineEventReview,
)
from app.schemas.review import (
    FactReviewUpdate,
    HumanNoteCreate,
    HypothesisReviewUpdate,
    TimelineReviewUpdate,
)


ReviewTargetT = TypeVar("ReviewTargetT", Fact, TimelineEvent, Hypothesis)


class ReviewError(RuntimeError):
    """Base class for safe human-review failures."""


class ReviewTargetNotFoundError(ReviewError):
    """Raised when a run or target is outside the requested incident scope."""


class ReviewTransitionError(ReviewError):
    """Raised when a run cannot accept human review."""


class ReviewPersistenceError(ReviewError):
    """Raised when a human decision cannot be persisted safely."""


class ReviewService:
    """Persist human decisions without overwriting AI-generated values."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def review_fact(
        self,
        incident_public_id: str,
        run_id: int,
        fact_id: int,
        update: FactReviewUpdate,
    ) -> Fact:
        """Persist one human fact decision within an incident-scoped run."""
        self._require_reviewable_run(incident_public_id, run_id)
        fact = self._get_scoped_target(Fact, fact_id, run_id, label="Fact")
        fact.human_status = update.decision
        self._commit("The fact review could not be saved.")
        return fact

    def review_timeline_event(
        self,
        incident_public_id: str,
        run_id: int,
        event_id: int,
        update: TimelineReviewUpdate,
    ) -> TimelineEventReview:
        """Persist a human description while retaining the AI description."""
        self._require_reviewable_run(incident_public_id, run_id)
        event = self._get_scoped_target(
            TimelineEvent,
            event_id,
            run_id,
            label="Timeline event",
        )
        review = self.session.scalar(
            select(TimelineEventReview).where(
                TimelineEventReview.timeline_event_id == event.id
            )
        )
        if review is None:
            review = TimelineEventReview(
                timeline_event=event,
                description=update.description,
            )
            self.session.add(review)
        else:
            review.description = update.description
        self._commit("The timeline review could not be saved.")
        return review

    def review_hypothesis(
        self,
        incident_public_id: str,
        run_id: int,
        hypothesis_id: int,
        update: HypothesisReviewUpdate,
    ) -> Hypothesis:
        """Persist human status and confidence separately from AI confidence."""
        self._require_reviewable_run(incident_public_id, run_id)
        hypothesis = self._get_scoped_target(
            Hypothesis,
            hypothesis_id,
            run_id,
            label="Hypothesis",
        )
        confidence_override = self.session.scalar(
            select(HypothesisConfidenceOverride).where(
                HypothesisConfidenceOverride.hypothesis_id == hypothesis.id
            )
        )
        if confidence_override is None:
            confidence_override = HypothesisConfidenceOverride(
                hypothesis=hypothesis,
                confidence=update.confidence,
            )
            self.session.add(confidence_override)
        else:
            confidence_override.confidence = update.confidence
        hypothesis.status = update.status
        self._commit("The hypothesis review could not be saved.")
        return hypothesis

    def add_human_note(
        self,
        incident_public_id: str,
        run_id: int,
        note_data: HumanNoteCreate,
    ) -> HumanNote:
        """Append one human-authored note to a completed analysis run."""
        analysis_run = self._require_reviewable_run(incident_public_id, run_id)
        note = HumanNote(analysis_run=analysis_run, note=note_data.note)
        self.session.add(note)
        self._commit("The human note could not be saved.")
        return note

    def _require_reviewable_run(
        self,
        incident_public_id: str,
        run_id: int,
    ) -> AnalysisRun:
        analysis_run = self.session.scalar(
            select(AnalysisRun)
            .join(AnalysisRun.incident)
            .where(
                AnalysisRun.id == run_id,
                Incident.public_id == incident_public_id,
            )
        )
        if analysis_run is None:
            raise ReviewTargetNotFoundError(
                f"Analysis run {run_id} was not found for incident "
                f"{incident_public_id}."
            )
        if analysis_run.status is not AnalysisRunStatus.COMPLETED:
            raise ReviewTransitionError(
                f"Analysis run {run_id} must be completed before human review."
            )
        return analysis_run

    def _get_scoped_target(
        self,
        model: type[ReviewTargetT],
        target_id: int,
        run_id: int,
        *,
        label: str,
    ) -> ReviewTargetT:
        target = self.session.scalar(
            select(model).where(
                model.id == target_id,
                model.analysis_run_id == run_id,
            )
        )
        if target is None:
            raise ReviewTargetNotFoundError(
                f"{label} {target_id} was not found in analysis run {run_id}."
            )
        return target

    def _commit(self, failure_message: str) -> None:
        try:
            self.session.commit()
        except SQLAlchemyError as exc:
            self.session.rollback()
            raise ReviewPersistenceError(failure_message) from exc
