"""Incident-scoped human-review form endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, status
from fastapi.responses import RedirectResponse
from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.database import get_db
from app.routers.validation import validation_messages
from app.schemas.review import (
    FactReviewUpdate,
    HumanNoteCreate,
    HypothesisReviewUpdate,
    TimelineReviewUpdate,
)
from app.services.review_service import (
    ReviewError,
    ReviewPersistenceError,
    ReviewService,
    ReviewTargetNotFoundError,
    ReviewTransitionError,
)
from app.success_notices import SuccessNotice, add_success_notice


router = APIRouter(
    prefix="/incidents/{public_id}/analysis/{run_id}",
    tags=["review"],
)
DatabaseSession = Annotated[Session, Depends(get_db)]


def _review_error(exc: ReviewError) -> HTTPException:
    if isinstance(exc, ReviewTargetNotFoundError):
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    if isinstance(exc, ReviewTransitionError):
        return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    if isinstance(exc, ReviewPersistenceError):
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="The human review could not be saved.",
    )


def _validation_error(exc: ValidationError) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        detail=validation_messages(exc),
    )


def _review_redirect(
    public_id: str,
    run_id: int,
    *,
    fragment: str,
) -> RedirectResponse:
    url = add_success_notice(
        f"/incidents/{public_id}/analysis/{run_id}#{fragment}",
        SuccessNotice.ANALYSIS_REVIEW_UPDATED,
    )
    return RedirectResponse(url=url, status_code=status.HTTP_303_SEE_OTHER)


@router.post("/facts/{fact_id}/review", name="review_fact")
def review_fact(
    public_id: str,
    run_id: int,
    fact_id: int,
    session: DatabaseSession,
    decision: Annotated[str, Form()],
) -> RedirectResponse:
    """Persist an accept, reject, or reclassification decision."""
    try:
        update = FactReviewUpdate(decision=decision)
        ReviewService(session).review_fact(
            public_id,
            run_id,
            fact_id,
            update,
        )
    except ValidationError as exc:
        raise _validation_error(exc) from exc
    except ReviewError as exc:
        raise _review_error(exc) from exc
    return _review_redirect(
        public_id,
        run_id,
        fragment="facts-assumptions-section",
    )


@router.post("/timeline/{event_id}/review", name="review_timeline_event")
def review_timeline_event(
    public_id: str,
    run_id: int,
    event_id: int,
    session: DatabaseSession,
    description: Annotated[str, Form()],
) -> RedirectResponse:
    """Persist a human-edited timeline description."""
    try:
        update = TimelineReviewUpdate(description=description)
        ReviewService(session).review_timeline_event(
            public_id,
            run_id,
            event_id,
            update,
        )
    except ValidationError as exc:
        raise _validation_error(exc) from exc
    except ReviewError as exc:
        raise _review_error(exc) from exc
    return _review_redirect(public_id, run_id, fragment="timeline-section")


@router.post("/hypotheses/{hypothesis_id}/review", name="review_hypothesis")
def review_hypothesis(
    public_id: str,
    run_id: int,
    hypothesis_id: int,
    session: DatabaseSession,
    confidence: Annotated[int, Form(ge=0, le=100)],
    hypothesis_status: Annotated[str, Form()],
) -> RedirectResponse:
    """Persist a human confidence override and hypothesis status."""
    try:
        update = HypothesisReviewUpdate(
            confidence=confidence,
            status=hypothesis_status,
        )
        ReviewService(session).review_hypothesis(
            public_id,
            run_id,
            hypothesis_id,
            update,
        )
    except ValidationError as exc:
        raise _validation_error(exc) from exc
    except ReviewError as exc:
        raise _review_error(exc) from exc
    return _review_redirect(public_id, run_id, fragment="hypotheses-section")


@router.post("/notes", name="add_analysis_human_note")
def add_analysis_human_note(
    public_id: str,
    run_id: int,
    session: DatabaseSession,
    note: Annotated[str, Form()],
) -> RedirectResponse:
    """Append a human-authored note to one completed analysis."""
    try:
        note_data = HumanNoteCreate(note=note)
        ReviewService(session).add_human_note(public_id, run_id, note_data)
    except ValidationError as exc:
        raise _validation_error(exc) from exc
    except ReviewError as exc:
        raise _review_error(exc) from exc
    return _review_redirect(public_id, run_id, fragment="human-notes-section")
