"""Validated form contracts for human analysis review."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from app.models import FactReviewStatus, HypothesisStatus


MAX_HUMAN_NOTE_LENGTH = 4000
MAX_TIMELINE_DESCRIPTION_LENGTH = 5000
ReviewText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]


class ReviewFormModel(BaseModel):
    """Reject unknown form values at the review boundary."""

    model_config = ConfigDict(extra="forbid")


class FactReviewUpdate(ReviewFormModel):
    """A final human decision about one AI-generated fact."""

    decision: FactReviewStatus

    @field_validator("decision")
    @classmethod
    def reject_pending_decision(
        cls,
        value: FactReviewStatus,
    ) -> FactReviewStatus:
        if value is FactReviewStatus.PENDING:
            raise ValueError("Select a human fact-review decision.")
        return value


class TimelineReviewUpdate(ReviewFormModel):
    """A human replacement description for one timeline event."""

    description: Annotated[
        ReviewText,
        Field(max_length=MAX_TIMELINE_DESCRIPTION_LENGTH),
    ]


class HypothesisReviewUpdate(ReviewFormModel):
    """A human status and confidence decision for one hypothesis."""

    confidence: int = Field(ge=0, le=100)
    status: HypothesisStatus


class HumanNoteCreate(ReviewFormModel):
    """A human-authored analysis note."""

    note: Annotated[
        ReviewText,
        Field(max_length=MAX_HUMAN_NOTE_LENGTH),
    ]
