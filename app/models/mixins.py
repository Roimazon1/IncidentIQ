"""Shared SQLAlchemy model conventions."""

from datetime import UTC, datetime

from sqlalchemy.orm import Mapped, mapped_column

from app.models.types import UTCDateTime


def utc_now() -> datetime:
    """Return the current time as an aware UTC datetime."""

    return datetime.now(UTC)


class TimestampMixin:
    """Add non-null UTC creation and update timestamps to a model."""

    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        default=utc_now,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )
