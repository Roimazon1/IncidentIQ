"""Incident persistence model."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Enum as SQLAlchemyEnum
from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.models.enums import IncidentStatus
from app.models.mixins import TimestampMixin
from app.models.types import UTCDateTime

if TYPE_CHECKING:
    from app.models.analysis import AnalysisRun
    from app.models.evidence import EvidenceItem


INCIDENT_PUBLIC_ID_LENGTH = 10
INCIDENT_NAME_MAX_LENGTH = 200
AFFECTED_SERVICE_MAX_LENGTH = 200


class Incident(TimestampMixin, Base):
    """An incident investigation and its current lifecycle state."""

    __tablename__ = "incidents"

    id: Mapped[int] = mapped_column(primary_key=True)
    public_id: Mapped[str] = mapped_column(
        String(INCIDENT_PUBLIC_ID_LENGTH),
        unique=True,
        index=True,
        nullable=False,
    )
    name: Mapped[str] = mapped_column(
        String(INCIDENT_NAME_MAX_LENGTH),
        nullable=False,
    )
    description: Mapped[str] = mapped_column(Text, nullable=False)
    affected_service: Mapped[str] = mapped_column(
        String(AFFECTED_SERVICE_MAX_LENGTH),
        nullable=False,
    )
    reported_start_time: Mapped[datetime | None] = mapped_column(
        UTCDateTime(),
        nullable=True,
    )
    status: Mapped[IncidentStatus] = mapped_column(
        SQLAlchemyEnum(
            IncidentStatus,
            name="incident_status",
            native_enum=False,
            create_constraint=True,
            validate_strings=True,
        ),
        default=IncidentStatus.DRAFT,
        nullable=False,
    )

    evidence_items: Mapped[list[EvidenceItem]] = relationship(
        back_populates="incident",
        cascade="all, delete-orphan",
        single_parent=True,
    )
    analysis_runs: Mapped[list[AnalysisRun]] = relationship(
        back_populates="incident",
        cascade="all, delete-orphan",
        single_parent=True,
    )
