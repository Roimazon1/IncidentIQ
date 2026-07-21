"""Incident persistence model."""

from datetime import datetime

from sqlalchemy import Enum as SQLAlchemyEnum
from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.models.enums import IncidentStatus
from app.models.mixins import TimestampMixin
from app.models.types import UTCDateTime


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
