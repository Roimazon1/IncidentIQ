"""Evidence item persistence model."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Enum as SQLAlchemyEnum
from sqlalchemy import ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.models.enums import EvidenceType
from app.models.mixins import TimestampMixin
from app.models.types import UTCDateTime

if TYPE_CHECKING:
    from app.models.incident import Incident


EVIDENCE_CODE_LENGTH = 5
SOURCE_NAME_MAX_LENGTH = 255
SHA256_HEX_LENGTH = 64


class EvidenceItem(TimestampMixin, Base):
    """A traceable source of evidence belonging to one incident."""

    __tablename__ = "evidence_items"
    __table_args__ = (
        UniqueConstraint(
            "incident_id",
            "evidence_code",
            name="uq_evidence_items_incident_code",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    incident_id: Mapped[int] = mapped_column(
        ForeignKey("incidents.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    evidence_code: Mapped[str] = mapped_column(
        String(EVIDENCE_CODE_LENGTH),
        nullable=False,
    )
    source_name: Mapped[str] = mapped_column(
        String(SOURCE_NAME_MAX_LENGTH),
        nullable=False,
    )
    evidence_type: Mapped[EvidenceType] = mapped_column(
        SQLAlchemyEnum(
            EvidenceType,
            name="evidence_type",
            native_enum=False,
            create_constraint=True,
            validate_strings=True,
        ),
        nullable=False,
    )
    original_text: Mapped[str] = mapped_column(Text, nullable=False)
    redacted_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    checksum: Mapped[str] = mapped_column(
        String(SHA256_HEX_LENGTH),
        nullable=False,
    )
    detected_start_time: Mapped[datetime | None] = mapped_column(
        UTCDateTime(),
        nullable=True,
    )
    detected_end_time: Mapped[datetime | None] = mapped_column(
        UTCDateTime(),
        nullable=True,
    )

    incident: Mapped[Incident] = relationship(back_populates="evidence_items")
