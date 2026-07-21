"""Generated and human-editable incident report persistence model."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import JSON, ForeignKey, Text
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.models.mixins import TimestampMixin

if TYPE_CHECKING:
    from app.models.analysis import AnalysisRun
    from app.models.incident import Incident


class Report(TimestampMixin, Base):
    """A generated report draft and its separately preserved human edits."""

    __tablename__ = "reports"

    id: Mapped[int] = mapped_column(primary_key=True)
    incident_id: Mapped[int] = mapped_column(
        ForeignKey("incidents.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    analysis_run_id: Mapped[int] = mapped_column(
        ForeignKey("analysis_runs.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    generated_text: Mapped[str] = mapped_column(Text, nullable=False)
    editable_text: Mapped[str] = mapped_column(Text, nullable=False)
    final_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    export_metadata: Mapped[dict[str, Any]] = mapped_column(
        MutableDict.as_mutable(JSON),
        default=dict,
        nullable=False,
    )

    incident: Mapped[Incident] = relationship(back_populates="reports")
    analysis_run: Mapped[AnalysisRun] = relationship(back_populates="reports")
