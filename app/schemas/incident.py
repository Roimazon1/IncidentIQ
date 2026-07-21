"""Pydantic contracts for incident creation, updates, and reads."""

from typing import Annotated

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
)

from app.models.enums import IncidentStatus
from app.models.incident import (
    AFFECTED_SERVICE_MAX_LENGTH,
    INCIDENT_NAME_MAX_LENGTH,
    INCIDENT_PUBLIC_ID_LENGTH,
)


IncidentName = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=INCIDENT_NAME_MAX_LENGTH,
    ),
]
IncidentDescription = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]
AffectedService = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=AFFECTED_SERVICE_MAX_LENGTH,
    ),
]
IncidentPublicId = Annotated[
    str,
    StringConstraints(
        min_length=INCIDENT_PUBLIC_ID_LENGTH,
        max_length=INCIDENT_PUBLIC_ID_LENGTH,
        pattern=r"^INC-\d{6}$",
    ),
]


class IncidentCreate(BaseModel):
    """User-supplied values for a new draft incident."""

    name: IncidentName
    description: IncidentDescription
    affected_service: AffectedService
    reported_start_time: AwareDatetime | None = None

    model_config = ConfigDict(extra="forbid")


class IncidentUpdate(BaseModel):
    """User-editable incident values for a partial update."""

    name: IncidentName | None = None
    description: IncidentDescription | None = None
    affected_service: AffectedService | None = None
    reported_start_time: AwareDatetime | None = None

    model_config = ConfigDict(extra="forbid")

    @field_validator(
        "name",
        "description",
        "affected_service",
        mode="before",
    )
    @classmethod
    def reject_null_for_required_columns(cls, value: object) -> object:
        """Reject explicit null while allowing an omitted partial-update field."""
        if value is None:
            raise ValueError("field cannot be null")
        return value


class IncidentRead(BaseModel):
    """Public incident representation loaded from persistence."""

    id: int = Field(gt=0)
    public_id: IncidentPublicId
    name: IncidentName
    description: IncidentDescription
    affected_service: AffectedService
    reported_start_time: AwareDatetime | None
    status: IncidentStatus
    created_at: AwareDatetime
    updated_at: AwareDatetime

    model_config = ConfigDict(from_attributes=True, extra="forbid")
