"""Pydantic contracts for evidence creation, updates, and reads."""

from typing import Annotated

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
)

from app.models.enums import EvidenceType
from app.models.evidence import (
    EVIDENCE_CODE_LENGTH,
    SHA256_HEX_LENGTH,
    SOURCE_NAME_MAX_LENGTH,
)


def _require_non_blank_text(value: str) -> str:
    if not value.strip():
        raise ValueError("evidence text must not be blank")
    return value


SourceName = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=SOURCE_NAME_MAX_LENGTH,
    ),
]
OriginalEvidenceText = Annotated[str, AfterValidator(_require_non_blank_text)]
EvidenceCode = Annotated[
    str,
    StringConstraints(
        min_length=EVIDENCE_CODE_LENGTH,
        max_length=EVIDENCE_CODE_LENGTH,
        pattern=r"^E-\d{3}$",
    ),
]
Sha256Checksum = Annotated[
    str,
    StringConstraints(
        min_length=SHA256_HEX_LENGTH,
        max_length=SHA256_HEX_LENGTH,
        pattern=r"^[0-9a-f]{64}$",
    ),
]


class EvidenceCreate(BaseModel):
    """Unmodified evidence content and user-selected source metadata."""

    source_name: SourceName
    evidence_type: EvidenceType
    original_text: OriginalEvidenceText

    model_config = ConfigDict(extra="forbid")


class EvidenceUpdate(BaseModel):
    """User-correctable evidence metadata for a partial update."""

    source_name: SourceName | None = None
    evidence_type: EvidenceType | None = None

    model_config = ConfigDict(extra="forbid")

    @field_validator("source_name", "evidence_type", mode="before")
    @classmethod
    def reject_null_for_required_columns(cls, value: object) -> object:
        """Reject explicit null while allowing an omitted partial-update field."""
        if value is None:
            raise ValueError("field cannot be null")
        return value


class EvidenceRead(BaseModel):
    """Traceable evidence representation loaded from persistence."""

    id: int = Field(gt=0)
    incident_id: int = Field(gt=0)
    evidence_code: EvidenceCode
    source_name: SourceName
    evidence_type: EvidenceType
    original_text: str
    redacted_text: str | None
    checksum: Sha256Checksum
    detected_start_time: AwareDatetime | None
    detected_end_time: AwareDatetime | None
    created_at: AwareDatetime
    updated_at: AwareDatetime

    model_config = ConfigDict(from_attributes=True, extra="forbid")
