"""Pydantic contracts for evidence creation, updates, and reads."""

from typing import Annotated, Literal

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
from app.schemas.incident import IncidentPublicId


def _require_non_blank_text(value: str) -> str:
    if not value.strip():
        raise ValueError("evidence text must not be blank")
    return value


def _validate_line_range(value: str) -> str:
    start_text, separator, end_text = value.partition("-")
    start_line = int(start_text)
    end_line = int(end_text) if separator else start_line
    if start_line < 1 or end_line < start_line:
        raise ValueError("line range must contain positive ascending lines")
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
LineRange = Annotated[
    str,
    StringConstraints(pattern=r"^\d+(?:-\d+)?$"),
    AfterValidator(_validate_line_range),
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


class EvidenceManifestSource(EvidenceCreate):
    """Local evidence input accepted by deterministic manifest construction."""

    evidence_code: EvidenceCode


class EvidenceManifestTimestamp(BaseModel):
    """Traceable deterministic timestamp metadata for one evidence source."""

    raw_text: str | None
    value: AwareDatetime | None
    line_number: int | None = Field(default=None, gt=0)
    column_number: int | None = Field(default=None, gt=0)
    status: Literal["detected", "unknown", "conflicting"]
    reason: str | None = None

    model_config = ConfigDict(extra="forbid", frozen=True)


class EvidenceManifestChunk(BaseModel):
    """A bounded redacted evidence payload with stable line coordinates."""

    sequence: int = Field(gt=0)
    line_range: LineRange
    content: str = Field(min_length=1)

    model_config = ConfigDict(extra="forbid", frozen=True)


class EvidenceManifestItem(BaseModel):
    """One evidence source and its redacted, traceable outbound chunks."""

    id: EvidenceCode
    type: EvidenceType
    source: str = Field(min_length=1)
    line_range: LineRange
    timestamps: tuple[EvidenceManifestTimestamp, ...] = Field(min_length=1)
    chunks: tuple[EvidenceManifestChunk, ...] = Field(min_length=1)

    model_config = ConfigDict(extra="forbid", frozen=True)


class EvidenceManifest(BaseModel):
    """Reusable redacted evidence input shared by every AI prompt stage."""

    incident_id: IncidentPublicId
    evidence: tuple[EvidenceManifestItem, ...] = Field(min_length=1)

    model_config = ConfigDict(extra="forbid", frozen=True)


class RedactionPreviewFinding(BaseModel):
    """Safe redaction metadata rendered without the matched value."""

    location: Literal["source_name", "content"]
    category: Literal[
        "api_key",
        "bearer_token",
        "password",
        "email",
        "ip_address",
        "authorization_header",
        "credit_card",
    ]
    line_number: int = Field(gt=0)
    column_number: int = Field(gt=0)
    replacement: str = Field(min_length=1)

    model_config = ConfigDict(extra="forbid", frozen=True)


class EvidenceRedactionPreview(BaseModel):
    """Outbound-safe preview for one locally stored evidence item."""

    evidence_code: EvidenceCode
    evidence_type: EvidenceType
    source_name: str = Field(min_length=1)
    redacted_content: str = Field(min_length=1)
    findings: tuple[RedactionPreviewFinding, ...]

    model_config = ConfigDict(extra="forbid", frozen=True)

    @property
    def redaction_count(self) -> int:
        """Return the number of masked source and content values."""
        return len(self.findings)
