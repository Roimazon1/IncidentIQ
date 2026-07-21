"""Deterministic public identifier formatting for persisted domain records."""

from app.models.evidence import EVIDENCE_CODE_LENGTH
from app.models.incident import INCIDENT_PUBLIC_ID_LENGTH


INCIDENT_PUBLIC_ID_PREFIX = "INC-"
EVIDENCE_CODE_PREFIX = "E-"
INCIDENT_PUBLIC_ID_DIGITS = INCIDENT_PUBLIC_ID_LENGTH - len(
    INCIDENT_PUBLIC_ID_PREFIX
)
EVIDENCE_CODE_DIGITS = EVIDENCE_CODE_LENGTH - len(EVIDENCE_CODE_PREFIX)
MAX_INCIDENT_SEQUENCE = (10**INCIDENT_PUBLIC_ID_DIGITS) - 1
MAX_EVIDENCE_SEQUENCE = (10**EVIDENCE_CODE_DIGITS) - 1


def _validate_sequence_number(
    sequence_number: int,
    *,
    maximum: int,
    identifier_name: str,
) -> None:
    if isinstance(sequence_number, bool) or not isinstance(sequence_number, int):
        raise TypeError(f"{identifier_name} sequence number must be an integer")
    if sequence_number < 1:
        raise ValueError(f"{identifier_name} sequence number must be positive")
    if sequence_number > maximum:
        raise ValueError(
            f"{identifier_name} sequence number cannot exceed {maximum}"
        )


def generate_incident_public_id(sequence_number: int) -> str:
    """Return the locked incident identifier for a one-based sequence number."""
    _validate_sequence_number(
        sequence_number,
        maximum=MAX_INCIDENT_SEQUENCE,
        identifier_name="incident",
    )
    return f"{INCIDENT_PUBLIC_ID_PREFIX}{sequence_number:0{INCIDENT_PUBLIC_ID_DIGITS}d}"


def generate_evidence_code(sequence_number: int) -> str:
    """Return the locked per-incident evidence code for a sequence number."""
    _validate_sequence_number(
        sequence_number,
        maximum=MAX_EVIDENCE_SEQUENCE,
        identifier_name="evidence",
    )
    return f"{EVIDENCE_CODE_PREFIX}{sequence_number:0{EVIDENCE_CODE_DIGITS}d}"
