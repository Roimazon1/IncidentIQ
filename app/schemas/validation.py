"""Shared deterministic validation status contracts."""

from enum import StrEnum


class EvidenceReferenceValidationStatus(StrEnum):
    """Provider-neutral deterministic outcomes for generated citations."""

    VALID = "valid"
    UNKNOWN_EVIDENCE_ID = "unknown_evidence_id"
    INVALID_LINE_RANGE = "invalid_line_range"
    EXCERPT_MISMATCH = "excerpt_mismatch"
