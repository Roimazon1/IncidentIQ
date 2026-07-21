"""Enumerated values shared by IncidentIQ persistence models."""

from enum import StrEnum


class IncidentStatus(StrEnum):
    """Lifecycle states for an incident investigation."""

    DRAFT = "DRAFT"
    READY = "READY"
    ANALYZING = "ANALYZING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class EvidenceType(StrEnum):
    """Supported classifications for an evidence item."""

    APPLICATION_LOG = "APPLICATION_LOG"
    ERROR_TRACE = "ERROR_TRACE"
    MONITORING_ALERT = "MONITORING_ALERT"
    DEPLOYMENT_NOTE = "DEPLOYMENT_NOTE"
    USER_COMPLAINT = "USER_COMPLAINT"
    API_RESPONSE = "API_RESPONSE"
    DATABASE_ERROR = "DATABASE_ERROR"
    OTHER = "OTHER"


class ClaimSupportStatus(StrEnum):
    """Deterministic support classifications for generated claims."""

    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    INFERRED = "INFERRED"
    CONTRADICTED = "CONTRADICTED"
    UNSUPPORTED = "UNSUPPORTED"


class AnalysisRunStatus(StrEnum):
    """Lifecycle states for one auditable analysis execution."""

    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class FactReviewStatus(StrEnum):
    """Human review decisions for an extracted fact."""

    PENDING = "PENDING"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    RECLASSIFIED_AS_ASSUMPTION = "RECLASSIFIED_AS_ASSUMPTION"


class HypothesisStatus(StrEnum):
    """Human review states for a generated hypothesis."""

    UNTESTED = "UNTESTED"
    SUPPORTED = "SUPPORTED"
    WEAKENED = "WEAKENED"
    REJECTED = "REJECTED"
    CONFIRMED_BY_HUMAN = "CONFIRMED_BY_HUMAN"
