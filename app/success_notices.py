"""Allowlisted success notices for Post/Redirect/Get web flows."""

from enum import StrEnum
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


SUCCESS_NOTICE_QUERY_PARAMETER = "notice"


class SuccessNotice(StrEnum):
    """Identifiers accepted by the shared success-toast renderer."""

    INCIDENT_CREATED = "incident-created"
    INCIDENT_UPDATED = "incident-updated"
    PASTED_EVIDENCE_CREATED = "pasted-evidence-created"
    EVIDENCE_FILES_UPLOADED = "evidence-files-uploaded"
    EVIDENCE_TYPE_UPDATED = "evidence-type-updated"
    ANALYSIS_REVIEW_UPDATED = "analysis-review-updated"


_SUCCESS_NOTICE_MESSAGES = {
    SuccessNotice.INCIDENT_CREATED: "Incident created successfully.",
    SuccessNotice.INCIDENT_UPDATED: "Incident updated successfully.",
    SuccessNotice.PASTED_EVIDENCE_CREATED: "Pasted evidence saved successfully.",
    SuccessNotice.EVIDENCE_FILES_UPLOADED: "Evidence files uploaded successfully.",
    SuccessNotice.EVIDENCE_TYPE_UPDATED: "Evidence type updated successfully.",
    SuccessNotice.ANALYSIS_REVIEW_UPDATED: "Human review saved successfully.",
}


def success_notice_message(identifier: str | None) -> str | None:
    """Return fixed display text only for an allowlisted notice identifier."""
    if identifier is None:
        return None
    try:
        notice = SuccessNotice(identifier)
    except ValueError:
        return None
    return _SUCCESS_NOTICE_MESSAGES[notice]


def add_success_notice(url: str, notice: SuccessNotice) -> str:
    """Add an allowlisted notice to an internal redirect URL."""
    split_url = urlsplit(url)
    query_parameters = [
        (name, value)
        for name, value in parse_qsl(split_url.query, keep_blank_values=True)
        if name != SUCCESS_NOTICE_QUERY_PARAMETER
    ]
    query_parameters.append((SUCCESS_NOTICE_QUERY_PARAMETER, notice.value))
    return urlunsplit(split_url._replace(query=urlencode(query_parameters)))
