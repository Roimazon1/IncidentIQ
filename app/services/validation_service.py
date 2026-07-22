"""Deterministic evidence-reference validation for structured AI output."""

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel

from app.schemas.ai_outputs import EvidenceReferenceV1
from app.schemas.evidence import (
    EvidenceCode,
    EvidenceManifest,
    EvidenceManifestItem,
    LineRange,
)
from app.services.preprocessing_service import PreprocessingService


_NUMBERED_LINE_PATTERN = re.compile(r"^L(?P<number>\d+):(?: (?P<content>.*))?$")


class EvidenceReferenceValidationStatus(StrEnum):
    """Deterministic citation checks, distinct from claim-support classification."""

    VALID = "valid"
    UNKNOWN_EVIDENCE_ID = "unknown_evidence_id"
    EXCERPT_MISMATCH = "excerpt_mismatch"


@dataclass(frozen=True, slots=True)
class EvidenceReferenceValidationOutcome:
    """Safe local validation result for one generated evidence reference."""

    evidence_id: EvidenceCode
    line_range: LineRange
    status: EvidenceReferenceValidationStatus
    message: str

    @property
    def is_valid(self) -> bool:
        """Return whether both identifier and optional excerpt were verified."""
        return self.status is EvidenceReferenceValidationStatus.VALID


class ValidationService:
    """Validate generated citations against one redacted evidence manifest."""

    @classmethod
    def validate_output_references(
        cls,
        output: BaseModel,
        evidence_manifest: EvidenceManifest,
    ) -> tuple[EvidenceReferenceValidationOutcome, ...]:
        """Return one ordered outcome for every nested evidence reference."""
        return tuple(
            cls.validate_supporting_excerpt(reference, evidence_manifest)
            for reference in cls._iter_evidence_references(output)
        )

    @classmethod
    def validate_evidence_ids(
        cls,
        references: Iterable[EvidenceReferenceV1],
        evidence_manifest: EvidenceManifest,
    ) -> tuple[EvidenceReferenceValidationOutcome, ...]:
        """Return an identifier outcome for every supplied reference."""
        valid_ids = {item.id for item in evidence_manifest.evidence}
        return tuple(
            cls._outcome(
                reference,
                (
                    EvidenceReferenceValidationStatus.VALID
                    if reference.evidence_id in valid_ids
                    else EvidenceReferenceValidationStatus.UNKNOWN_EVIDENCE_ID
                ),
            )
            for reference in references
        )

    @classmethod
    def validate_supporting_excerpt(
        cls,
        reference: EvidenceReferenceV1,
        evidence_manifest: EvidenceManifest,
    ) -> EvidenceReferenceValidationOutcome:
        """Return an exact normalized-redacted excerpt validation outcome."""

        evidence_item = next(
            (
                item
                for item in evidence_manifest.evidence
                if item.id == reference.evidence_id
            ),
            None,
        )
        if evidence_item is None:
            return cls._outcome(
                reference,
                EvidenceReferenceValidationStatus.UNKNOWN_EVIDENCE_ID,
            )
        if reference.excerpt is None:
            return cls._outcome(
                reference,
                EvidenceReferenceValidationStatus.VALID,
            )

        referenced_text = cls._referenced_normalized_text(
            evidence_item,
            reference.line_range,
        )
        normalized_excerpt = PreprocessingService.normalize_text(reference.excerpt)
        if not normalized_excerpt or normalized_excerpt not in referenced_text:
            return cls._outcome(
                reference,
                EvidenceReferenceValidationStatus.EXCERPT_MISMATCH,
            )
        return cls._outcome(reference, EvidenceReferenceValidationStatus.VALID)

    @staticmethod
    def _outcome(
        reference: EvidenceReferenceV1,
        status: EvidenceReferenceValidationStatus,
    ) -> EvidenceReferenceValidationOutcome:
        messages = {
            EvidenceReferenceValidationStatus.VALID: (
                "Evidence reference matches the redacted analysis manifest."
            ),
            EvidenceReferenceValidationStatus.UNKNOWN_EVIDENCE_ID: (
                "Evidence identifier is not present in the analysis manifest."
            ),
            EvidenceReferenceValidationStatus.EXCERPT_MISMATCH: (
                "Excerpt does not match the referenced normalized redacted evidence."
            ),
        }
        return EvidenceReferenceValidationOutcome(
            evidence_id=reference.evidence_id,
            line_range=reference.line_range,
            status=status,
            message=messages[status],
        )

    @classmethod
    def _iter_evidence_references(
        cls,
        value: object,
    ) -> Iterator[EvidenceReferenceV1]:
        if isinstance(value, EvidenceReferenceV1):
            yield value
            return
        if isinstance(value, BaseModel):
            for field_name in type(value).model_fields:
                yield from cls._iter_evidence_references(getattr(value, field_name))
            return
        if isinstance(value, (tuple, list)):
            for item in value:
                yield from cls._iter_evidence_references(item)

    @staticmethod
    def _referenced_normalized_text(
        evidence_item: EvidenceManifestItem,
        line_range: str,
    ) -> str:
        start_text, separator, end_text = line_range.partition("-")
        start_line = int(start_text)
        end_line = int(end_text) if separator else start_line
        numbered_text = "".join(
            chunk.content
            for chunk in sorted(evidence_item.chunks, key=lambda item: item.sequence)
        )
        referenced_lines: list[str] = []
        for numbered_line in numbered_text.splitlines():
            match = _NUMBERED_LINE_PATTERN.fullmatch(numbered_line)
            if match is None:
                continue
            line_number = int(match.group("number"))
            if start_line <= line_number <= end_line:
                referenced_lines.append(match.group("content") or "")
        return PreprocessingService.normalize_text("\n".join(referenced_lines))
