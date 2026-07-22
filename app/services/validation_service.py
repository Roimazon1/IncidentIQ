"""Deterministic evidence-reference validation for structured AI output."""

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel

from app.models import ClaimSupportStatus
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
    INVALID_LINE_RANGE = "invalid_line_range"
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
        """Return an identifier and line-range outcome for every reference."""
        outcomes: list[EvidenceReferenceValidationOutcome] = []
        for reference in references:
            _, invalid_outcome = cls._validate_reference_location(
                reference,
                evidence_manifest,
            )
            outcomes.append(
                invalid_outcome
                if invalid_outcome is not None
                else cls._outcome(reference, EvidenceReferenceValidationStatus.VALID)
            )
        return tuple(outcomes)

    @classmethod
    def validate_supporting_excerpt(
        cls,
        reference: EvidenceReferenceV1,
        evidence_manifest: EvidenceManifest,
    ) -> EvidenceReferenceValidationOutcome:
        """Return an exact normalized-redacted excerpt validation outcome."""

        evidence_item, invalid_outcome = cls._validate_reference_location(
            reference,
            evidence_manifest,
        )
        if invalid_outcome is not None:
            return invalid_outcome
        assert evidence_item is not None
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
    def classify_claim_support(
        outcomes: Iterable[EvidenceReferenceValidationOutcome],
        *,
        is_inferred: bool = False,
        has_valid_contradicting_evidence: bool = False,
    ) -> ClaimSupportStatus:
        """Classify a claim from deterministic reference-validation outcomes."""
        outcome_set = tuple(outcomes)
        valid_count = sum(outcome.is_valid for outcome in outcome_set)
        if valid_count == 0:
            return ClaimSupportStatus.UNSUPPORTED
        if has_valid_contradicting_evidence:
            return ClaimSupportStatus.CONTRADICTED
        if is_inferred:
            return ClaimSupportStatus.INFERRED
        if valid_count < len(outcome_set):
            return ClaimSupportStatus.PARTIALLY_SUPPORTED
        return ClaimSupportStatus.SUPPORTED

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
            EvidenceReferenceValidationStatus.INVALID_LINE_RANGE: (
                "Line range is outside the referenced evidence."
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
    def _validate_reference_location(
        cls,
        reference: EvidenceReferenceV1,
        evidence_manifest: EvidenceManifest,
    ) -> tuple[
        EvidenceManifestItem | None,
        EvidenceReferenceValidationOutcome | None,
    ]:
        evidence_item = next(
            (
                item
                for item in evidence_manifest.evidence
                if item.id == reference.evidence_id
            ),
            None,
        )
        if evidence_item is None:
            return None, cls._outcome(
                reference,
                EvidenceReferenceValidationStatus.UNKNOWN_EVIDENCE_ID,
            )
        if not cls._line_range_exists(evidence_item, reference.line_range):
            return evidence_item, cls._outcome(
                reference,
                EvidenceReferenceValidationStatus.INVALID_LINE_RANGE,
            )
        return evidence_item, None

    @staticmethod
    def _line_range_exists(
        evidence_item: EvidenceManifestItem,
        line_range: str,
    ) -> bool:
        start_text, separator, end_text = line_range.partition("-")
        start_line = int(start_text)
        end_line = int(end_text) if separator else start_line
        existing_line_numbers = {
            int(match.group("number"))
            for chunk in evidence_item.chunks
            for numbered_line in chunk.content.splitlines()
            if (match := _NUMBERED_LINE_PATTERN.fullmatch(numbered_line)) is not None
        }
        requested_line_count = end_line - start_line + 1
        matching_line_count = sum(
            start_line <= line_number <= end_line
            for line_number in existing_line_numbers
        )
        return matching_line_count == requested_line_count

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
