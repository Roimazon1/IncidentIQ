"""Construction of reusable, redacted evidence manifests for AI stages."""

from collections.abc import Iterable

from app.schemas.evidence import (
    EvidenceManifest,
    EvidenceManifestChunk,
    EvidenceManifestItem,
    EvidenceManifestSource,
    EvidenceManifestTimestamp,
    EvidenceRedactionPreview,
    RedactionPreviewFinding,
)
from app.services.preprocessing_service import (
    DEFAULT_CHUNK_MAX_CHARACTERS,
    DEFAULT_CHUNK_MAX_LINES,
    PreprocessingService,
    StructuredTextError,
)
from app.services.redaction_service import RedactionResult, RedactionService


class EvidencePreviewValidationError(ValueError):
    """Raised when saved structured evidence cannot produce a safe preview."""


class EvidenceManifestService:
    """Build validated outbound manifests without exposing local originals."""

    @classmethod
    def build_evidence_manifest(
        cls,
        incident_id: str,
        evidence_sources: Iterable[EvidenceManifestSource],
        *,
        max_chunk_characters: int = DEFAULT_CHUNK_MAX_CHARACTERS,
        max_chunk_lines: int = DEFAULT_CHUNK_MAX_LINES,
    ) -> EvidenceManifest:
        """Build one deterministic manifest for reuse by every AI stage."""
        sources = sorted(
            evidence_sources,
            key=lambda source: source.evidence_code,
        )
        if not sources:
            raise ValueError("at least one evidence source is required")

        evidence_ids = [source.evidence_code for source in sources]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("evidence IDs must be unique within a manifest")

        manifest_items = tuple(
            cls._build_manifest_item(
                source,
                max_chunk_characters=max_chunk_characters,
                max_chunk_lines=max_chunk_lines,
            )
            for source in sources
        )
        return EvidenceManifest(
            incident_id=incident_id,
            evidence=manifest_items,
        )

    @classmethod
    def build_redaction_preview(
        cls,
        source: EvidenceManifestSource,
    ) -> EvidenceRedactionPreview:
        """Build a safe preview matching the manifest redaction pipeline."""
        try:
            _, source_redaction, content_redaction = cls._prepare_redactions(source)
        except StructuredTextError as exc:
            raise EvidencePreviewValidationError(
                "The saved structured evidence is malformed and cannot be prepared "
                "for redaction preview. Correct or replace it before external AI use."
            ) from exc
        findings = tuple(
            RedactionPreviewFinding(
                location=location,
                category=detection.category.value,
                line_number=detection.line_number,
                column_number=detection.column_number,
                replacement=detection.replacement,
            )
            for location, result in (
                ("source_name", source_redaction),
                ("content", content_redaction),
            )
            for detection in result.detections
        )
        return EvidenceRedactionPreview(
            evidence_code=source.evidence_code,
            evidence_type=source.evidence_type,
            source_name=source_redaction.redacted_text,
            redacted_content=PreprocessingService.add_line_numbers(
                content_redaction.redacted_text
            ),
            findings=findings,
        )

    @staticmethod
    def _build_manifest_item(
        source: EvidenceManifestSource,
        *,
        max_chunk_characters: int,
        max_chunk_lines: int,
    ) -> EvidenceManifestItem:
        normalized, source_redaction, content_redaction = (
            EvidenceManifestService._prepare_redactions(source)
        )
        source_range = PreprocessingService.get_source_range(normalized)
        if source_range is None:
            raise ValueError("normalized evidence must not be empty")

        timestamps = tuple(
            EvidenceManifestTimestamp(
                raw_text=timestamp.raw_text,
                value=timestamp.value,
                line_number=timestamp.line_number,
                column_number=timestamp.column_number,
                status=timestamp.status.value,
                reason=timestamp.reason,
            )
            for timestamp in PreprocessingService.extract_timestamps(normalized)
        )
        numbered_redacted_text = PreprocessingService.add_line_numbers(
            content_redaction.redacted_text
        )
        chunks = PreprocessingService.split_into_chunks(
            numbered_redacted_text,
            evidence_id=source.evidence_code,
            max_characters=max_chunk_characters,
            max_lines=max_chunk_lines,
        )
        return EvidenceManifestItem(
            id=source.evidence_code,
            type=source.evidence_type,
            source=source_redaction.redacted_text,
            line_range=source_range.label,
            timestamps=timestamps,
            chunks=tuple(
                EvidenceManifestChunk(
                    sequence=chunk.sequence,
                    line_range=chunk.source_range.label,
                    content=chunk.text,
                )
                for chunk in chunks
            ),
        )

    @staticmethod
    def _prepare_redactions(
        source: EvidenceManifestSource,
    ) -> tuple[str, RedactionResult, RedactionResult]:
        normalized = PreprocessingService.normalize_by_source(
            source.original_text,
            source.source_name,
        )
        return (
            normalized,
            RedactionService.redact_text(source.source_name),
            RedactionService.redact_text(normalized),
        )
