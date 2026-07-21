"""Focused tests for secure deterministic evidence manifest construction."""

from datetime import UTC, datetime

import pytest

from app.models.enums import EvidenceType
from app.schemas.evidence import EvidenceManifestSource
from app.services.evidence_manifest_service import EvidenceManifestService


def test_manifest_contains_only_redacted_line_numbered_chunks() -> None:
    secret = "sk-production-secret-1234"
    source = EvidenceManifestSource(
        evidence_code="E-001",
        source_name="checkout.log",
        evidence_type=EvidenceType.APPLICATION_LOG,
        original_text=(
            f"2025-02-14T10:15:30Z checkout failed\napi_key={secret}\nretry=false"
        ),
    )
    original_text = source.original_text

    manifest = EvidenceManifestService.build_evidence_manifest(
        "INC-000001",
        [source],
        max_chunk_characters=1_000,
        max_chunk_lines=2,
    )

    item = manifest.evidence[0]
    assert item.id == "E-001"
    assert item.type is EvidenceType.APPLICATION_LOG
    assert item.source == "checkout.log"
    assert item.line_range == "1-3"
    assert item.timestamps[0].raw_text == "2025-02-14T10:15:30Z"
    assert item.timestamps[0].value == datetime(
        2025,
        2,
        14,
        10,
        15,
        30,
        tzinfo=UTC,
    )
    assert item.timestamps[0].status == "detected"
    assert [chunk.line_range for chunk in item.chunks] == ["1-2", "3"]
    assert "".join(chunk.content for chunk in item.chunks) == (
        "L0001: 2025-02-14T10:15:30Z checkout failed\n"
        "L0002: api_key=[REDACTED_API_KEY]\n"
        "L0003: retry=false"
    )
    serialized = manifest.model_dump_json()
    assert secret not in serialized
    assert "original_text" not in serialized
    assert source.original_text == original_text


def test_manifest_redacts_source_name_without_changing_local_metadata() -> None:
    source_name = "oncall@example.com"
    source = EvidenceManifestSource(
        evidence_code="E-004",
        source_name=source_name,
        evidence_type=EvidenceType.USER_COMPLAINT,
        original_text="checkout failed",
    )

    manifest = EvidenceManifestService.build_evidence_manifest(
        "INC-000001",
        [source],
    )

    assert manifest.evidence[0].source == "[REDACTED_EMAIL]"
    assert source_name not in manifest.model_dump_json()
    assert source.source_name == source_name


def test_manifest_orders_evidence_deterministically_by_code() -> None:
    sources = [
        EvidenceManifestSource(
            evidence_code=evidence_code,
            source_name=f"{evidence_code}.log",
            evidence_type=EvidenceType.APPLICATION_LOG,
            original_text=f"content for {evidence_code}",
        )
        for evidence_code in ("E-010", "E-002", "E-001")
    ]

    manifest = EvidenceManifestService.build_evidence_manifest(
        "INC-000001",
        sources,
    )

    assert [item.id for item in manifest.evidence] == [
        "E-001",
        "E-002",
        "E-010",
    ]


def test_manifest_records_unknown_timestamp_explicitly() -> None:
    source = EvidenceManifestSource(
        evidence_code="E-002",
        source_name="support.txt",
        evidence_type=EvidenceType.USER_COMPLAINT,
        original_text="checkout failed with no recorded time",
    )

    manifest = EvidenceManifestService.build_evidence_manifest(
        "INC-000001",
        [source],
    )

    timestamp = manifest.evidence[0].timestamps[0]
    assert timestamp.status == "unknown"
    assert timestamp.raw_text is None
    assert timestamp.value is None
    assert timestamp.reason == "no direct timestamp found"


def test_manifest_preserves_explicitly_conflicting_timestamps() -> None:
    source = EvidenceManifestSource(
        evidence_code="E-003",
        source_name="conflict.log",
        evidence_type=EvidenceType.APPLICATION_LOG,
        original_text=(
            "timestamp conflict: 2025-02-14T10:15:30Z versus 2025-02-14T10:17:30Z"
        ),
    )

    manifest = EvidenceManifestService.build_evidence_manifest(
        "INC-000001",
        [source],
    )

    assert [timestamp.status for timestamp in manifest.evidence[0].timestamps] == [
        "conflicting",
        "conflicting",
    ]


def test_manifest_rejects_duplicate_evidence_ids() -> None:
    sources = [
        EvidenceManifestSource(
            evidence_code="E-001",
            source_name=source_name,
            evidence_type=EvidenceType.OTHER,
            original_text=source_name,
        )
        for source_name in ("first.txt", "second.txt")
    ]

    with pytest.raises(ValueError, match="evidence IDs must be unique"):
        EvidenceManifestService.build_evidence_manifest(
            "INC-000001",
            sources,
        )


def test_manifest_requires_at_least_one_evidence_source() -> None:
    with pytest.raises(ValueError, match="at least one evidence source"):
        EvidenceManifestService.build_evidence_manifest("INC-000001", [])
