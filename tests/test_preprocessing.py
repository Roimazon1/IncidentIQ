"""Focused tests for deterministic evidence normalization."""

import csv
import json
from datetime import UTC, datetime
from io import StringIO

import pytest

from app.services.preprocessing_service import (
    ExtractedTimestamp,
    PreprocessingService,
    SourceRange,
    StructuredTextError,
    TimestampStatus,
)


def test_normalize_text_preserves_content_while_normalizing_whitespace() -> None:
    original = (
        "\r\n  ERROR checkout failed  \t\r\n \t\r\n"
        "\tstack frame\t \rnext  value  \n \t\n"
    )

    normalized = PreprocessingService.normalize_text(original)

    assert normalized == "  ERROR checkout failed\n\n\tstack frame\nnext  value"
    assert original.startswith("\r\n")


def test_normalize_text_is_idempotent() -> None:
    normalized = PreprocessingService.normalize_text("first\r\nsecond  \r\n")

    assert PreprocessingService.normalize_text(normalized) == normalized


def test_json_is_parsed_and_rendered_deterministically() -> None:
    original = (
        '\ufeff {"message":"checkout  failed","details":{"retry":false},'
        '"items":[2,1],"owner":"צוות"}\r\n'
    )

    normalized = PreprocessingService.normalize_by_source(
        original,
        "incident.JSON",
    )

    assert json.loads(normalized) == {
        "message": "checkout  failed",
        "details": {"retry": False},
        "items": [2, 1],
        "owner": "צוות",
    }
    assert "\r" not in normalized
    assert "צוות" in normalized
    assert original.endswith("\r\n")
    assert (
        PreprocessingService.normalize_by_source(normalized, "incident.json")
        == normalized
    )


def test_invalid_json_is_rejected() -> None:
    with pytest.raises(StructuredTextError, match="contains invalid JSON"):
        PreprocessingService.normalize_by_source('{"status":', "incident.json")


def test_duplicate_json_keys_are_rejected() -> None:
    with pytest.raises(StructuredTextError, match="duplicate JSON key"):
        PreprocessingService.normalize_by_source(
            '{"status":"ready","status":"failed"}',
            "incident.json",
        )


def test_nonstandard_json_constants_are_rejected() -> None:
    with pytest.raises(StructuredTextError, match="non-standard JSON constant"):
        PreprocessingService.normalize_by_source(
            '{"duration":NaN}',
            "incident.json",
        )


def test_csv_is_rendered_with_lf_without_changing_cell_values() -> None:
    original = (
        '\ufeffsource,message\r\napp,"checkout, failed"\r\n'
        'notes,"  keep spaces  "\r\ntrace,"line one\r\nline two"\r\n'
    )

    normalized = PreprocessingService.normalize_by_source(
        original,
        "monitoring.CSV",
    )

    assert list(csv.reader(StringIO(normalized, newline=""))) == [
        ["source", "message"],
        ["app", "checkout, failed"],
        ["notes", "  keep spaces  "],
        ["trace", "line one\nline two"],
    ]
    assert "\r" not in normalized
    assert original.endswith("\r\n")
    assert (
        PreprocessingService.normalize_by_source(normalized, "monitoring.csv")
        == normalized
    )


def test_malformed_csv_is_rejected() -> None:
    with pytest.raises(StructuredTextError, match="contains invalid CSV"):
        PreprocessingService.normalize_by_source(
            'source,message\napp,"unterminated',
            "monitoring.csv",
        )


def test_non_structured_extension_uses_plain_text_normalization() -> None:
    normalized = PreprocessingService.normalize_by_source(
        "\r\nERROR checkout failed  \r\n",
        "checkout.LOG",
    )

    assert normalized == "ERROR checkout failed"


def test_line_numbers_are_stable_and_preserve_internal_blank_lines() -> None:
    original = "\r\nfirst line  \r\n\r\n\tthird line\t\r\n"

    numbered = PreprocessingService.add_line_numbers(original)

    assert numbered == "L0001: first line\nL0002:\nL0003: \tthird line"
    assert original.startswith("\r\n")


def test_line_numbering_is_deterministic() -> None:
    text = "first\nsecond"

    first_result = PreprocessingService.add_line_numbers(text)

    assert PreprocessingService.add_line_numbers(text) == first_result


def test_custom_start_line_coordinates_numbering_and_source_range() -> None:
    text = "first\n\nthird"

    numbered = PreprocessingService.add_line_numbers(text, start_line=7)
    source_range = PreprocessingService.get_source_range(text, start_line=7)

    assert numbered == "L0007: first\nL0008:\nL0009: third"
    assert source_range == SourceRange(start_line=7, end_line=9)
    assert source_range.label == "7-9"


def test_single_line_source_range_has_concise_label() -> None:
    source_range = PreprocessingService.get_source_range(
        "one line",
        start_line=4,
    )

    assert source_range == SourceRange(start_line=4, end_line=4)
    assert source_range.label == "4"


def test_empty_text_has_no_numbered_lines_or_source_range() -> None:
    assert PreprocessingService.add_line_numbers(" \t\r\n") == ""
    assert PreprocessingService.get_source_range(" \t\r\n") is None


@pytest.mark.parametrize("start_line", [0, -1])
def test_non_positive_start_line_is_rejected(start_line: int) -> None:
    with pytest.raises(ValueError, match="start_line must be positive"):
        PreprocessingService.add_line_numbers("evidence", start_line=start_line)

    with pytest.raises(ValueError, match="start_line must be positive"):
        PreprocessingService.get_source_range("evidence", start_line=start_line)


def test_aware_iso_timestamps_are_normalized_to_utc_with_source_coordinates() -> None:
    text = (
        "\r\ndeployed 2025-02-14T12:15:30+02:00  \r\n"
        "failed 2025-02-14 10:16:45.250Z\r\n"
    )

    extracted = PreprocessingService.extract_timestamps(text)

    assert extracted == [
        ExtractedTimestamp(
            raw_text="2025-02-14T12:15:30+02:00",
            value=datetime(2025, 2, 14, 10, 15, 30, tzinfo=UTC),
            line_number=1,
            column_number=10,
            status=TimestampStatus.DETECTED,
        ),
        ExtractedTimestamp(
            raw_text="2025-02-14 10:16:45.250Z",
            value=datetime(2025, 2, 14, 10, 16, 45, 250000, tzinfo=UTC),
            line_number=2,
            column_number=8,
            status=TimestampStatus.DETECTED,
        ),
    ]


def test_timestamp_coordinates_respect_validated_start_line() -> None:
    extracted = PreprocessingService.extract_timestamps(
        "first 2025-02-14T10:15:30Z\nsecond 2025-02-14T10:16:30Z",
        start_line=7,
    )

    assert [record.line_number for record in extracted] == [7, 8]


@pytest.mark.parametrize("start_line", [0, -1])
def test_timestamp_extraction_rejects_non_positive_start_line(
    start_line: int,
) -> None:
    with pytest.raises(ValueError, match="start_line must be positive"):
        PreprocessingService.extract_timestamps(
            "2025-02-14T10:15:30Z",
            start_line=start_line,
        )


def test_timestamp_without_timezone_is_recorded_as_unknown() -> None:
    extracted = PreprocessingService.extract_timestamps(
        "event at 2025-02-14T10:15:30",
    )

    assert extracted == [
        ExtractedTimestamp(
            raw_text="2025-02-14T10:15:30",
            value=None,
            line_number=1,
            column_number=10,
            status=TimestampStatus.UNKNOWN,
            reason="timestamp has no explicit timezone",
        )
    ]


def test_invalid_iso_timestamp_is_recorded_as_unknown() -> None:
    extracted = PreprocessingService.extract_timestamps(
        "event at 2025-02-30T10:15:30Z",
    )

    assert extracted[0].raw_text == "2025-02-30T10:15:30Z"
    assert extracted[0].value is None
    assert extracted[0].status is TimestampStatus.UNKNOWN
    assert extracted[0].reason == "timestamp is not a valid ISO-8601 date-time"


def test_missing_timestamp_returns_explicit_unknown_record() -> None:
    extracted = PreprocessingService.extract_timestamps("checkout failed")

    assert extracted == [
        ExtractedTimestamp(
            raw_text=None,
            value=None,
            line_number=None,
            column_number=None,
            status=TimestampStatus.UNKNOWN,
            reason="no direct timestamp found",
        )
    ]


def test_explicit_unknown_time_retains_its_source_location() -> None:
    extracted = PreprocessingService.extract_timestamps(
        "checkout failed\ntimestamp = unknown",
    )

    assert extracted == [
        ExtractedTimestamp(
            raw_text="timestamp = unknown",
            value=None,
            line_number=2,
            column_number=1,
            status=TimestampStatus.UNKNOWN,
            reason="source explicitly records the time as unknown",
        )
    ]


def test_explicitly_conflicting_timestamps_remain_visible() -> None:
    extracted = PreprocessingService.extract_timestamps(
        "timestamp conflict: 2025-02-14T10:15:30Z versus 2025-02-14T10:17:30Z",
    )

    assert [record.raw_text for record in extracted] == [
        "2025-02-14T10:15:30Z",
        "2025-02-14T10:17:30Z",
    ]
    assert [record.value for record in extracted] == [
        datetime(2025, 2, 14, 10, 15, 30, tzinfo=UTC),
        datetime(2025, 2, 14, 10, 17, 30, tzinfo=UTC),
    ]
    assert all(record.status is TimestampStatus.CONFLICTING for record in extracted)
    assert all(record.line_number == 1 for record in extracted)


def test_multiple_timestamps_are_not_implicitly_marked_conflicting() -> None:
    extracted = PreprocessingService.extract_timestamps(
        "window 2025-02-14T10:15:30Z to 2025-02-14T10:17:30Z",
    )

    assert len(extracted) == 2
    assert all(record.status is TimestampStatus.DETECTED for record in extracted)
