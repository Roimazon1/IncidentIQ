"""Focused tests for deterministic evidence normalization."""

import csv
import json
from io import StringIO

import pytest

from app.services.preprocessing_service import (
    PreprocessingService,
    StructuredTextError,
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
