"""Deterministic normalization for incident evidence text."""

import csv
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from io import StringIO
from pathlib import PurePath
from typing import Any


_ISO_TIMESTAMP_PATTERN = re.compile(
    r"(?<!\d)(?P<timestamp>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}"
    r"(?::\d{2}(?:\.\d{1,6})?)?(?:Z|[+-]\d{2}:\d{2})?)(?!\d)"
)
_TIMEZONE_SUFFIX_PATTERN = re.compile(r"(?:Z|[+-]\d{2}:\d{2})$")
_EXPLICIT_UNKNOWN_TIME_PATTERN = re.compile(
    r"\b(?:timestamp|time)\s*[:=]\s*(?:unknown|unavailable|n/?a)\b",
    re.IGNORECASE,
)
_EXPLICIT_TIME_CONFLICT_PATTERN = re.compile(
    r"\b(?:conflicting\s+timestamps?|timestamp\s+conflict)\b",
    re.IGNORECASE,
)


class StructuredTextError(ValueError):
    """Raised when declared structured evidence cannot be parsed safely."""


class TimestampStatus(StrEnum):
    """Deterministic interpretation state for a timestamp occurrence."""

    DETECTED = "detected"
    UNKNOWN = "unknown"
    CONFLICTING = "conflicting"


@dataclass(frozen=True, slots=True)
class ExtractedTimestamp:
    """A timestamp occurrence with source coordinates and uncertainty."""

    raw_text: str | None
    value: datetime | None
    line_number: int | None
    column_number: int | None
    status: TimestampStatus
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class SourceRange:
    """Inclusive source-line coordinates for normalized evidence text."""

    start_line: int
    end_line: int

    def __post_init__(self) -> None:
        if self.start_line < 1:
            raise ValueError("start_line must be positive")
        if self.end_line < self.start_line:
            raise ValueError("end_line must not precede start_line")

    @property
    def label(self) -> str:
        """Return a concise human-readable inclusive range."""
        if self.start_line == self.end_line:
            return str(self.start_line)
        return f"{self.start_line}-{self.end_line}"


def _reject_duplicate_json_keys(
    pairs: Iterable[tuple[str, Any]],
) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for key, value in pairs:
        if key in parsed:
            raise StructuredTextError(f'duplicate JSON key "{key}"')
        parsed[key] = value
    return parsed


def _reject_nonstandard_json_constant(value: str) -> None:
    raise StructuredTextError(f"non-standard JSON constant {value}")


class PreprocessingService:
    """Normalize evidence deterministically without mutating original content."""

    _LINE_NUMBER_WIDTH = 4

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize line endings and non-semantic surrounding whitespace."""
        normalized_endings = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = [line.rstrip(" \t") for line in normalized_endings.split("\n")]

        first_content_index = 0
        while first_content_index < len(lines) and not lines[first_content_index]:
            first_content_index += 1

        last_content_index = len(lines)
        while (
            last_content_index > first_content_index
            and not lines[last_content_index - 1]
        ):
            last_content_index -= 1
        return "\n".join(lines[first_content_index:last_content_index])

    @classmethod
    def normalize_by_source(cls, text: str, source_name: str) -> str:
        """Normalize declared JSON/CSV evidence or plain text by extension."""
        extension = PurePath(source_name).suffix.lower()
        normalized_endings = text.replace("\r\n", "\n").replace("\r", "\n")

        if extension == ".json":
            return cls._normalize_json(normalized_endings, source_name)
        if extension == ".csv":
            return cls._normalize_csv(normalized_endings, source_name)
        return cls.normalize_text(text)

    @classmethod
    def add_line_numbers(cls, text: str, *, start_line: int = 1) -> str:
        """Normalize text and prefix each source line with a stable label."""
        lines = cls._normalized_lines(text, start_line=start_line)
        numbered_lines = []
        for offset, line in enumerate(lines):
            line_number = start_line + offset
            prefix = f"L{line_number:0{cls._LINE_NUMBER_WIDTH}d}:"
            numbered_lines.append(f"{prefix} {line}" if line else prefix)
        return "\n".join(numbered_lines)

    @classmethod
    def get_source_range(
        cls,
        text: str,
        *,
        start_line: int = 1,
    ) -> SourceRange | None:
        """Return the inclusive source range for normalized non-empty text."""
        lines = cls._normalized_lines(text, start_line=start_line)
        if not lines:
            return None
        return SourceRange(
            start_line=start_line,
            end_line=start_line + len(lines) - 1,
        )

    @classmethod
    def extract_timestamps(
        cls,
        text: str,
        *,
        start_line: int = 1,
    ) -> list[ExtractedTimestamp]:
        """Extract direct ISO timestamps without guessing missing timezones."""
        lines = cls._normalized_lines(text, start_line=start_line)
        extracted: list[ExtractedTimestamp] = []

        for line_number, line in enumerate(lines, start=start_line):
            unknown_matches = list(_EXPLICIT_UNKNOWN_TIME_PATTERN.finditer(line))
            timestamp_matches = list(_ISO_TIMESTAMP_PATTERN.finditer(line))
            line_records = [
                ExtractedTimestamp(
                    raw_text=match.group(0),
                    value=None,
                    line_number=line_number,
                    column_number=match.start() + 1,
                    status=TimestampStatus.UNKNOWN,
                    reason="source explicitly records the time as unknown",
                )
                for match in unknown_matches
            ]
            line_records.extend(
                cls._parse_timestamp_match(match, line_number=line_number)
                for match in timestamp_matches
            )

            is_conflicting = bool(timestamp_matches) and (
                bool(unknown_matches)
                or _EXPLICIT_TIME_CONFLICT_PATTERN.search(line) is not None
            )
            if is_conflicting:
                line_records = [
                    ExtractedTimestamp(
                        raw_text=record.raw_text,
                        value=record.value,
                        line_number=record.line_number,
                        column_number=record.column_number,
                        status=TimestampStatus.CONFLICTING,
                        reason="source explicitly contains conflicting time information",
                    )
                    for record in line_records
                ]

            extracted.extend(
                sorted(
                    line_records,
                    key=lambda record: record.column_number or 0,
                )
            )

        if extracted:
            return extracted
        return [
            ExtractedTimestamp(
                raw_text=None,
                value=None,
                line_number=None,
                column_number=None,
                status=TimestampStatus.UNKNOWN,
                reason="no direct timestamp found",
            )
        ]

    @classmethod
    def _normalized_lines(cls, text: str, *, start_line: int) -> list[str]:
        if start_line < 1:
            raise ValueError("start_line must be positive")
        normalized = cls.normalize_text(text)
        return normalized.split("\n") if normalized else []

    @staticmethod
    def _parse_timestamp_match(
        match: re.Match[str],
        *,
        line_number: int,
    ) -> ExtractedTimestamp:
        raw_text = match.group("timestamp")
        location = {
            "raw_text": raw_text,
            "line_number": line_number,
            "column_number": match.start("timestamp") + 1,
        }
        if _TIMEZONE_SUFFIX_PATTERN.search(raw_text) is None:
            return ExtractedTimestamp(
                **location,
                value=None,
                status=TimestampStatus.UNKNOWN,
                reason="timestamp has no explicit timezone",
            )

        iso_value = raw_text[:-1] + "+00:00" if raw_text.endswith("Z") else raw_text
        try:
            parsed = datetime.fromisoformat(iso_value)
        except ValueError:
            return ExtractedTimestamp(
                **location,
                value=None,
                status=TimestampStatus.UNKNOWN,
                reason="timestamp is not a valid ISO-8601 date-time",
            )
        return ExtractedTimestamp(
            **location,
            value=parsed.astimezone(UTC),
            status=TimestampStatus.DETECTED,
        )

    @staticmethod
    def _normalize_json(text: str, source_name: str) -> str:
        try:
            parsed = json.loads(
                text.removeprefix("\ufeff"),
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=_reject_nonstandard_json_constant,
            )
        except (json.JSONDecodeError, StructuredTextError) as exc:
            raise StructuredTextError(
                f"{source_name} contains invalid JSON: {exc}"
            ) from exc
        return json.dumps(
            parsed,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )

    @staticmethod
    def _normalize_csv(text: str, source_name: str) -> str:
        try:
            rows = list(
                csv.reader(
                    StringIO(text.removeprefix("\ufeff"), newline=""),
                    strict=True,
                )
            )
        except csv.Error as exc:
            raise StructuredTextError(
                f"{source_name} contains invalid CSV: {exc}"
            ) from exc

        output = StringIO(newline="")
        csv.writer(output, lineterminator="\n").writerows(rows)
        return output.getvalue().removesuffix("\n")
