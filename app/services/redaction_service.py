"""Deterministic masking of sensitive values before external AI use."""

import ipaddress
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import StrEnum


class SensitiveValueType(StrEnum):
    """Supported categories of sensitive evidence content."""

    API_KEY = "api_key"
    BEARER_TOKEN = "bearer_token"
    PASSWORD = "password"
    EMAIL = "email"
    IP_ADDRESS = "ip_address"
    AUTHORIZATION_HEADER = "authorization_header"
    CREDIT_CARD = "credit_card"


_REPLACEMENTS = {
    category: f"[REDACTED_{category.value.upper()}]" for category in SensitiveValueType
}
_ASSIGNED_VALUE_PATTERN = r"(?P<secret>\"[^\"\r\n]+\"|'[^'\r\n]+'|[^\s,;&]+)"


def _is_ip_address(value: str) -> bool:
    try:
        ipaddress.ip_address(value)
    except ValueError:
        return False
    return True


@dataclass(frozen=True, slots=True)
class SensitiveValue:
    """Safe metadata describing a sensitive span without retaining its value."""

    category: SensitiveValueType
    start: int
    end: int
    line_number: int
    column_number: int
    replacement: str

    @property
    def preview(self) -> str:
        """Describe the masked location without exposing the original value."""
        return (
            f"{self.category.value} at line {self.line_number}, "
            f"column {self.column_number}: {self.replacement}"
        )


@dataclass(frozen=True, slots=True)
class RedactionResult:
    """Redacted outbound text and safe metadata for human preview."""

    redacted_text: str
    detections: tuple[SensitiveValue, ...]

    @property
    def redaction_count(self) -> int:
        """Return the number of independently masked spans."""
        return len(self.detections)


@dataclass(frozen=True, slots=True)
class _PatternSpec:
    category: SensitiveValueType
    pattern: re.Pattern[str]
    validator: Callable[[str], bool] | None = None


@dataclass(frozen=True, slots=True)
class _Candidate:
    category: SensitiveValueType
    start: int
    end: int
    priority: int


_PATTERN_SPECS = (
    _PatternSpec(
        SensitiveValueType.AUTHORIZATION_HEADER,
        re.compile(
            r"\bAuthorization[ \t]*:[ \t]*"
            r"(?P<secret>\S(?:[^\r\n]*?\S)?)[ \t]*(?=\r?$)",
            re.IGNORECASE | re.MULTILINE,
        ),
    ),
    _PatternSpec(
        SensitiveValueType.API_KEY,
        re.compile(
            r"\b(?:api[_-]?key|access[_-]?key|secret[_-]?key)\s*[:=]\s*"
            + _ASSIGNED_VALUE_PATTERN,
            re.IGNORECASE,
        ),
    ),
    _PatternSpec(
        SensitiveValueType.PASSWORD,
        re.compile(
            r"\b(?:password|passwd|pwd)\s*[:=]\s*" + _ASSIGNED_VALUE_PATTERN,
            re.IGNORECASE,
        ),
    ),
    _PatternSpec(
        SensitiveValueType.BEARER_TOKEN,
        re.compile(
            r"\bBearer\s+(?P<secret>[A-Za-z0-9._~+/=-]+)",
            re.IGNORECASE,
        ),
    ),
    _PatternSpec(
        SensitiveValueType.API_KEY,
        re.compile(
            r"(?<![\w-])(?P<secret>(?:sk-[A-Za-z0-9_-]{8,}|"
            r"AKIA[A-Z0-9]{16}))(?![\w-])"
        ),
    ),
    _PatternSpec(
        SensitiveValueType.EMAIL,
        re.compile(
            r"(?<![\w.+-])(?P<secret>[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})",
            re.IGNORECASE,
        ),
    ),
    _PatternSpec(
        SensitiveValueType.IP_ADDRESS,
        re.compile(r"(?<![\w.])(?P<secret>(?:\d{1,3}\.){3}\d{1,3})(?![\w.])"),
        _is_ip_address,
    ),
    _PatternSpec(
        SensitiveValueType.IP_ADDRESS,
        re.compile(
            r"(?<![\w:])(?P<secret>(?=[0-9A-Fa-f:]*:)"
            r"[0-9A-Fa-f:]{2,})(?![\w:])"
        ),
        _is_ip_address,
    ),
    _PatternSpec(
        SensitiveValueType.CREDIT_CARD,
        re.compile(r"(?<!\d)(?P<secret>(?:\d[ -]?){12,18}\d)(?!\d)"),
    ),
)


class RedactionService:
    """Detect and mask supported sensitive values deterministically."""

    @classmethod
    def detect_sensitive_values(cls, text: str) -> list[SensitiveValue]:
        """Return safe metadata for non-overlapping sensitive spans."""
        candidates = sorted(
            cls._iter_candidates(text),
            key=lambda candidate: (
                candidate.priority,
                candidate.start,
                -(candidate.end - candidate.start),
            ),
        )
        selected: list[_Candidate] = []
        for candidate in candidates:
            if any(
                candidate.start < existing.end and existing.start < candidate.end
                for existing in selected
            ):
                continue
            selected.append(candidate)

        return [
            cls._to_sensitive_value(text, candidate)
            for candidate in sorted(selected, key=lambda candidate: candidate.start)
        ]

    @classmethod
    def redact_text(cls, text: str) -> RedactionResult:
        """Mask detected values while preserving all other source text."""
        detections = cls.detect_sensitive_values(text)
        redacted_text = text
        for detection in reversed(detections):
            redacted_text = (
                redacted_text[: detection.start]
                + detection.replacement
                + redacted_text[detection.end :]
            )
        return RedactionResult(
            redacted_text=redacted_text,
            detections=tuple(detections),
        )

    @staticmethod
    def _iter_candidates(text: str) -> Iterator[_Candidate]:
        for priority, specification in enumerate(_PATTERN_SPECS):
            for match in specification.pattern.finditer(text):
                start, end = match.span("secret")
                value = text[start:end]
                if specification.validator and not specification.validator(value):
                    continue
                yield _Candidate(
                    category=specification.category,
                    start=start,
                    end=end,
                    priority=priority,
                )

    @staticmethod
    def _to_sensitive_value(
        text: str,
        candidate: _Candidate,
    ) -> SensitiveValue:
        preceding_newline = text.rfind("\n", 0, candidate.start)
        return SensitiveValue(
            category=candidate.category,
            start=candidate.start,
            end=candidate.end,
            line_number=text.count("\n", 0, candidate.start) + 1,
            column_number=candidate.start - preceding_newline,
            replacement=_REPLACEMENTS[candidate.category],
        )
