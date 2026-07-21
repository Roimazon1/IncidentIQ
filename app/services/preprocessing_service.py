"""Deterministic normalization for incident evidence text."""

import csv
import json
from collections.abc import Iterable
from io import StringIO
from pathlib import PurePath
from typing import Any


class StructuredTextError(ValueError):
    """Raised when declared structured evidence cannot be parsed safely."""


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
