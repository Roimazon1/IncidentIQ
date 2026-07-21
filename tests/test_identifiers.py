"""Tests for deterministic incident and evidence identifier generation."""

from collections.abc import Callable

import pytest

from app.models.identifiers import (
    MAX_EVIDENCE_SEQUENCE,
    MAX_INCIDENT_SEQUENCE,
    generate_evidence_code,
    generate_incident_public_id,
)


@pytest.mark.parametrize(
    ("sequence_number", "expected"),
    [
        (1, "INC-000001"),
        (42, "INC-000042"),
        (MAX_INCIDENT_SEQUENCE, "INC-999999"),
    ],
)
def test_generate_incident_public_id_uses_locked_format(
    sequence_number: int,
    expected: str,
) -> None:
    assert generate_incident_public_id(sequence_number) == expected


@pytest.mark.parametrize(
    ("sequence_number", "expected"),
    [
        (1, "E-001"),
        (42, "E-042"),
        (MAX_EVIDENCE_SEQUENCE, "E-999"),
    ],
)
def test_generate_evidence_code_uses_locked_format(
    sequence_number: int,
    expected: str,
) -> None:
    assert generate_evidence_code(sequence_number) == expected


def test_identifier_generation_is_deterministic() -> None:
    assert generate_incident_public_id(17) == generate_incident_public_id(17)
    assert generate_evidence_code(17) == generate_evidence_code(17)


@pytest.mark.parametrize("sequence_number", [0, -1])
@pytest.mark.parametrize(
    "generator",
    [generate_incident_public_id, generate_evidence_code],
)
def test_identifier_generation_rejects_non_positive_sequences(
    sequence_number: int,
    generator: Callable[[object], str],
) -> None:
    with pytest.raises(ValueError):
        generator(sequence_number)


@pytest.mark.parametrize("sequence_number", [True, 1.0, "1"])
@pytest.mark.parametrize(
    "generator",
    [generate_incident_public_id, generate_evidence_code],
)
def test_identifier_generation_rejects_non_integer_sequences(
    sequence_number: object,
    generator: Callable[[object], str],
) -> None:
    with pytest.raises(TypeError):
        generator(sequence_number)


def test_incident_identifier_rejects_fixed_width_overflow() -> None:
    with pytest.raises(ValueError):
        generate_incident_public_id(MAX_INCIDENT_SEQUENCE + 1)


def test_evidence_identifier_rejects_fixed_width_overflow() -> None:
    with pytest.raises(ValueError):
        generate_evidence_code(MAX_EVIDENCE_SEQUENCE + 1)
