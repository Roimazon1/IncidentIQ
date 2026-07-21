"""Focused tests for deterministic sensitive-value redaction."""

import pytest

from app.services.redaction_service import (
    RedactionService,
    SensitiveValueType,
)


@pytest.mark.parametrize(
    ("text", "secret", "category", "replacement"),
    [
        (
            "api_key=sk-production-secret-1234",
            "sk-production-secret-1234",
            SensitiveValueType.API_KEY,
            "[REDACTED_API_KEY]",
        ),
        (
            "request used Bearer eyJhbGciOiJIUzI1NiJ9.payload.signature",
            "eyJhbGciOiJIUzI1NiJ9.payload.signature",
            SensitiveValueType.BEARER_TOKEN,
            "[REDACTED_BEARER_TOKEN]",
        ),
        (
            "password='checkout-secret'",
            "'checkout-secret'",
            SensitiveValueType.PASSWORD,
            "[REDACTED_PASSWORD]",
        ),
        (
            "owner=oncall@example.com",
            "oncall@example.com",
            SensitiveValueType.EMAIL,
            "[REDACTED_EMAIL]",
        ),
        (
            "client_ip=203.0.113.42",
            "203.0.113.42",
            SensitiveValueType.IP_ADDRESS,
            "[REDACTED_IP_ADDRESS]",
        ),
        (
            "Authorization: Basic dXNlcjpwYXNzd29yZA==",
            "Basic dXNlcjpwYXNzd29yZA==",
            SensitiveValueType.AUTHORIZATION_HEADER,
            "[REDACTED_AUTHORIZATION_HEADER]",
        ),
        (
            "card=4111 1111 1111 1111",
            "4111 1111 1111 1111",
            SensitiveValueType.CREDIT_CARD,
            "[REDACTED_CREDIT_CARD]",
        ),
    ],
)
def test_each_required_sensitive_value_is_masked(
    text: str,
    secret: str,
    category: SensitiveValueType,
    replacement: str,
) -> None:
    original = text

    result = RedactionService.redact_text(text)

    assert secret not in result.redacted_text
    assert replacement in result.redacted_text
    assert result.redaction_count == 1
    assert result.detections[0].category is category
    assert result.detections[0].replacement == replacement
    assert secret not in repr(result.detections[0])
    assert text == original


def test_ipv6_address_is_masked() -> None:
    result = RedactionService.redact_text("peer=2001:db8::8a2e:370:7334")

    assert result.redacted_text == "peer=[REDACTED_IP_ADDRESS]"
    assert result.detections[0].category is SensitiveValueType.IP_ADDRESS


def test_safe_preview_contains_location_but_not_secret() -> None:
    secret = "oncall@example.com"

    detections = RedactionService.detect_sensitive_values(
        f"checkout failed\nemail {secret}",
    )

    assert len(detections) == 1
    assert detections[0].line_number == 2
    assert detections[0].column_number == 7
    assert detections[0].preview == ("email at line 2, column 7: [REDACTED_EMAIL]")
    assert secret not in detections[0].preview


def test_authorization_header_wins_over_embedded_bearer_token() -> None:
    result = RedactionService.redact_text(
        "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload.signature",
    )

    assert result.redacted_text == "Authorization: [REDACTED_AUTHORIZATION_HEADER]"
    assert [detection.category for detection in result.detections] == [
        SensitiveValueType.AUTHORIZATION_HEADER
    ]


def test_non_sensitive_text_is_preserved_exactly() -> None:
    text = "checkout failed\nretry scheduled"

    result = RedactionService.redact_text(text)

    assert result.redacted_text == text
    assert result.detections == ()


def test_invalid_ip_address_is_not_redacted() -> None:
    text = "invalid_address=999.999.999.999"

    result = RedactionService.redact_text(text)

    assert result.redacted_text == text
    assert result.detections == ()


def test_multiple_values_are_redacted_without_changing_surrounding_text() -> None:
    result = RedactionService.redact_text(
        "user=oncall@example.com\nmessage=keep this\nclient=192.0.2.5",
    )

    assert result.redacted_text == (
        "user=[REDACTED_EMAIL]\nmessage=keep this\nclient=[REDACTED_IP_ADDRESS]"
    )
    assert result.redaction_count == 2


def test_unquoted_password_stops_before_next_query_parameter() -> None:
    result = RedactionService.redact_text(
        "/login?password=checkout-secret&redirect=orders",
    )

    assert result.redacted_text == (
        "/login?password=[REDACTED_PASSWORD]&redirect=orders"
    )
    assert result.redaction_count == 1


def test_unquoted_api_key_stops_before_next_query_parameter() -> None:
    result = RedactionService.redact_text(
        "/events?api_key=sk-production-secret-1234&limit=20",
    )

    assert result.redacted_text == ("/events?api_key=[REDACTED_API_KEY]&limit=20")
    assert result.redaction_count == 1


def test_redaction_is_deterministic() -> None:
    text = "api_key=sk-production-secret-1234"

    first_result = RedactionService.redact_text(text)

    assert RedactionService.redact_text(text) == first_result
