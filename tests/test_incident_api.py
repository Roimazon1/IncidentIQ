"""Focused tests for incident service-backed HTTP routes."""

from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app import templating
from app.config import Settings
from app.models import Incident
from app.routers import incidents as incident_router


def _incident_form_data(**overrides: str) -> dict[str, str]:
    values = {
        "name": "Checkout failures",
        "description": "Intermittent checkout errors",
        "affected_service": "checkout",
        "reported_start_time": "2025-01-01T10:00",
    }
    values.update(overrides)
    return values


def test_new_incident_form_renders(database_client: TestClient) -> None:
    response = database_client.get("/incidents/new")

    assert response.status_code == 200
    assert "Create incident" in response.text
    assert "Incident title" in response.text
    assert 'type="datetime-local"' in response.text
    assert 'step="60"' in response.text
    assert "it is stored in UTC" in response.text
    assert "Evidence type" not in response.text


def test_incidents_can_be_created_reopened_and_listed(
    database_client: TestClient,
) -> None:
    first_response = database_client.post(
        "/incidents",
        data=_incident_form_data(),
        follow_redirects=False,
    )
    second_response = database_client.post(
        "/incidents",
        data=_incident_form_data(
            name="Payment failures",
            affected_service="payments",
        ),
        follow_redirects=False,
    )

    assert first_response.status_code == 303
    assert first_response.headers["location"] == "/incidents/INC-000001"
    assert second_response.status_code == 303
    assert second_response.headers["location"] == "/incidents/INC-000002"

    detail_response = database_client.get("/incidents/INC-000001")
    assert detail_response.status_code == 200
    assert "Checkout failures" in detail_response.text
    assert "Intermittent checkout errors" in detail_response.text
    assert "DRAFT" in detail_response.text

    list_response = database_client.get("/incidents")
    assert list_response.status_code == 200
    assert [item["public_id"] for item in list_response.json()] == [
        "INC-000002",
        "INC-000001",
    ]


def test_dashboard_lists_saved_incidents_newest_first_with_statuses(
    database_client: TestClient,
) -> None:
    database_client.post(
        "/incidents",
        data=_incident_form_data(),
    )
    database_client.post(
        "/incidents",
        data=_incident_form_data(
            name="Payment <failures>",
            affected_service="payments",
        ),
    )
    database_client.post(
        "/incidents/INC-000001/evidence/text",
        data={
            "source_name": "Checkout log",
            "original_text": "Checkout failed",
            "evidence_type": "APPLICATION_LOG",
        },
    )

    response = database_client.get("/")

    assert response.status_code == 200
    assert "No incidents saved yet" not in response.text
    assert "Payment &lt;failures&gt;" in response.text
    assert "Payment <failures>" not in response.text
    newest_position = response.text.index("INC-000002")
    older_position = response.text.index("INC-000001")
    assert newest_position < older_position
    assert "DRAFT" in response.text[newest_position:older_position]
    assert "READY" in response.text[older_position:]
    assert "/incidents/INC-000002" in response.text
    assert "/incidents/INC-000001" in response.text
    assert "payments" in response.text
    assert "checkout" in response.text


def test_incident_timestamps_use_configured_timezone_and_keep_iso_values(
    database_client: TestClient,
    database_session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured_settings = Settings(
        display_timezone="Asia/Jerusalem",
        _env_file=None,
    )
    monkeypatch.setattr(incident_router, "settings", configured_settings)
    monkeypatch.setattr(templating, "get_settings", lambda: configured_settings)
    database_client.post(
        "/incidents",
        data=_incident_form_data(
            reported_start_time="2025-07-01T12:00",
        ),
    )
    with database_session_factory() as session:
        incident = session.scalar(select(Incident))
        assert incident is not None
        assert incident.reported_start_time == datetime(
            2025,
            7,
            1,
            9,
            0,
            tzinfo=UTC,
        )
        incident.created_at = datetime(2025, 7, 1, 9, 0, tzinfo=UTC)
        session.commit()

    dashboard_response = database_client.get("/")
    detail_response = database_client.get("/incidents/INC-000001")

    assert dashboard_response.status_code == 200
    assert 'datetime="2025-07-01T09:00:00+00:00"' in dashboard_response.text
    assert "Jul 01, 2025 at 12:00 IDT" in dashboard_response.text
    assert detail_response.status_code == 200
    assert detail_response.text.count('datetime="2025-07-01T09:00:00+00:00"') == 2
    assert 'value="2025-07-01T12:00"' in detail_response.text
    assert "Enter the time in Asia/Jerusalem" in detail_response.text
    assert "Jul 01, 2025 at 12:00 IDT" in detail_response.text
    api_timestamp = database_client.get("/incidents").json()[0]["reported_start_time"]
    assert datetime.fromisoformat(api_timestamp.replace("Z", "+00:00")) == datetime(
        2025,
        7,
        1,
        9,
        0,
        tzinfo=UTC,
    )


def test_invalid_local_incident_time_returns_a_clear_form_error(
    database_client: TestClient,
) -> None:
    response = database_client.post(
        "/incidents",
        data=_incident_form_data(reported_start_time="not-a-local-time"),
    )

    assert response.status_code == 422
    assert "must be a valid local date and time" in response.text
    assert 'value="not-a-local-time"' in response.text
    assert database_client.get("/incidents").json() == []


def test_incident_update_persists_and_can_clear_reported_start_time(
    database_client: TestClient,
) -> None:
    database_client.post("/incidents", data=_incident_form_data())

    update_response = database_client.post(
        "/incidents/INC-000001",
        data=_incident_form_data(
            description="Updated investigation description",
            reported_start_time="",
        ),
        follow_redirects=False,
    )

    assert update_response.status_code == 303
    assert update_response.headers["location"] == "/incidents/INC-000001"

    detail_response = database_client.get("/incidents/INC-000001")
    assert "Updated investigation description" in detail_response.text
    assert "Not provided" in detail_response.text

    listed_incident = database_client.get("/incidents").json()[0]
    assert listed_incident["reported_start_time"] is None
    assert listed_incident["status"] == "DRAFT"


def test_invalid_incident_form_returns_clear_validation_error(
    database_client: TestClient,
) -> None:
    response = database_client.post(
        "/incidents",
        data=_incident_form_data(name="   "),
    )

    assert response.status_code == 422
    assert "Please correct the following" in response.text
    assert "at least 1 character" in response.text
    assert database_client.get("/incidents").json() == []


def test_missing_incident_returns_not_found(database_client: TestClient) -> None:
    response = database_client.get("/incidents/INC-999999")

    assert response.status_code == 404
    assert response.json() == {"detail": "Incident INC-999999 was not found."}


def test_updating_missing_incident_returns_not_found(
    database_client: TestClient,
) -> None:
    response = database_client.post(
        "/incidents/INC-999999",
        data=_incident_form_data(),
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "Incident INC-999999 was not found."}
