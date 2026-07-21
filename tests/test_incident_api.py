"""Focused tests for incident service-backed HTTP routes."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from app.database import get_db
from app.main import app


@pytest.fixture
def incident_client(
    database_session_factory: sessionmaker[Session],
) -> Iterator[TestClient]:
    def override_get_db() -> Iterator[Session]:
        with database_session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.clear()


def _incident_form_data(**overrides: str) -> dict[str, str]:
    values = {
        "name": "Checkout failures",
        "description": "Intermittent checkout errors",
        "affected_service": "checkout",
        "reported_start_time": "2025-01-01T10:00:00Z",
    }
    values.update(overrides)
    return values


def test_new_incident_form_renders(incident_client: TestClient) -> None:
    response = incident_client.get("/incidents/new")

    assert response.status_code == 200
    assert "Create incident" in response.text
    assert "Incident title" in response.text
    assert "Evidence type" not in response.text


def test_incidents_can_be_created_reopened_and_listed(
    incident_client: TestClient,
) -> None:
    first_response = incident_client.post(
        "/incidents",
        data=_incident_form_data(),
        follow_redirects=False,
    )
    second_response = incident_client.post(
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

    detail_response = incident_client.get("/incidents/INC-000001")
    assert detail_response.status_code == 200
    assert "Checkout failures" in detail_response.text
    assert "Intermittent checkout errors" in detail_response.text
    assert "DRAFT" in detail_response.text

    list_response = incident_client.get("/incidents")
    assert list_response.status_code == 200
    assert [item["public_id"] for item in list_response.json()] == [
        "INC-000002",
        "INC-000001",
    ]


def test_incident_update_persists_and_can_clear_reported_start_time(
    incident_client: TestClient,
) -> None:
    incident_client.post("/incidents", data=_incident_form_data())

    update_response = incident_client.post(
        "/incidents/INC-000001",
        data=_incident_form_data(
            description="Updated investigation description",
            reported_start_time="",
        ),
        follow_redirects=False,
    )

    assert update_response.status_code == 303
    assert update_response.headers["location"] == "/incidents/INC-000001"

    detail_response = incident_client.get("/incidents/INC-000001")
    assert "Updated investigation description" in detail_response.text
    assert "Not provided" in detail_response.text

    listed_incident = incident_client.get("/incidents").json()[0]
    assert listed_incident["reported_start_time"] is None
    assert listed_incident["status"] == "DRAFT"


def test_invalid_incident_form_returns_clear_validation_error(
    incident_client: TestClient,
) -> None:
    response = incident_client.post(
        "/incidents",
        data=_incident_form_data(name="   "),
    )

    assert response.status_code == 422
    assert "Please correct the following" in response.text
    assert "at least 1 character" in response.text
    assert incident_client.get("/incidents").json() == []


def test_missing_incident_returns_not_found(incident_client: TestClient) -> None:
    response = incident_client.get("/incidents/INC-999999")

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Incident INC-999999 was not found."
    }


def test_updating_missing_incident_returns_not_found(
    incident_client: TestClient,
) -> None:
    response = incident_client.post(
        "/incidents/INC-999999",
        data=_incident_form_data(),
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Incident INC-999999 was not found."
    }
