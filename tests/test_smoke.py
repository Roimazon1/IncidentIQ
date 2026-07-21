"""Smoke tests for the application entry points."""

from fastapi.testclient import TestClient


def test_health_endpoint_returns_success(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_dashboard_renders_html(database_client: TestClient) -> None:
    response = database_client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Incident dashboard" in response.text
    assert "No incidents saved yet" in response.text
    assert "Hypotheses are not confirmed facts." in response.text
