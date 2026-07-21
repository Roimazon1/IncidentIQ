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


def test_success_notice_is_allowlisted_and_client_script_clears_it(
    database_client: TestClient,
) -> None:
    untrusted_notice = "User-controlled success message"

    response = database_client.get("/", params={"notice": untrusted_notice})
    script_response = database_client.get("/static/js/app.js")

    assert response.status_code == 200
    assert untrusted_notice not in response.text
    assert 'id="success-toast"' not in response.text
    assert script_response.status_code == 200
    assert 'searchParams.delete("notice")' in script_response.text
    assert "history.replaceState" in script_response.text
    assert "bootstrap.Toast.getOrCreateInstance" in script_response.text
