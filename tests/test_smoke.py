"""Smoke tests for the application entry points."""

from pathlib import Path

from fastapi.testclient import TestClient


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


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


def test_shared_form_guard_is_dedicated_scoped_and_restorable() -> None:
    guard_path = REPOSITORY_ROOT / "app" / "static" / "js" / "form-submit-guard.js"
    guard_source = guard_path.read_text(encoding="utf-8")
    app_source = (REPOSITORY_ROOT / "app" / "static" / "js" / "app.js").read_text(
        encoding="utf-8"
    )
    base_source = (REPOSITORY_ROOT / "app" / "templates" / "base.html").read_text(
        encoding="utf-8"
    )

    assert "form-submit-guard.js" not in app_source
    assert "defer src=\"{{ url_for('static', path='/js/form-submit-guard.js') }}\"" in (
        base_source
    )
    assert "submittingForms.has(form)" in guard_source
    assert "event.preventDefault()" in guard_source
    assert "form.querySelectorAll(submitButtonSelector)" in guard_source
    assert 'form.setAttribute("aria-busy", "true")' in guard_source
    assert "spinner-border" in guard_source
    assert "dataset.loadingLabel" in guard_source
    assert "preserveSubmitterValue" in guard_source
    assert 'window.addEventListener("pageshow"' in guard_source
    assert "restoreForm(form, state)" in guard_source
    assert "document.querySelectorAll" not in guard_source
    assert "beforeunload" not in guard_source


def test_forms_declare_contextual_loading_labels() -> None:
    template_directory = REPOSITORY_ROOT / "app" / "templates"
    template_source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            template_directory / "incident_form.html",
            template_directory / "incident_detail.html",
            template_directory / "evidence_form.html",
        )
    )

    for label in (
        "Creating incident…",
        "Saving…",
        "Uploading…",
        "Updating…",
        "Running analysis…",
    ):
        assert f'data-loading-label="{label}"' in template_source
