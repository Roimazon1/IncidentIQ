"""Focused tests for incident service-backed HTTP routes."""

from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from httpx import Response
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app import templating
from app.config import Settings
from app.models import AnalysisRun, AnalysisRunStatus, Incident
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


def _element_start_tag(response: Response, tag_name: str, element_id: str) -> str:
    id_marker = f'id="{element_id}"'
    marker_position = response.text.index(id_marker)
    tag_start = response.text.rfind(f"<{tag_name}", 0, marker_position)
    tag_end = response.text.index(">", marker_position)
    return response.text[tag_start:tag_end]


def _assert_active_incident_tab(response: Response, tab_name: str) -> None:
    tab_markup = _element_start_tag(response, "a", f"{tab_name}-tab")
    panel_markup = _element_start_tag(response, "section", f"{tab_name}-panel")

    assert 'class="nav-link active"' in tab_markup
    assert 'aria-selected="true"' in tab_markup
    assert "show active" in panel_markup


def test_new_incident_form_renders(database_client: TestClient) -> None:
    response = database_client.get("/incidents/new")

    assert response.status_code == 200
    assert "Create incident" in response.text
    assert "Incident title" in response.text
    assert "Approximate start time (optional)" in response.text
    assert 'type="datetime-local"' in response.text
    assert 'step="60"' in response.text
    assert "Leave blank if the start time is unknown." in response.text
    assert "Asia/Jerusalem" not in response.text
    assert "stored in UTC" not in response.text
    start_time_input = _element_start_tag(
        response,
        "input",
        "reported_start_time",
    )
    assert "required" not in start_time_input
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
    assert first_response.headers["location"] == (
        "/incidents/INC-000001?notice=incident-created"
    )
    assert second_response.status_code == 303
    assert second_response.headers["location"] == (
        "/incidents/INC-000002?notice=incident-created"
    )

    detail_response = database_client.get(first_response.headers["location"])
    assert detail_response.status_code == 200
    assert "Incident created successfully." in detail_response.text
    assert 'id="success-toast"' in detail_response.text
    assert 'role="status"' in detail_response.text
    assert 'aria-live="polite"' in detail_response.text
    assert 'aria-atomic="true"' in detail_response.text
    assert 'data-bs-autohide="true"' in detail_response.text
    assert 'data-bs-dismiss="toast"' in detail_response.text
    assert 'aria-label="Close"' in detail_response.text
    assert "Checkout failures" in detail_response.text
    assert "Intermittent checkout errors" in detail_response.text
    assert "DRAFT" in detail_response.text

    list_response = database_client.get("/incidents")
    assert list_response.status_code == 200
    assert [item["public_id"] for item in list_response.json()] == [
        "INC-000002",
        "INC-000001",
    ]


def test_incident_without_analysis_runs_shows_history_empty_state(
    database_client: TestClient,
) -> None:
    database_client.post("/incidents", data=_incident_form_data())

    response = database_client.get("/incidents/INC-000001")

    assert response.status_code == 200
    assert "Analysis history" in response.text
    assert "Analysis history (0)" in response.text
    assert "No analysis has been run for this incident." in response.text


def test_incident_detail_defaults_to_overview_and_validates_tab_query(
    database_client: TestClient,
) -> None:
    database_client.post("/incidents", data=_incident_form_data())

    default_response = database_client.get("/incidents/INC-000001")
    invalid_response = database_client.get("/incidents/INC-000001?tab=unknown")

    assert default_response.status_code == 200
    _assert_active_incident_tab(default_response, "overview")
    assert invalid_response.status_code == 422
    for tab_name in ("overview", "edit", "history"):
        tab_markup = _element_start_tag(
            default_response,
            "a",
            f"{tab_name}-tab",
        )
        panel_markup = _element_start_tag(
            default_response,
            "section",
            f"{tab_name}-panel",
        )
        assert f'aria-controls="{tab_name}-panel"' in tab_markup
        assert f'aria-labelledby="{tab_name}-tab"' in panel_markup


@pytest.mark.parametrize("tab_name", ["edit", "history"])
def test_incident_detail_preserves_explicit_tab_selection(
    database_client: TestClient,
    tab_name: str,
) -> None:
    database_client.post("/incidents", data=_incident_form_data())

    response = database_client.get(f"/incidents/INC-000001?tab={tab_name}")

    assert response.status_code == 200
    _assert_active_incident_tab(response, tab_name)


def test_incident_primary_actions_remain_outside_tab_panels(
    database_client: TestClient,
) -> None:
    database_client.post("/incidents", data=_incident_form_data())

    response = database_client.get("/incidents/INC-000001?tab=history")

    tab_content_position = response.text.index('id="incident-detail-tab-content"')
    assert response.text.index("Run analysis") < tab_content_position
    assert response.text.index("View all evidence") < tab_content_position
    assert response.text.index("Add evidence") < tab_content_position


def test_incident_analysis_history_lists_only_its_runs_newest_first(
    database_client: TestClient,
    database_session_factory: sessionmaker[Session],
) -> None:
    database_client.post("/incidents", data=_incident_form_data())
    database_client.post(
        "/incidents",
        data=_incident_form_data(name="Other incident"),
    )
    with database_session_factory() as session:
        first_incident = session.scalar(
            select(Incident).where(Incident.public_id == "INC-000001")
        )
        other_incident = session.scalar(
            select(Incident).where(Incident.public_id == "INC-000002")
        )
        assert first_incident is not None
        assert other_incident is not None
        session.add_all(
            [
                AnalysisRun(
                    incident_id=first_incident.id,
                    status=AnalysisRunStatus.COMPLETED,
                    started_at=datetime(2025, 1, 1, 8, 0, tzinfo=UTC),
                    provider_name="fake",
                    model_name="completed-model",
                    raw_response="completed-raw-secret",
                ),
                AnalysisRun(
                    incident_id=first_incident.id,
                    status=AnalysisRunStatus.FAILED,
                    started_at=datetime(2025, 1, 1, 9, 0, tzinfo=UTC),
                    provider_name="gemini",
                    model_name="failed-model",
                    raw_response="failed-raw-secret",
                ),
                AnalysisRun(
                    incident_id=first_incident.id,
                    status=AnalysisRunStatus.RUNNING,
                    started_at=datetime(2025, 1, 1, 10, 0, tzinfo=UTC),
                    provider_name="fake",
                    model_name="running-model",
                    raw_response="running-raw-secret",
                ),
                AnalysisRun(
                    incident_id=other_incident.id,
                    status=AnalysisRunStatus.COMPLETED,
                    started_at=datetime(2025, 1, 1, 11, 0, tzinfo=UTC),
                    provider_name="fake",
                    model_name="other-incident-model",
                    raw_response="other-raw-secret",
                ),
            ]
        )
        session.commit()
        run_ids = list(
            session.scalars(
                select(AnalysisRun.id)
                .where(AnalysisRun.incident_id == first_incident.id)
                .order_by(AnalysisRun.started_at.desc())
            )
        )

    response = database_client.get("/incidents/INC-000001")

    assert response.status_code == 200
    assert "No analysis has been run for this incident." not in response.text
    assert response.text.index("running-model") < response.text.index("failed-model")
    assert response.text.index("failed-model") < response.text.index("completed-model")
    assert all(status.value in response.text for status in AnalysisRunStatus)
    for run_id in run_ids:
        assert f"/incidents/INC-000001/analysis/{run_id}" in response.text
    assert response.text.count("Open analysis") == 3
    assert "Analysis history (3)" in response.text
    assert "other-incident-model" not in response.text
    assert "raw-secret" not in response.text


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
    assert "Approximate start time (optional)" in detail_response.text
    assert "Leave blank if the start time is unknown." in detail_response.text
    assert "Asia/Jerusalem" not in detail_response.text
    assert "stored in UTC" not in detail_response.text
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


def test_nonexistent_dst_start_time_is_still_rejected_without_timezone_copy(
    database_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured_settings = Settings(
        display_timezone="Asia/Jerusalem",
        _env_file=None,
    )
    monkeypatch.setattr(incident_router, "settings", configured_settings)

    response = database_client.post(
        "/incidents",
        data=_incident_form_data(reported_start_time="2025-03-28T02:30"),
    )

    assert response.status_code == 422
    assert "daylight-saving time change" in response.text
    assert "Asia/Jerusalem" not in response.text
    assert database_client.get("/incidents").json() == []


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
    assert update_response.headers["location"] == (
        "/incidents/INC-000001?tab=edit&notice=incident-updated"
    )

    detail_response = database_client.get(update_response.headers["location"])
    assert "Incident updated successfully." in detail_response.text
    assert "Updated investigation description" in detail_response.text
    assert "Not provided" in detail_response.text
    _assert_active_incident_tab(detail_response, "edit")

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


def test_invalid_incident_update_activates_edit_tab_and_preserves_values(
    database_client: TestClient,
) -> None:
    database_client.post("/incidents", data=_incident_form_data())

    response = database_client.post(
        "/incidents/INC-000001?tab=overview",
        data=_incident_form_data(
            name="   ",
            description="Submitted invalid update",
        ),
    )

    assert response.status_code == 422
    _assert_active_incident_tab(response, "edit")
    assert "Submitted invalid update" in response.text
    assert "at least 1 character" in response.text


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
