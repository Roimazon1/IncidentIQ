"""Focused API coverage for editable incident-scoped report drafts."""

from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from app.main import app
from app.models import AnalysisRun, AnalysisRunStatus, Incident, Report
from app.routers import reports as reports_router
from app.services.report_service import ReportProviderExecutionError


GENERATED_TEXT = "# Original AI draft\n\nRoot cause is not confirmed."
EDITABLE_TEXT = "# Investigator draft\n\nPreserve uncertainty."
RAW_AUDIT_SECRET = "raw-provider-secret"


def _persist_report(
    session_factory: sessionmaker[Session],
) -> tuple[str, str, int, int]:
    with session_factory() as session:
        incident = Incident(
            public_id="INC-000001",
            name="Checkout failures",
            description="Intermittent checkout failures.",
            affected_service="checkout",
        )
        other_incident = Incident(
            public_id="INC-000002",
            name="Unrelated incident",
            description="Separate incident scope.",
            affected_service="payments",
        )
        analysis_run = AnalysisRun(
            incident=incident,
            status=AnalysisRunStatus.COMPLETED,
            provider_name="fake",
            model_name="fixture-v1",
            raw_response=f'{{"raw_response":"{RAW_AUDIT_SECRET}"}}',
        )
        report = Report(
            incident=incident,
            analysis_run=analysis_run,
            generated_text=GENERATED_TEXT,
            editable_text=EDITABLE_TEXT,
            final_text=None,
            export_metadata={
                "generation": {
                    "provider_name": "fake",
                    "model_name": "fixture-v1",
                    "task_prompt": {
                        "name": "postmortem",
                        "version": "v2",
                    },
                }
            },
        )
        session.add_all((incident, other_incident, analysis_run, report))
        session.commit()
        return (
            incident.public_id,
            other_incident.public_id,
            analysis_run.id,
            report.id,
        )


class _GeneratingReportService:
    def __init__(self, report_id: int) -> None:
        self._report_id = report_id
        self.calls: list[tuple[str, int]] = []

    def generate_draft_report(
        self,
        public_id: str,
        run_id: int,
    ) -> SimpleNamespace:
        self.calls.append((public_id, run_id))
        return SimpleNamespace(id=self._report_id)


class _FailingReportService:
    def generate_draft_report(self, public_id: str, run_id: int) -> None:
        del public_id, run_id
        raise ReportProviderExecutionError(
            "The postmortem provider request failed safely."
        )


def test_generate_report_redirects_to_incident_scoped_preview(
    database_client: TestClient,
    database_session_factory: sessionmaker[Session],
) -> None:
    public_id, _, run_id, report_id = _persist_report(database_session_factory)
    service = _GeneratingReportService(report_id)
    app.dependency_overrides[reports_router.get_configured_report_service] = lambda: (
        service
    )

    response = database_client.post(
        f"/incidents/{public_id}/analysis/{run_id}/report",
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"] == (
        f"/incidents/{public_id}/reports/{report_id}?notice=report-draft-ready"
    )
    assert service.calls == [(public_id, run_id)]


def test_generate_report_maps_report_provider_failure_without_boundary_leak(
    database_client: TestClient,
) -> None:
    app.dependency_overrides[reports_router.get_configured_report_service] = (
        _FailingReportService
    )

    response = database_client.post(
        "/incidents/INC-000001/analysis/1/report",
    )

    assert response.status_code == 502
    assert response.json() == {
        "detail": "The postmortem provider request failed safely."
    }


def test_report_preview_distinguishes_editable_and_generated_content_without_audit(
    database_client: TestClient,
    database_session_factory: sessionmaker[Session],
) -> None:
    public_id, _, run_id, report_id = _persist_report(database_session_factory)

    response = database_client.get(f"/incidents/{public_id}/reports/{report_id}")

    assert response.status_code == 200
    assert "Editable postmortem draft" in response.text
    assert "Current human draft preview" in response.text
    assert "Original AI-generated draft" in response.text
    assert EDITABLE_TEXT in response.text
    assert GENERATED_TEXT in response.text
    assert f"Back to analysis run {run_id}" in response.text
    assert 'name="editable_text"' in response.text
    assert 'data-loading-label="Saving draft…"' in response.text
    assert RAW_AUDIT_SECRET not in response.text
    assert '"raw_response"' not in response.text


def test_save_report_updates_only_sanitized_editable_content(
    database_client: TestClient,
    database_session_factory: sessionmaker[Session],
) -> None:
    public_id, _, _, report_id = _persist_report(database_session_factory)
    submitted_secret = "human-edit-secret"
    edited_text = (
        "# Human-reviewed draft\n\n"
        "Cause remains uncertain.\n"
        f"api_key={submitted_secret}"
    )

    response = database_client.post(
        f"/incidents/{public_id}/reports/{report_id}",
        data={"editable_text": edited_text},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"] == (
        f"/incidents/{public_id}/reports/{report_id}?notice=report-draft-updated"
    )
    with database_session_factory() as session:
        report = session.get(Report, report_id)
        assert report is not None
        assert report.generated_text == GENERATED_TEXT
        assert report.editable_text.startswith("# Human-reviewed draft")
        assert "[REDACTED_API_KEY]" in report.editable_text
        assert submitted_secret not in report.editable_text
        assert report.final_text is None
        assert report.export_metadata["generation"]["task_prompt"]["version"] == "v2"


def test_report_preview_and_save_reject_cross_incident_access(
    database_client: TestClient,
    database_session_factory: sessionmaker[Session],
) -> None:
    _, other_public_id, _, report_id = _persist_report(database_session_factory)

    preview_response = database_client.get(
        f"/incidents/{other_public_id}/reports/{report_id}"
    )
    save_response = database_client.post(
        f"/incidents/{other_public_id}/reports/{report_id}",
        data={"editable_text": "Cross-incident overwrite"},
    )

    assert preview_response.status_code == 404
    assert save_response.status_code == 404
    with database_session_factory() as session:
        report = session.get(Report, report_id)
        assert report is not None
        assert report.editable_text == EDITABLE_TEXT
        assert report.generated_text == GENERATED_TEXT


def test_report_save_rejects_blank_human_draft_without_mutation(
    database_client: TestClient,
    database_session_factory: sessionmaker[Session],
) -> None:
    public_id, _, _, report_id = _persist_report(database_session_factory)

    response = database_client.post(
        f"/incidents/{public_id}/reports/{report_id}",
        data={"editable_text": "   "},
    )

    assert response.status_code == 422
    with database_session_factory() as session:
        report = session.get(Report, report_id)
        assert report is not None
        assert report.editable_text == EDITABLE_TEXT
        assert report.generated_text == GENERATED_TEXT


def test_markdown_export_uses_sanitized_human_edit_and_safe_headers(
    database_client: TestClient,
    database_session_factory: sessionmaker[Session],
) -> None:
    public_id, _, _, report_id = _persist_report(database_session_factory)
    export_secret = "markdown-export-secret"
    with database_session_factory() as session:
        report = session.get(Report, report_id)
        assert report is not None
        report.editable_text = (
            "# Human export\n\n"
            f"api_key={export_secret}\n\n"
            "<script>unsafe</script>"
        )
        session.commit()

    response = database_client.get(
        f"/incidents/{public_id}/reports/{report_id}/export.md"
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/markdown; charset=utf-8"
    assert response.headers["content-disposition"] == (
        'attachment; filename="INC-000001-postmortem.md"'
    )
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.text.startswith("# Human export")
    assert "[REDACTED_API_KEY]" in response.text
    assert "&lt;script&gt;unsafe&lt;/script&gt;" in response.text
    assert export_secret not in response.text
    assert GENERATED_TEXT not in response.text
    assert RAW_AUDIT_SECRET not in response.text
    assert '"generation"' not in response.text


def test_print_view_uses_only_sanitized_human_edit(
    database_client: TestClient,
    database_session_factory: sessionmaker[Session],
) -> None:
    public_id, _, run_id, report_id = _persist_report(database_session_factory)
    print_secret = "print-view-secret"
    with database_session_factory() as session:
        report = session.get(Report, report_id)
        assert report is not None
        report.editable_text = (
            "# Human print draft\n\n"
            f"api_key={print_secret}\n\n"
            "<iframe>unsafe</iframe>"
        )
        session.commit()

    response = database_client.get(
        f"/incidents/{public_id}/reports/{report_id}/print"
    )

    assert response.status_code == 200
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "script-src 'none'" in response.headers["content-security-policy"]
    assert "Human print draft" in response.text
    assert f"Analysis run {run_id}" in response.text
    assert "[REDACTED_API_KEY]" in response.text
    assert print_secret not in response.text
    assert "<iframe>unsafe</iframe>" not in response.text
    assert GENERATED_TEXT not in response.text
    assert RAW_AUDIT_SECRET not in response.text
    assert "Safe generation metadata" not in response.text
    assert "@media print" in response.text


def test_export_and_print_reject_cross_incident_access(
    database_client: TestClient,
    database_session_factory: sessionmaker[Session],
) -> None:
    _, other_public_id, _, report_id = _persist_report(database_session_factory)

    export_response = database_client.get(
        f"/incidents/{other_public_id}/reports/{report_id}/export.md"
    )
    print_response = database_client.get(
        f"/incidents/{other_public_id}/reports/{report_id}/print"
    )

    assert export_response.status_code == 404
    assert print_response.status_code == 404
