"""Tests for generated and editable report persistence."""

from datetime import UTC

from sqlalchemy import delete, inspect, select
from sqlalchemy.orm import Session, sessionmaker

from app.models import AnalysisRun, Incident, Report


def _incident_with_analysis_run() -> tuple[Incident, AnalysisRun]:
    incident = Incident(
        public_id="INC-000001",
        name="Checkout failures",
        description="Intermittent checkout errors after deployment v2.4.1",
        affected_service="checkout",
    )
    analysis_run = AnalysisRun(
        incident=incident,
        model_name="fake-model",
        provider_name="fake",
    )
    return incident, analysis_run


def test_report_model_fields_constraints_and_relationships() -> None:
    mapper = inspect(Report)
    columns = Report.__table__.c

    assert set(mapper.columns.keys()) == {
        "id",
        "incident_id",
        "analysis_run_id",
        "generated_text",
        "editable_text",
        "final_text",
        "export_metadata",
        "created_at",
        "updated_at",
    }
    assert set(mapper.relationships.keys()) == {"analysis_run", "incident"}
    assert columns.id.primary_key is True
    assert columns.incident_id.nullable is False
    assert columns.analysis_run_id.nullable is False
    assert columns.generated_text.nullable is False
    assert columns.editable_text.nullable is False
    assert columns.final_text.nullable is True
    assert columns.export_metadata.nullable is False
    assert {
        foreign_key.target_fullname for foreign_key in columns.incident_id.foreign_keys
    } == {"incidents.id"}
    assert {
        foreign_key.target_fullname
        for foreign_key in columns.analysis_run_id.foreign_keys
    } == {"analysis_runs.id"}
    assert all(
        foreign_key.ondelete == "CASCADE"
        for column in (columns.incident_id, columns.analysis_run_id)
        for foreign_key in column.foreign_keys
    )


def test_report_text_metadata_and_links_round_trip(
    model_session_factory: sessionmaker[Session],
) -> None:
    incident, analysis_run = _incident_with_analysis_run()
    report = Report(
        incident=incident,
        analysis_run=analysis_run,
        generated_text="# Generated postmortem",
        editable_text="# Investigator draft",
        final_text="# Approved postmortem",
        export_metadata={"format": "markdown", "filename": "checkout.md"},
    )

    with model_session_factory() as session:
        session.add(report)
        session.flush()
        report_id = report.id
        incident_id = incident.id
        analysis_run_id = analysis_run.id
        session.commit()

    with model_session_factory() as session:
        loaded_report = session.get(Report, report_id)
        assert loaded_report is not None
        assert loaded_report.generated_text == "# Generated postmortem"
        assert loaded_report.editable_text == "# Investigator draft"
        assert loaded_report.final_text == "# Approved postmortem"
        assert loaded_report.export_metadata == {
            "format": "markdown",
            "filename": "checkout.md",
        }
        assert loaded_report.incident.id == incident_id
        assert loaded_report.analysis_run.id == analysis_run_id
        assert loaded_report.created_at.tzinfo is UTC
        assert loaded_report.updated_at.tzinfo is UTC


def test_in_place_export_metadata_change_persists(
    model_session_factory: sessionmaker[Session],
) -> None:
    incident, analysis_run = _incident_with_analysis_run()
    report = Report(
        incident=incident,
        analysis_run=analysis_run,
        generated_text="Generated",
        editable_text="Editable",
    )

    with model_session_factory() as session:
        session.add(report)
        session.flush()
        report_id = report.id
        assert report.final_text is None
        assert report.export_metadata == {}
        report.export_metadata["filename"] = "incident-report.md"
        session.commit()

    with model_session_factory() as session:
        loaded_report = session.get(Report, report_id)
        assert loaded_report is not None
        assert loaded_report.export_metadata == {"filename": "incident-report.md"}


def test_database_delete_of_incident_cascades_to_report(
    model_session_factory: sessionmaker[Session],
) -> None:
    incident, analysis_run = _incident_with_analysis_run()
    report = Report(
        incident=incident,
        analysis_run=analysis_run,
        generated_text="Generated",
        editable_text="Editable",
    )

    with model_session_factory() as session:
        session.add(report)
        session.flush()
        incident_id = incident.id
        report_id = report.id
        session.commit()

    with model_session_factory() as session:
        session.execute(delete(Incident).where(Incident.id == incident_id))
        session.commit()

    with model_session_factory() as session:
        assert session.scalar(select(Report).where(Report.id == report_id)) is None
