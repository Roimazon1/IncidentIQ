"""Persistence and status transitions for auditable analysis runs."""

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.models import (
    AnalysisRun,
    AnalysisRunStatus,
    EvidenceItem,
    Incident,
    IncidentStatus,
    utc_now,
)
from app.services.incident_service import IncidentService


class AnalysisEvidenceRequiredError(ValueError):
    """Raised when an incident has no evidence to analyze."""


class AnalysisAlreadyRunningError(RuntimeError):
    """Raised when an incident already has an active analysis run."""


class AnalysisRunNotFoundError(LookupError):
    """Raised when an analysis run identifier does not exist."""


class AnalysisRunTransitionError(RuntimeError):
    """Raised when a terminal or otherwise invalid transition is requested."""


class AnalysisPersistenceError(RuntimeError):
    """Raised when an analysis lifecycle write cannot be completed safely."""


class AnalysisService:
    """Create analysis runs and persist their legal lifecycle transitions."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self._incident_service = IncidentService(session)

    def start_analysis_run(
        self,
        incident_public_id: str,
        *,
        provider_name: str,
        model_name: str,
    ) -> AnalysisRun:
        """Create one running analysis for an incident that has evidence."""
        incident = self._incident_service.get_incident_or_raise(incident_public_id)
        self._require_evidence(incident)
        self._require_no_running_analysis(incident)

        analysis_run = AnalysisRun(
            incident=incident,
            provider_name=provider_name,
            model_name=model_name,
            status=AnalysisRunStatus.RUNNING,
        )
        incident.status = IncidentStatus.ANALYZING
        self.session.add(analysis_run)
        self._commit(
            analysis_run,
            failure_message="The analysis run could not be started.",
        )
        return analysis_run

    def mark_analysis_run_completed(self, run_id: int) -> AnalysisRun:
        """Move a running analysis to its successful terminal state."""
        analysis_run = self._get_analysis_run_or_raise(run_id)
        self._require_running(analysis_run, target_status=AnalysisRunStatus.COMPLETED)

        analysis_run.status = AnalysisRunStatus.COMPLETED
        analysis_run.completed_at = utc_now()
        analysis_run.error_message = None
        analysis_run.incident.status = IncidentStatus.COMPLETED
        self._commit(
            analysis_run,
            failure_message="The completed analysis run could not be saved.",
        )
        return analysis_run

    def mark_analysis_run_failed(
        self,
        run_id: int,
        *,
        error_message: str,
    ) -> AnalysisRun:
        """Retain a running analysis as failed with a safe explanation."""
        safe_error_message = error_message.strip()
        if not safe_error_message:
            raise ValueError("analysis failure explanation must not be empty")

        analysis_run = self._get_analysis_run_or_raise(run_id)
        self._require_running(analysis_run, target_status=AnalysisRunStatus.FAILED)

        analysis_run.status = AnalysisRunStatus.FAILED
        analysis_run.completed_at = utc_now()
        analysis_run.error_message = safe_error_message
        analysis_run.incident.status = IncidentStatus.FAILED
        self._commit(
            analysis_run,
            failure_message="The failed analysis run could not be saved.",
        )
        return analysis_run

    def _get_analysis_run_or_raise(self, run_id: int) -> AnalysisRun:
        analysis_run = self.session.scalar(
            select(AnalysisRun).where(AnalysisRun.id == run_id)
        )
        if analysis_run is None:
            raise AnalysisRunNotFoundError(f"Analysis run {run_id} was not found.")
        return analysis_run

    def _require_evidence(self, incident: Incident) -> None:
        evidence_id = self.session.scalar(
            select(EvidenceItem.id)
            .where(EvidenceItem.incident_id == incident.id)
            .limit(1)
        )
        if evidence_id is None:
            raise AnalysisEvidenceRequiredError(
                f"Incident {incident.public_id} requires evidence before analysis."
            )

    def _require_no_running_analysis(self, incident: Incident) -> None:
        running_run_id = self.session.scalar(
            select(AnalysisRun.id)
            .where(
                AnalysisRun.incident_id == incident.id,
                AnalysisRun.status == AnalysisRunStatus.RUNNING,
            )
            .limit(1)
        )
        if running_run_id is not None:
            raise AnalysisAlreadyRunningError(
                f"Incident {incident.public_id} already has a running analysis."
            )

    @staticmethod
    def _require_running(
        analysis_run: AnalysisRun,
        *,
        target_status: AnalysisRunStatus,
    ) -> None:
        if analysis_run.status is not AnalysisRunStatus.RUNNING:
            raise AnalysisRunTransitionError(
                f"Analysis run {analysis_run.id} cannot transition from "
                f"{analysis_run.status.value} to {target_status.value}."
            )

    def _commit(
        self,
        analysis_run: AnalysisRun,
        *,
        failure_message: str,
    ) -> None:
        try:
            self.session.flush()
            self.session.refresh(analysis_run)
            self.session.commit()
        except SQLAlchemyError as exc:
            self.session.rollback()
            raise AnalysisPersistenceError(failure_message) from exc
