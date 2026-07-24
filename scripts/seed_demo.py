"""Idempotently seed the synthetic checkout incident and its evidence."""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from app.models import Incident


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIRECTORY = REPOSITORY_ROOT / "data" / "demo_checkout_incident"
LOGGER = logging.getLogger(__name__)


class DemoDatasetError(RuntimeError):
    """Raised when the synthetic dataset cannot be seeded safely."""


class DemoEvidenceDefinition(BaseModel):
    """One allowlisted evidence file declared by the demo manifest."""

    source_name: str = Field(min_length=1)
    evidence_type: str = Field(min_length=1)

    model_config = ConfigDict(extra="forbid", frozen=True)


class DemoIncidentDefinition(BaseModel):
    """Validated structure of the synthetic incident manifest."""

    dataset_version: str = Field(min_length=1)
    synthetic: bool
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    affected_service: str = Field(min_length=1)
    reported_start_time: str
    evidence: tuple[DemoEvidenceDefinition, ...] = Field(min_length=1)

    model_config = ConfigDict(extra="forbid", frozen=True)


@dataclass(frozen=True, slots=True)
class DemoSeedResult:
    """Summary of one idempotent seed operation."""

    incident_public_id: str
    incident_created: bool
    added_evidence_codes: tuple[str, ...]


def load_demo_definition(
    dataset_directory: Path = DEFAULT_DATASET_DIRECTORY,
) -> DemoIncidentDefinition:
    """Load and validate the synthetic manifest without changing persistence."""
    manifest_path = dataset_directory / "incident.json"
    try:
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        definition = DemoIncidentDefinition.model_validate(manifest_data)
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError) as exc:
        raise DemoDatasetError(
            "The synthetic checkout incident manifest is invalid."
        ) from exc
    if not definition.synthetic:
        raise DemoDatasetError("The demo seed accepts only a synthetic dataset.")
    return definition


def seed_demo(
    session: Session,
    dataset_directory: Path = DEFAULT_DATASET_DIRECTORY,
) -> DemoSeedResult:
    """Create the demo incident and any missing evidence exactly once."""
    from app.models import EvidenceItem, EvidenceType, Incident
    from app.schemas.incident import IncidentCreate
    from app.services.incident_service import (
        IncidentPersistenceError,
        IncidentService,
    )
    from app.services.ingestion_service import (
        EvidencePersistenceError,
        EvidenceUpload,
        EvidenceUploadValidationError,
        IngestionService,
    )

    definition = load_demo_definition(dataset_directory)
    try:
        incident_data = IncidentCreate.model_validate(
            {
                "name": definition.name,
                "description": definition.description,
                "affected_service": definition.affected_service,
                "reported_start_time": definition.reported_start_time,
            }
        )
        uploads_by_source = _load_uploads(
            definition,
            dataset_directory,
            evidence_type_class=EvidenceType,
            evidence_upload_class=EvidenceUpload,
        )
    except (ValidationError, ValueError, OSError, UnicodeError) as exc:
        raise DemoDatasetError(
            "The synthetic checkout incident data is invalid."
        ) from exc

    try:
        matching_incidents = list(
            session.scalars(select(Incident).where(Incident.name == incident_data.name))
        )
        if len(matching_incidents) > 1:
            raise DemoDatasetError(
                "Multiple incidents match the synthetic demo identity."
            )

        incident_created = not matching_incidents
        if incident_created:
            incident = IncidentService(session).create_incident(
                incident_data,
                commit=False,
            )
        else:
            incident = matching_incidents[0]
            _require_matching_incident(incident, incident_data)

        existing_evidence = list(
            session.scalars(
                select(EvidenceItem).where(EvidenceItem.incident_id == incident.id)
            )
        )
        existing_by_source: dict[str, EvidenceItem] = {}
        for evidence in existing_evidence:
            if evidence.source_name in existing_by_source:
                raise DemoDatasetError(
                    "Duplicate evidence source names exist on the demo incident."
                )
            existing_by_source[evidence.source_name] = evidence

        missing_uploads = []
        for source_name, upload in uploads_by_source.items():
            existing = existing_by_source.get(source_name)
            if existing is None:
                missing_uploads.append(upload)
                continue
            expected_text = IngestionService.decode_text(
                upload.filename,
                upload.content,
            )
            expected_checksum = IngestionService.calculate_checksum(expected_text)
            if (
                existing.evidence_type is not upload.evidence_type
                or existing.checksum != expected_checksum
                or existing.original_text != expected_text
            ):
                raise DemoDatasetError(
                    f"Existing demo evidence {source_name} does not match the dataset."
                )

        added_evidence = (
            IngestionService(session).ingest_uploaded_files(
                incident.public_id,
                missing_uploads,
                commit=False,
            )
            if missing_uploads
            else []
        )
        session.commit()
        session.refresh(incident)
    except DemoDatasetError:
        session.rollback()
        raise
    except (
        EvidencePersistenceError,
        EvidenceUploadValidationError,
        IncidentPersistenceError,
        SQLAlchemyError,
    ) as exc:
        session.rollback()
        raise DemoDatasetError(
            "The synthetic checkout incident could not be seeded."
        ) from exc

    return DemoSeedResult(
        incident_public_id=incident.public_id,
        incident_created=incident_created,
        added_evidence_codes=tuple(
            evidence.evidence_code for evidence in added_evidence
        ),
    )


def _load_uploads(
    definition: DemoIncidentDefinition,
    dataset_directory: Path,
    *,
    evidence_type_class: type,
    evidence_upload_class: type,
) -> dict[str, object]:
    uploads: dict[str, object] = {}
    for evidence in definition.evidence:
        source_path = (dataset_directory / evidence.source_name).resolve()
        if source_path.parent != dataset_directory.resolve():
            raise DemoDatasetError("A demo evidence path leaves the dataset directory.")
        if evidence.source_name in uploads:
            raise DemoDatasetError(
                "The demo manifest contains a duplicate evidence source."
            )
        uploads[evidence.source_name] = evidence_upload_class(
            filename=evidence.source_name,
            content=source_path.read_bytes(),
            evidence_type=evidence_type_class(evidence.evidence_type),
        )
    return uploads


def _require_matching_incident(
    incident: Incident,
    incident_data: object,
) -> None:
    expected_values = incident_data.model_dump()
    if any(
        getattr(incident, field_name) != expected_value
        for field_name, expected_value in expected_values.items()
    ):
        raise DemoDatasetError(
            "An existing incident uses the synthetic demo name with different data."
        )


def main() -> None:
    """Initialize the configured database and seed the synthetic incident."""
    if str(REPOSITORY_ROOT) not in sys.path:
        sys.path.insert(0, str(REPOSITORY_ROOT))

    from app.database import SessionLocal, engine
    from scripts.init_db import initialize_database

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    initialize_database(engine)
    with SessionLocal() as session:
        result = seed_demo(session)
    LOGGER.info(
        "Demo incident %s ready; incident_created=%s; evidence_added=%s",
        result.incident_public_id,
        result.incident_created,
        len(result.added_evidence_codes),
    )


if __name__ == "__main__":
    main()
