# IncidentIQ

IncidentIQ is an AI-assisted incident response and root-cause analysis tool.

The system accepts incident evidence such as logs, error traces, deployment notes,
monitoring alerts, and support messages. It produces a structured analysis while
keeping facts, assumptions, hypotheses, and recommended actions separate.

## Core Principle

AI output is not treated as truth.

Every factual claim should be linked to evidence, uncertainty must be visible,
and a human reviewer must confirm conclusions.

## Technology Stack

- Python 3.12+
- FastAPI
- SQLAlchemy
- Pydantic
- SQLite
- Jinja2
- Bootstrap 5.3.8, vendored locally
- Vanilla JavaScript
- Pytest
- Ruff

## Prerequisites

- Python 3.12 or newer
- Windows PowerShell

Run all commands below from the repository root.

## Setup and Installation

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The commands use the virtual environment's interpreter directly, so activating
the environment is optional.

## Configuration

The application uses the defaults shown in `.env.example` and can start without
a local `.env` file. To create a local configuration file, run this once when
`.env` does not already exist:

```powershell
Copy-Item .env.example .env
```

The default `AI_PROVIDER=fake` setting does not require Gemini credentials. Keep
the real `GEMINI_API_KEY` only in the ignored `.env` file; never add it to
`.env.example`. Select the real Gemini model through `GEMINI_MODEL`.
Set `DISPLAY_TIMEZONE` to an IANA timezone name such as `UTC` or
`Asia/Jerusalem` to control how incident timestamps are displayed. Stored values
and HTML `datetime` attributes remain ISO timestamps.

## Initialize the Database

Create any missing tables in the configured database:

```powershell
.venv\Scripts\python.exe scripts\init_db.py
```

The command is idempotent: running it again preserves existing data.

## Run

```powershell
.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/` for the dashboard. The health endpoint is available
at `http://127.0.0.1:8000/health`.

## Test

Run the current automated test suite:

```powershell
.venv\Scripts\python.exe -m pytest
```

Run only the application smoke tests:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_smoke.py
```

## Lint

```powershell
.venv\Scripts\python.exe -m ruff check .
```

## Current Status

The application currently provides environment-backed settings, SQLAlchemy
engine and session wiring, incident creation and editing, pasted-text and
multi-file evidence creation, a FastAPI dashboard, a health endpoint, local
static assets, upload validation, evidence-type selection and correction, and
saved original-evidence previews. The dashboard lists saved incidents with their
current status. Automated tests cover the current behavior. Evidence
preprocessing and AI analysis belong to later tasks and are not implemented yet.
