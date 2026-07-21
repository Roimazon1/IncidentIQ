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

- Python 3.13
- FastAPI
- SQLAlchemy
- Pydantic
- SQLite
- Jinja2
- Bootstrap 5
- Vanilla JavaScript
- Pytest
- Ruff

## Local Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run

The application run command will be added after the initial scaffold is created.

## Current Status

Project environment and repository initialization.
