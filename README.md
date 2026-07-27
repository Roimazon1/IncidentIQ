# IncidentIQ

IncidentIQ is an AI-assisted incident-response and root-cause investigation
tool. It accepts incident descriptions, logs, traces, monitoring data,
deployment notes, API or database errors, and support messages. It turns that
evidence into a structured investigation and an editable postmortem draft.

The system is intentionally designed for critical AI use. AI output is treated
as a proposal to investigate, not as truth. Facts, assumptions, hypotheses,
recommended actions, and reasoning risks remain separate; citations are checked
against the evidence supplied to that analysis run; uncertainty stays visible;
and a human reviewer controls confirmation.

## Submitted by

- Roi Mazon
- Amnon Yaakov

## Main Features

- Create and edit incidents and retain their analysis history.
- Paste evidence or upload `.txt`, `.log`, `.json`, `.csv`, and `.md` files.
- Classify evidence and preview both saved content and the redacted form used
  for external AI requests.
- Produce a professional summary with impact, uncertainty, facts, assumptions,
  and unknowns.
- Reconstruct direct and inferred timeline events with confidence, uncertainty,
  and evidence references.
- Generate at least three ranked hypotheses with supporting evidence,
  contradicting evidence, missing evidence, confidence, risk, and a recommended
  validation test.
- Run a separate adversarial critique and identify reasoning risks, open
  questions, and evidence-linked investigation actions.
- Validate evidence identifiers, line ranges, and quoted excerpts
  deterministically.
- Review facts, reclassify facts as assumptions, edit timeline descriptions,
  add human notes, and set human hypothesis confidence and status.
- Preserve original AI-generated values separately from human-reviewed values.
- Generate, edit, save, reopen, download, and print a sanitized postmortem
  draft.
- Retain safe generation metadata without rendering raw provider responses or
  secrets.

IncidentIQ never executes AI-recommended operational commands.

## Architecture

IncidentIQ uses a server-rendered layered architecture:

| Layer | Responsibility |
| --- | --- |
| FastAPI routers | Validate HTTP input, enforce incident/run scoping, and translate service results into HTML or downloads |
| Jinja2 templates and static assets | Render the Bootstrap dashboard, evidence, analysis, review, report, and print views |
| Domain services | Handle ingestion, preprocessing, redaction, analysis, deterministic validation, human review, and report generation |
| Typed schemas | Define request, provider, analysis-output, review, validation, and report contracts with Pydantic |
| AI provider boundary | Select the offline fake provider or Gemini while using the same registered prompts and validated output schemas |
| SQLAlchemy models | Persist incidents, evidence, analysis/audit records, human reviews, and report drafts in SQLite |

Versioned prompt files live in `app/prompts/`, synthetic example evidence lives
in `data/demo_checkout_incident/`, and command-line entry points live in
`scripts/`.

## Technology Stack

- Python 3.12+
- FastAPI
- SQLAlchemy 2.x
- Pydantic and pydantic-settings
- SQLite
- Jinja2
- Bootstrap 5.3.8, vendored locally
- Vanilla JavaScript
- Google Gen AI SDK
- Pytest
- Ruff

## Prerequisites

- Windows with PowerShell
- Python 3.12 or newer
- Internet access to install dependencies
- Optional: a Gemini API key when using the Gemini provider

Run every command below from the repository root.

## Setup with Windows PowerShell

Confirm Python is available, create a virtual environment, and install the
project dependencies:

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Activating the environment is optional because the examples invoke its Python
executable directly. To activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Create a local configuration file without overwriting an existing one:

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

## Safe Configuration

Settings load from environment variables and the local `.env` file.
`.env.example` contains safe defaults:

```dotenv
APP_NAME=IncidentIQ
DATABASE_URL=sqlite:///./incidentiq.db
AI_PROVIDER=fake
GEMINI_API_KEY=
GEMINI_MODEL=gemini-3.5-flash-lite
MAX_UPLOAD_BYTES=10485760
DISPLAY_TIMEZONE=UTC
DEBUG=true
```

The local `.env` file and SQLite database files are ignored by Git. Never place
a real API key in source control, `.env.example`, documentation, fixtures, or
example output.

### Fake Provider

`AI_PROVIDER=fake` is the default. It uses deterministic local fixtures, makes
no external AI request, and requires no API key. It supports the complete
analysis and postmortem demonstration.

### Gemini Provider

To use Gemini, edit only your local `.env`:

```dotenv
AI_PROVIDER=gemini
GEMINI_MODEL=gemini-3.5-flash-lite
```

Set `GEMINI_API_KEY` in that same untracked local file. The repository's current
default Gemini model is `gemini-3.5-flash-lite`.

Local runs read `GEMINI_API_KEY` only from the untracked `.env` file. The key is
never exposed in source code, GitHub, browser JavaScript, or documentation.

Before a Gemini request, IncidentIQ normalizes and redacts the evidence. The
provider receives the redacted evidence manifest or the typed, reviewed report
input required for that request. Original evidence and raw audit data are not
rendered in public errors, analysis pages, or exports.

## Initialize the Database

Create missing SQLite tables without deleting existing data:

```powershell
.\.venv\Scripts\python.exe scripts\init_db.py
```

The default database is the local ignored file `incidentiq.db`. Use
`DATABASE_URL` in `.env` to select a different SQLite location.

## Seed the Synthetic Demo Incident

Load the bundled checkout incident:

```powershell
.\.venv\Scripts\python.exe scripts\seed_demo.py
```

The seed command is idempotent. Running it again verifies the existing
synthetic incident and adds only missing evidence; it does not duplicate the
incident or its files.

The dataset is entirely synthetic and contains uncertainty, contradictions,
multiple plausible hypotheses, and an unrelated distracting warning:

| File | Example input represented |
| --- | --- |
| `data/demo_checkout_incident/incident.json` | Incident metadata and the allowlisted evidence manifest |
| `deployment-v2.4.1.md` | Deployment changes, checks, observations, and unresolved questions |
| `checkout.log` | Checkout HTTP 500 errors, timeouts, retries, pool pressure, rollback, and recovery signals |
| `monitoring.csv` | Time-series error-rate, latency, saturation, and availability observations |
| `support-message.txt` | Synthetic customer-impact and support observations |

No file in this dataset contains real private data or credentials.

## Run the Application

Start FastAPI:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Open:

- Dashboard: `http://localhost:8000/`
- Health check: `http://localhost:8000/health`
- Interactive API documentation: `http://localhost:8000/docs`

Stop the development server with `Ctrl+C`.

## Use the Demo Incident

1. Keep `AI_PROVIDER=fake` for an offline deterministic demonstration.
2. Initialize and seed the database.
3. Start the application and open the dashboard.
4. Open **Checkout failures after v2.4.1 deployment**.
5. Review the four saved evidence items and their redaction previews.
6. Select **Run analysis**.
7. Inspect the summary, facts and assumptions, timeline, hypotheses,
   adversarial critique, next actions, reasoning risks, open questions, and
   safe audit metadata.
8. Submit human review decisions: accept, reject, or reclassify a fact; edit a
   timeline description; add a note; or override a hypothesis confidence and
   status.
9. Select **Generate or open draft**, edit the postmortem, and save it.
10. Reopen the draft, download Markdown, or open the print view.

Human decisions persist with the incident-scoped analysis run. Original AI
values and original generated report content remain separately retained.

## Expected Analysis Output

For the bundled inputs, the application produces the following structured
views rather than raw JSON:

- incident summary, impact, uncertainty, and unknowns;
- confirmed and unconfirmed claims with support status and citations;
- assumptions and evidence still required;
- direct and inferred timeline events with confidence and uncertainty;
- three or more ranked hypotheses, each with evidence for and against, missing
  evidence, confidence, risk, and a validation test;
- an adversarial challenge with ignored evidence and an alternative hypothesis
  where available;
- prioritized investigation actions linked to hypotheses and evidence;
- reasoning risks and fallacies with mitigations;
- open questions with the evidence and resolution criteria needed;
- safe provider, model, prompt-version, and validation audit metadata; and
- a structured editable postmortem draft.

See the
[sanitized demo analysis output](docs/examples/demo-analysis-output.md) for a
concise reviewer-facing example based on the synthetic incident and documented
Gemini prompt comparison.

Results depend on the selected provider. A hypothesis is not a confirmed root
cause unless an investigator explicitly assigns the human confirmation status.

## Evidence and Human-Review Safeguards

- Original evidence is stored locally; external AI analysis stages receive
  normalized, bounded, redacted evidence.
- Redaction masks detected API keys, bearer-token values, authorization headers,
  passwords, email addresses, IP addresses, and payment-card-like values.
- Every AI citation is checked against the analysis run's original evidence
  snapshot for evidence ID, line range, and excerpt agreement.
- Invalid quotations are marked `excerpt_mismatch`; unavailable references and
  unsupported claims remain visibly unconfirmed.
- Direct events remain separate from inferred events, and inferred uncertainty
  is displayed.
- Supporting, contradicting, and missing evidence remain separate for every
  hypothesis.
- Human edits are visibly marked and do not overwrite original AI values.
- Reports use reviewed typed data, preserve uncertainty and limitations, and
  exclude evidence bodies, secrets, and raw provider responses.

These controls assist professional judgment; they do not replace evidence
review or operational testing.

## Important Prompts and Versions

Prompts are loaded only through the central registry and are paired with strict
Pydantic output schemas.

| Prompt reference | File | Use |
| --- | --- | --- |
| `system/v1` | `system_v1.txt` | Shared evidence-first and uncertainty rules |
| `summary/v1` | `summary_v1.txt` | Summary, impact, facts, assumptions, and unknowns |
| `timeline/v1` | `timeline_v1.txt` | Direct and inferred timeline reconstruction |
| `hypotheses/v1` | `hypotheses_v1.txt` | Standard ranked hypotheses |
| `critic/v1` | `critic_v1.txt` | Core adversarial review |
| `bias/v1` | `bias_v1.txt` | Reasoning-risk and fallacy analysis |
| `open_questions/v1` | `open_questions_v1.txt` | Traceable unresolved questions and investigation actions |
| `postmortem/v2` | `postmortem_v2.txt` | Current structured postmortem generation |

Additional registered versions support evaluation and auditability:
`hypotheses/v2` is the leading deployment prompt, `critic/v2` challenges the
top hypothesis, and `postmortem/v1` is retained as the earlier postmortem
version. Normal application analysis uses `hypotheses/v1` and `critic/v1`;
current report generation explicitly uses `postmortem/v2`.

## Postmortem Export

A completed analysis can generate one editable postmortem draft. The draft
contains executive summary, impact, detection, evidence reviewed, timeline,
facts, unresolved questions, hypotheses, evidence for and against, investigation
and recovery actions, reasoning risks, AI limitations, lessons, and follow-up
actions.

Saving changes updates only the human-editable copy. The original generated
draft remains unchanged. Exports use the saved human-edited content:

- **Download Markdown** returns a sanitized `.md` file with a safe filename and
  download headers.
- **Print view** renders a print-friendly page that can be printed or saved as
  PDF through the browser.

## Prompt-Comparison Evaluation

The
[prompt-comparison evaluation](docs/evaluation/prompt-comparison-evaluation.md)
summarizes a sanitized Gemini comparison for the same synthetic incident using
neutral, leading, and adversarial variants. It records citation mismatches,
explains how the critic broadened the investigation, and documents five
reasoning-bias or fallacy risks. It includes no API key, raw provider response,
full prompt, or unredacted evidence.

The comparison script defaults to the offline fake provider:

```powershell
.\.venv\Scripts\python.exe scripts\compare_prompt_versions.py
```

Use real-provider evaluation only from a protected local environment and never
commit its generated output.

## Tests and Ruff

Automated tests use fake or mocked providers and do not require a Gemini key.

Run the complete test suite:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Run Ruff lint and formatting checks:

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
```

## Submission Artifacts

- Demo video: [IncidentIQ Final Project Demo](https://drive.google.com/file/d/1vIHGKDH4E3a9aLq5klyE8-yiEuVV7LYo/view?usp=drive_link)
