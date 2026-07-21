"""FastAPI application entry point."""

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import get_settings


APP_DIRECTORY = Path(__file__).resolve().parent
settings = get_settings()

app = FastAPI(title=settings.app_name, debug=settings.debug)
app.mount(
    "/static",
    StaticFiles(directory=APP_DIRECTORY / "static"),
    name="static",
)
templates = Jinja2Templates(directory=APP_DIRECTORY / "templates")


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request) -> HTMLResponse:
    """Render the application dashboard."""

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={"app_name": settings.app_name},
    )


@app.get("/health")
def health() -> dict[str, str]:
    """Return a lightweight application health response."""

    return {"status": "ok"}
