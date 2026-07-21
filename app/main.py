"""FastAPI application entry point."""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.routers.incidents import router as incidents_router
from app.templating import APP_DIRECTORY, templates


settings = get_settings()

app = FastAPI(title=settings.app_name, debug=settings.debug)
app.mount(
    "/static",
    StaticFiles(directory=APP_DIRECTORY / "static"),
    name="static",
)
app.include_router(incidents_router)


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
