"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.routers.analysis import router as analysis_router
from app.routers.dashboard import router as dashboard_router
from app.routers.evidence import router as evidence_router
from app.routers.incidents import router as incidents_router
from app.routers.review import router as review_router
from app.templating import APP_DIRECTORY


settings = get_settings()

app = FastAPI(title=settings.app_name, debug=settings.debug)
app.mount(
    "/static",
    StaticFiles(directory=APP_DIRECTORY / "static"),
    name="static",
)
app.include_router(dashboard_router)
app.include_router(incidents_router)
app.include_router(evidence_router)
app.include_router(analysis_router)
app.include_router(review_router)


@app.get("/health")
def health() -> dict[str, str]:
    """Return a lightweight application health response."""

    return {"status": "ok"}
