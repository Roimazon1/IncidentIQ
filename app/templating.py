"""Shared Jinja template configuration for application routes."""

from pathlib import Path

from fastapi.templating import Jinja2Templates


APP_DIRECTORY = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=APP_DIRECTORY / "templates")
