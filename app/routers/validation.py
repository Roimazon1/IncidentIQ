"""Validation presentation helpers shared by HTML routers."""

from pydantic import ValidationError


def validation_messages(exc: ValidationError) -> list[str]:
    """Return user-facing messages from a Pydantic validation error."""
    return [error["msg"] for error in exc.errors()]
