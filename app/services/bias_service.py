"""Deterministic validation for provider-generated reasoning-risk warnings."""

from app.schemas.ai_outputs import ReasoningRiskV1, ReasoningRisksOutputV1


REQUIRED_REASONING_RISK_NAMES = frozenset(
    {
        "confirmation bias",
        "anchoring bias",
        "automation bias",
        "post hoc fallacy",
        "overconfidence bias",
    }
)


class BiasAnalysisError(ValueError):
    """Raised when structured reasoning risks omit a required category."""


class BiasService:
    """Validate and return the required provider-neutral reasoning risks."""

    @classmethod
    def identify_risks(
        cls,
        output: ReasoningRisksOutputV1,
    ) -> tuple[ReasoningRiskV1, ...]:
        """Require every locked core risk and preserve the provider ordering."""
        identified_names = {cls._normalize_name(risk.name) for risk in output.risks}
        if not REQUIRED_REASONING_RISK_NAMES.issubset(identified_names):
            raise BiasAnalysisError(
                "The reasoning-risk analysis omitted a required warning category."
            )
        return output.risks

    @staticmethod
    def _normalize_name(value: str) -> str:
        return " ".join(value.casefold().split())
