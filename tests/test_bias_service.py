"""Focused tests for required reasoning-risk warning validation."""

import json
from pathlib import Path

import pytest

from app.schemas.ai_outputs import ReasoningRisksOutputV1
from app.services.bias_service import (
    REQUIRED_REASONING_RISK_NAMES,
    BiasAnalysisError,
    BiasService,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "fake_ai_responses.json"


def _valid_bias_output() -> ReasoningRisksOutputV1:
    fixture_bank = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    return ReasoningRisksOutputV1.model_validate_json(
        fixture_bank["valid_bias"]["raw_response"]
    )


def test_identify_risks_requires_all_five_core_warning_categories() -> None:
    risks = BiasService.identify_risks(_valid_bias_output())

    assert {" ".join(risk.name.casefold().split()) for risk in risks} == (
        REQUIRED_REASONING_RISK_NAMES
    )
    assert all(risk.location for risk in risks)
    assert all(risk.trigger for risk in risks)
    assert all(risk.potential_effect for risk in risks)
    assert all(risk.mitigation for risk in risks)
    assert all("could" in risk.potential_effect.casefold() for risk in risks)


@pytest.mark.parametrize(
    "required_risk_name",
    (
        "confirmation bias",
        "anchoring bias",
        "automation bias",
        "post hoc fallacy",
        "overconfidence bias",
    ),
)
def test_each_required_reasoning_risk_has_actionable_mitigation(
    required_risk_name: str,
) -> None:
    risks_by_name = {
        " ".join(risk.name.casefold().split()): risk
        for risk in BiasService.identify_risks(_valid_bias_output())
    }

    risk = risks_by_name[required_risk_name]

    assert risk.location
    assert risk.trigger
    assert risk.potential_effect
    assert risk.mitigation
    assert 0 <= risk.confidence <= 100


def test_identify_risks_rejects_missing_required_warning_safely() -> None:
    output = _valid_bias_output()
    incomplete_output = output.model_copy(update={"risks": output.risks[:-1]})

    with pytest.raises(
        BiasAnalysisError,
        match="omitted a required warning category",
    ) as error_info:
        BiasService.identify_risks(incomplete_output)

    assert "raw_response" not in str(error_info.value)
    assert "checkout" not in str(error_info.value).casefold()
