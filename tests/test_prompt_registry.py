"""Focused tests for centralized versioned prompt registration and loading."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from app.schemas.ai_provider import (
    AIFailureCategory,
    AnalysisStage,
    PromptBundle,
    PromptName,
    PromptReference,
    PromptVersion,
)
from app.services.ai_provider import PromptBundleValidator, PromptResolver
from app.services.prompt_registry import (
    PromptRegistry,
    PromptRegistryError,
    PromptRole,
)


EXPECTED_PROMPTS = {
    PromptName.SYSTEM: ("system_v1.txt", AnalysisStage.SYSTEM, PromptRole.SYSTEM),
    PromptName.SUMMARY: ("summary_v1.txt", AnalysisStage.SUMMARY, PromptRole.TASK),
    PromptName.TIMELINE: (
        "timeline_v1.txt",
        AnalysisStage.TIMELINE,
        PromptRole.TASK,
    ),
    PromptName.HYPOTHESES: (
        "hypotheses_v1.txt",
        AnalysisStage.HYPOTHESES,
        PromptRole.TASK,
    ),
    PromptName.CRITIC: ("critic_v1.txt", AnalysisStage.CRITIC, PromptRole.TASK),
    PromptName.BIAS: ("bias_v1.txt", AnalysisStage.BIAS, PromptRole.TASK),
    PromptName.POSTMORTEM: (
        "postmortem_v1.txt",
        AnalysisStage.POSTMORTEM,
        PromptRole.TASK,
    ),
}


def _bundle(task_name: PromptName = PromptName.SUMMARY) -> PromptBundle:
    return PromptBundle(
        system=PromptReference(name=PromptName.SYSTEM, version=PromptVersion.V1),
        task=PromptReference(name=task_name, version=PromptVersion.V1),
    )


def test_every_listed_prompt_is_registered_loadable_and_mapped() -> None:
    registry = PromptRegistry()

    assert len(registry.registrations) == len(EXPECTED_PROMPTS)
    for registration in registry.registrations:
        expected_file, expected_stage, expected_role = EXPECTED_PROMPTS[
            registration.reference.name
        ]
        resolved = registry.resolve(registration.reference)

        assert registration.reference.version is PromptVersion.V1
        assert registration.file_path.name == expected_file
        assert registration.file_path.is_file()
        assert registration.stage is expected_stage
        assert registration.role is expected_role
        assert resolved.reference == registration.reference
        assert resolved.stage is expected_stage
        assert resolved.role is expected_role
        assert resolved.content.strip()


def test_no_unregistered_prompt_files_exist() -> None:
    registry = PromptRegistry()
    registered_paths = {
        registration.file_path for registration in registry.registrations
    }
    prompt_directory = registry.registrations[0].file_path.parent
    prompt_paths = set(prompt_directory.glob("*.txt"))

    assert prompt_paths == registered_paths


def test_resolved_bundle_is_immutable_and_retains_traceability() -> None:
    registry = PromptRegistry()

    resolved = registry.resolve_bundle(_bundle(), AnalysisStage.SUMMARY)

    assert resolved.system.reference.name is PromptName.SYSTEM
    assert resolved.system.reference.version is PromptVersion.V1
    assert resolved.task.reference.name is PromptName.SUMMARY
    assert resolved.task.reference.version is PromptVersion.V1
    assert resolved.task.stage is AnalysisStage.SUMMARY
    with pytest.raises(FrozenInstanceError):
        setattr(resolved.task, "content", "replacement")


def test_content_resolver_returns_validated_prompt_content() -> None:
    registry = PromptRegistry()
    reference = PromptReference(
        name=PromptName.SUMMARY,
        version=PromptVersion.V1,
    )
    resolver: PromptResolver = registry.resolve_content

    assert resolver(reference) == registry.resolve(reference).content


def test_bundle_validator_is_directly_compatible_with_provider_boundary() -> None:
    registry = PromptRegistry()
    validator: PromptBundleValidator = registry.validate_bundle

    validator(_bundle(), AnalysisStage.SUMMARY)


def test_content_resolver_rejects_unknown_prompt_safely() -> None:
    registry = PromptRegistry()
    reference = PromptReference.model_construct(
        name="sensitive-unknown-prompt",
        version=PromptVersion.V1,
    )

    with pytest.raises(PromptRegistryError, match="not registered") as error_info:
        registry.resolve_content(reference)

    error = error_info.value
    assert error.details.category is AIFailureCategory.UNKNOWN_PROMPT
    assert "sensitive-unknown-prompt" not in str(error)
    assert "sensitive-unknown-prompt" not in repr(error)


def test_registry_rejects_task_prompt_for_wrong_analysis_stage() -> None:
    registry = PromptRegistry()

    with pytest.raises(PromptRegistryError, match="does not match") as error_info:
        registry.resolve_bundle(_bundle(), AnalysisStage.TIMELINE)

    assert error_info.value.details.category is AIFailureCategory.UNKNOWN_PROMPT


def test_registry_rejects_reversed_prompt_roles() -> None:
    registry = PromptRegistry()
    reversed_bundle = PromptBundle.model_construct(
        system=PromptReference(name=PromptName.SUMMARY, version=PromptVersion.V1),
        task=PromptReference(name=PromptName.SYSTEM, version=PromptVersion.V1),
    )

    with pytest.raises(PromptRegistryError, match="system prompt"):
        registry.resolve_bundle(reversed_bundle, AnalysisStage.SUMMARY)


@pytest.mark.parametrize(
    "reference",
    [
        PromptReference.model_construct(name="unknown", version=PromptVersion.V1),
        PromptReference.model_construct(name=PromptName.SUMMARY, version="v2"),
    ],
)
def test_registry_rejects_unknown_prompt_name_or_version(
    reference: PromptReference,
) -> None:
    registry = PromptRegistry()

    with pytest.raises(PromptRegistryError, match="not registered") as error_info:
        registry.resolve(reference)

    assert error_info.value.details.category is AIFailureCategory.UNKNOWN_PROMPT


@pytest.mark.parametrize("failure_mode", ["missing", "blank"])
def test_registry_rejects_missing_or_blank_prompt_content(
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
) -> None:
    original_read_text = Path.read_text

    def controlled_read_text(
        path: Path,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> str:
        if path.name == "summary_v1.txt":
            if failure_mode == "missing":
                raise FileNotFoundError
            return "   \n"
        return original_read_text(path, encoding=encoding, errors=errors)

    monkeypatch.setattr(Path, "read_text", controlled_read_text)

    with pytest.raises(PromptRegistryError) as error_info:
        PromptRegistry()

    error = error_info.value
    assert error.details.category is AIFailureCategory.CONFIGURATION
    assert "summary_v1.txt" not in str(error)
    assert "summary_v1.txt" not in repr(error)


def test_every_prompt_enforces_shared_evidence_and_safety_rules() -> None:
    registry = PromptRegistry()
    required_phrases = (
        "redacted evidence",
        "do not invent",
        "facts, assumptions, hypotheses, and recommended actions",
        "evidence identifier",
        "source range",
        "uncertainty",
        "confirmed root cause",
        "structured json",
        "markdown fences",
        "api keys",
        "reconstruct",
        "insufficient evidence",
    )

    for registration in registry.registrations:
        content = registry.resolve(registration.reference).content.lower()
        assert all(phrase in content for phrase in required_phrases), registration


def test_stage_prompts_use_their_declared_uncertainty_fields() -> None:
    registry = PromptRegistry()

    summary = registry.resolve_content(
        PromptReference(name=PromptName.SUMMARY, version=PromptVersion.V1)
    ).lower()
    timeline = registry.resolve_content(
        PromptReference(name=PromptName.TIMELINE, version=PromptVersion.V1)
    ).lower()
    hypotheses = registry.resolve_content(
        PromptReference(name=PromptName.HYPOTHESES, version=PromptVersion.V1)
    ).lower()
    bias = registry.resolve_content(
        PromptReference(name=PromptName.BIAS, version=PromptVersion.V1)
    ).lower()

    assert "summary.uncertainty" in summary
    assert "unknowns" in summary
    assert "dedicated contradicting-evidence field" in summary
    assert "uncertainty_explanation" in timeline
    assert "dedicated contradiction field" in timeline
    assert all(
        field in hypotheses
        for field in (
            "confidence",
            "missing_evidence",
            "contradicting_evidence",
            "validation_test",
        )
    )
    assert "additional uncertainty field" in hypotheses
    assert "confidence" in bias
    assert "additional uncertainty field" in bias


def test_application_code_does_not_load_prompt_files_outside_registry() -> None:
    forbidden_terms = (
        "system_v1.txt",
        "summary_v1.txt",
        "timeline_v1.txt",
        "hypotheses_v1.txt",
        "critic_v1.txt",
        "bias_v1.txt",
        "postmortem_v1.txt",
        "app/prompts",
    )

    for python_path in Path("app").rglob("*.py"):
        if python_path.name == "prompt_registry.py":
            continue
        source = python_path.read_text(encoding="utf-8")
        assert all(term not in source for term in forbidden_terms), python_path
