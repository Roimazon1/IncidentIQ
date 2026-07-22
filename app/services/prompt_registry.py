"""Centralized loading and validation for versioned AI prompts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from app.schemas.ai_provider import (
    AIFailureCategory,
    AIFailureDetails,
    AnalysisStage,
    PromptBundle,
    PromptName,
    PromptReference,
    PromptVersion,
)


class PromptRole(StrEnum):
    """Allowlisted roles for registered prompt content."""

    SYSTEM = "system"
    TASK = "task"


@dataclass(frozen=True, slots=True)
class PromptRegistration:
    """Immutable registry-owned metadata for one versioned prompt file."""

    reference: PromptReference
    stage: AnalysisStage
    role: PromptRole
    file_path: Path


@dataclass(frozen=True, slots=True)
class ResolvedPrompt:
    """Immutable validated prompt content and traceability metadata."""

    reference: PromptReference
    stage: AnalysisStage
    role: PromptRole
    content: str


@dataclass(frozen=True, slots=True)
class ResolvedPromptBundle:
    """One resolved system prompt and one stage-matched task prompt."""

    system: ResolvedPrompt
    task: ResolvedPrompt


class PromptRegistryError(ValueError):
    """Safe provider-neutral failure raised by prompt registration or lookup."""

    def __init__(
        self,
        category: AIFailureCategory,
        explanation: str,
    ) -> None:
        self.details = AIFailureDetails(
            category=category,
            explanation=explanation,
        )
        super().__init__(self.details.explanation)

    def __repr__(self) -> str:
        """Exclude file paths and prompt content from the representation."""
        return (
            f"{type(self).__name__}("
            f"category={self.details.category.value!r}, "
            f"explanation={self.details.explanation!r})"
        )


@dataclass(frozen=True, slots=True)
class _RegistrationDefinition:
    name: PromptName
    version: PromptVersion
    file_name: str
    stage: AnalysisStage
    role: PromptRole


_REGISTRATION_DEFINITIONS = (
    _RegistrationDefinition(
        PromptName.SYSTEM,
        PromptVersion.V1,
        "system_v1.txt",
        AnalysisStage.SYSTEM,
        PromptRole.SYSTEM,
    ),
    _RegistrationDefinition(
        PromptName.SUMMARY,
        PromptVersion.V1,
        "summary_v1.txt",
        AnalysisStage.SUMMARY,
        PromptRole.TASK,
    ),
    _RegistrationDefinition(
        PromptName.TIMELINE,
        PromptVersion.V1,
        "timeline_v1.txt",
        AnalysisStage.TIMELINE,
        PromptRole.TASK,
    ),
    _RegistrationDefinition(
        PromptName.HYPOTHESES,
        PromptVersion.V1,
        "hypotheses_v1.txt",
        AnalysisStage.HYPOTHESES,
        PromptRole.TASK,
    ),
    _RegistrationDefinition(
        PromptName.CRITIC,
        PromptVersion.V1,
        "critic_v1.txt",
        AnalysisStage.CRITIC,
        PromptRole.TASK,
    ),
    _RegistrationDefinition(
        PromptName.BIAS,
        PromptVersion.V1,
        "bias_v1.txt",
        AnalysisStage.BIAS,
        PromptRole.TASK,
    ),
    _RegistrationDefinition(
        PromptName.POSTMORTEM,
        PromptVersion.V1,
        "postmortem_v1.txt",
        AnalysisStage.POSTMORTEM,
        PromptRole.TASK,
    ),
)


class PromptRegistry:
    """Own all prompt paths, load their content, and enforce bundle mappings."""

    def __init__(self, prompt_directory: Path | None = None) -> None:
        directory = prompt_directory or Path(__file__).resolve().parents[1] / "prompts"
        self._prompt_directory = directory.resolve()
        registrations = tuple(
            self._build_registration(definition)
            for definition in _REGISTRATION_DEFINITIONS
        )
        self._registrations = registrations
        self._resolved_prompts = {
            self._reference_key(registration.reference): self._load(registration)
            for registration in registrations
        }

    @property
    def registrations(self) -> tuple[PromptRegistration, ...]:
        """Return immutable metadata for every allowlisted prompt."""
        return self._registrations

    def resolve(self, reference: PromptReference) -> ResolvedPrompt:
        """Resolve one allowlisted prompt reference without accepting a path."""
        prompt = self._resolved_prompts.get(self._reference_key(reference))
        if prompt is None:
            raise PromptRegistryError(
                AIFailureCategory.UNKNOWN_PROMPT,
                "The requested AI prompt is not registered.",
            )
        return prompt

    def resolve_content(self, reference: PromptReference) -> str:
        """Return validated content for an allowlisted prompt reference."""
        return self.resolve(reference).content

    def resolve_bundle(
        self,
        bundle: PromptBundle,
        analysis_stage: AnalysisStage,
    ) -> ResolvedPromptBundle:
        """Resolve and validate system/task roles and the requested task stage."""
        system = self.resolve(bundle.system)
        task = self.resolve(bundle.task)
        if (
            system.role is not PromptRole.SYSTEM
            or system.stage is not AnalysisStage.SYSTEM
        ):
            raise PromptRegistryError(
                AIFailureCategory.UNKNOWN_PROMPT,
                "The requested system prompt has an invalid role or stage.",
            )
        if task.role is not PromptRole.TASK:
            raise PromptRegistryError(
                AIFailureCategory.UNKNOWN_PROMPT,
                "The requested task prompt has an invalid role.",
            )
        if task.stage is not analysis_stage:
            raise PromptRegistryError(
                AIFailureCategory.UNKNOWN_PROMPT,
                "The requested task prompt does not match the analysis stage.",
            )
        return ResolvedPromptBundle(system=system, task=task)

    def _build_registration(
        self,
        definition: _RegistrationDefinition,
    ) -> PromptRegistration:
        file_path = (self._prompt_directory / definition.file_name).resolve()
        if file_path.parent != self._prompt_directory:
            raise PromptRegistryError(
                AIFailureCategory.CONFIGURATION,
                "The registered AI prompt location is invalid.",
            )
        return PromptRegistration(
            reference=PromptReference(
                name=definition.name,
                version=definition.version,
            ),
            stage=definition.stage,
            role=definition.role,
            file_path=file_path,
        )

    @staticmethod
    def _load(registration: PromptRegistration) -> ResolvedPrompt:
        try:
            content = registration.file_path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError):
            raise PromptRegistryError(
                AIFailureCategory.CONFIGURATION,
                "Registered AI prompt content is unavailable.",
            ) from None
        if not content:
            raise PromptRegistryError(
                AIFailureCategory.CONFIGURATION,
                "Registered AI prompt content must not be blank.",
            )
        return ResolvedPrompt(
            reference=registration.reference,
            stage=registration.stage,
            role=registration.role,
            content=content,
        )

    @staticmethod
    def _reference_key(reference: PromptReference) -> tuple[object, object]:
        return reference.name, reference.version
