"""Deterministic source identifiers for traceable open questions."""

from app.models import ClaimSupportStatus
from app.schemas.ai_outputs import OpenQuestionSourceKind
from app.schemas.ai_provider import (
    OpenQuestionsContextV1,
    OpenQuestionSourceOptionV1,
)


class OpenQuestionSourceService:
    """Build the exact source allowlist shared by providers and validation."""

    @classmethod
    def build_source_options(
        cls,
        context: OpenQuestionsContextV1,
    ) -> tuple[OpenQuestionSourceOptionV1, ...]:
        """Assign stable short identifiers to unresolved typed analysis sources."""
        analysis = context.analysis_context
        candidates = (
            *(
                (OpenQuestionSourceKind.UNRESOLVED_CLAIM, fact.claim)
                for fact in analysis.validated_analysis.facts
                if fact.support_status is not ClaimSupportStatus.SUPPORTED
            ),
            *(
                (OpenQuestionSourceKind.UNRESOLVED_CLAIM, finding.affected_claim)
                for finding in analysis.critic.findings
            ),
            *(
                (OpenQuestionSourceKind.HYPOTHESIS, hypothesis.hypothesis_id)
                for hypothesis in analysis.validated_analysis.hypotheses
            ),
            *(
                (
                    OpenQuestionSourceKind.CONTRADICTION,
                    cls.build_contradiction_source_reference(
                        hypothesis.hypothesis_id,
                        evidence.reference.reference.evidence_id,
                        evidence.reference.reference.line_range,
                    ),
                )
                for hypothesis in analysis.validated_analysis.hypotheses
                for evidence in hypothesis.contradicting_evidence
            ),
            *(
                (OpenQuestionSourceKind.ASSUMPTION, assumption.claim)
                for assumption in analysis.original_analysis.summary.assumptions
            ),
            *(
                (OpenQuestionSourceKind.MISSING_EVIDENCE, missing_evidence)
                for hypothesis in analysis.validated_analysis.hypotheses
                for missing_evidence in hypothesis.missing_evidence
            ),
        )
        unique_candidates = tuple(dict.fromkeys(candidates))
        return tuple(
            OpenQuestionSourceOptionV1(
                source_id=f"S-{index:03d}",
                source_kind=source_kind,
                source_reference=source_reference,
            )
            for index, (source_kind, source_reference) in enumerate(
                unique_candidates,
                start=1,
            )
        )

    @staticmethod
    def build_contradiction_source_reference(
        hypothesis_id: str,
        evidence_id: str,
        line_range: str,
    ) -> str:
        """Return the exact stable reference for one typed contradiction."""
        return f"{hypothesis_id}|{evidence_id}|{line_range}"
