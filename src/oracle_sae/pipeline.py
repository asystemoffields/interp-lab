from __future__ import annotations

from dataclasses import replace

from oracle_sae.agent_actions import add_inspection_agent_actions
from oracle_sae.adapters.base import FeatureProvider, InterventionRunner, Verbalizer
from oracle_sae.criteria import CriterionCompiler, HeuristicCriterionCompiler
from oracle_sae.fingerprints import build_fingerprint
from oracle_sae.matching import match_feature_cards
from oracle_sae.schema import FeatureCard, FeatureEvidence, InspectionReport, MatchReport
from oracle_sae.scoring import score_feature


def inspect_model(
    *,
    model: str,
    criterion_text: str,
    feature_provider: FeatureProvider,
    verbalizer: Verbalizer,
    intervention_runner: InterventionRunner,
    compiler: CriterionCompiler | None = None,
    top_k: int = 8,
) -> InspectionReport:
    compiler = compiler or HeuristicCriterionCompiler()
    criterion = compiler.compile(criterion_text)
    evidence_items = feature_provider.features_for(model, criterion)
    cards: list[FeatureCard] = []
    for evidence in evidence_items:
        if not _should_keep_evidence(evidence, intervention_runner, criterion):
            continue
        causal_effects = intervention_runner.estimate(evidence, criterion)
        evidence = _with_causal_effects(evidence, causal_effects)
        evidence = _with_intervention_metadata(evidence, intervention_runner, criterion)
        explanation = verbalizer.explain(evidence, criterion)
        scores = score_feature(evidence, criterion)
        fingerprint = build_fingerprint(evidence, criterion, explanation)
        metadata = dict(evidence.metadata)
        metadata.update(_verbalizer_card_metadata(verbalizer, evidence, criterion))
        cards.append(
            FeatureCard(
                feature_id=evidence.feature_id,
                model=evidence.model,
                layer=evidence.layer,
                label=evidence.label,
                explanation=explanation,
                importance=scores["importance"],
                association=scores["association"],
                specificity=scores["specificity"],
                causal_effect=scores["causal_effect"],
                stability=scores["stability"],
                examples=evidence.examples,
                source=evidence.source,
                fingerprint=fingerprint,
                metadata=metadata,
                causal_effects=evidence.causal_effects,
            )
        )
    cards.sort(key=lambda card: card.importance, reverse=True)
    metadata = {"feature_count": len(evidence_items), "kept_feature_count": len(cards)}
    metadata.update(_provider_report_metadata(feature_provider))
    metadata.update(_verbalizer_report_metadata(verbalizer))
    metadata.update(_runner_report_metadata(intervention_runner))
    report = InspectionReport(
        model=model,
        criterion=criterion,
        cards=cards[:top_k],
        metadata=metadata,
    )
    return add_inspection_agent_actions(report)


def match_reports(left: InspectionReport, right: InspectionReport, *, top_k: int = 10) -> MatchReport:
    return MatchReport(
        left_model=left.model,
        right_model=right.model,
        matches=match_feature_cards(left.cards, right.cards, top_k=top_k),
    )


def _with_causal_effects(evidence: FeatureEvidence, causal_effects: dict[str, float]) -> FeatureEvidence:
    merged = dict(evidence.causal_effects)
    merged.update(causal_effects)
    return replace(evidence, causal_effects=merged)


def _with_intervention_metadata(
    evidence: FeatureEvidence,
    intervention_runner: InterventionRunner,
    criterion,
) -> FeatureEvidence:
    metadata_for = getattr(intervention_runner, "metadata_for", None)
    if metadata_for is None:
        return evidence
    extra_metadata = metadata_for(evidence, criterion)
    if not extra_metadata:
        return evidence
    merged = dict(evidence.metadata)
    merged.update(extra_metadata)
    return replace(evidence, metadata=merged)


def _should_keep_evidence(
    evidence: FeatureEvidence,
    intervention_runner: InterventionRunner,
    criterion,
) -> bool:
    should_keep = getattr(intervention_runner, "should_keep", None)
    if should_keep is None:
        return True
    return bool(should_keep(evidence, criterion))


def _provider_report_metadata(feature_provider: FeatureProvider) -> dict:
    metadata_for_report = getattr(feature_provider, "report_metadata", None)
    if metadata_for_report is None:
        return {}
    metadata = metadata_for_report()
    return dict(metadata) if metadata else {}


def _runner_report_metadata(intervention_runner: InterventionRunner) -> dict:
    metadata_for_report = getattr(intervention_runner, "report_metadata", None)
    if metadata_for_report is None:
        return {}
    metadata = metadata_for_report()
    return dict(metadata) if metadata else {}


def _verbalizer_card_metadata(verbalizer: Verbalizer, evidence: FeatureEvidence, criterion) -> dict:
    metadata_for = getattr(verbalizer, "metadata_for", None)
    if metadata_for is None:
        return {}
    metadata = metadata_for(evidence, criterion)
    return dict(metadata) if metadata else {}


def _verbalizer_report_metadata(verbalizer: Verbalizer) -> dict:
    metadata_for_report = getattr(verbalizer, "report_metadata", None)
    if metadata_for_report is None:
        return {}
    metadata = metadata_for_report()
    return dict(metadata) if metadata else {}
