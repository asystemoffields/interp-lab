"""Calibration harness: measure interp-lab's own honesty against planted ground truth.

The toolkit's brand is honest grading -- correlational and causal evidence are kept
strictly separate, and claims are graded rather than asserted. This module turns
that brand into a measured, publishable number: it generates synthetic worlds with
*planted* causal structure, runs the REAL pipeline (records backend + intervention
records + the real match grader) over them, and reports how well the pipeline's
verdicts track the planted truth.

How a planted world works
=========================

``generate_planted_world(seed)`` builds, deterministically from ``random.Random(seed)``:

- ``n_causal`` **truly causal** features: their activations correlate with the
  criterion AND their planted intervention records carry real signed directed
  effects (drawn from ``effect_range``) with controls (random_feature /
  matched_frequency / placebo) showing ~zero effect.
- some **correlational decoys**: activations correlate just as strongly, but their
  planted intervention records show ~zero directed effect (and matching controls).
- the rest are **noise**: activations independent of the criterion and no
  intervention records at all.

The world is emitted as real artifacts -- an activation-records JSONL and an
intervention-records JSONL in the exact formats the records backend
(``adapters.records.ActivationRecordFeatureProvider``) and the intervention loader
(``adapters.interventions.InterventionRecordRunner``) parse. ``PlantedWorld.write``
round-trips both files through those real loaders before returning, so a schema
drift in either format fails loudly here. The artifacts are *blind*: ground truth
(kind, planted effect) lives only on the ``PlantedWorld`` object, never in the
files the pipeline sees.

What gets graded, and by which real machinery
=============================================

``run_calibration`` runs ``interp_lab.api.inspect`` (records backend +
interventions -- the exact CLI code path) per seed and grades each planted feature
on two real axes:

1. **Evidence tier** -- derived from the pipeline's own outputs on each card:
   ``matching.has_intervention_provenance`` (the provenance gate scoring.py uses),
   the intervention runner's Student-t CI on the mean directed effect, and its
   control-adjusted ``strong_causal_score``. A card is ``measured_causal`` only if
   it is intervention-backed, its CI excludes zero, its control-adjusted causal
   score is positive, and |signed effect| >= ``min_abs_effect``. Intervention-backed
   cards that fail that are ``intervention_null``; cards with no intervention
   records are ``correlational_only``.
2. **Self-match claim grade** -- the report is matched against itself
   (``pipeline.match_reports``) and graded by the real
   ``match_validation.build_match_validation_report`` grader; the self-pair claim
   grade isolates the evidence-grading axis (provenance + signed-effect thresholds)
   from cross-model matching noise. "When interp-lab says ``validated_equivalent``,
   how often is the feature truly causal?" is answered directly from this table.

Headline metrics: discovery precision/recall@k (k = ``n_causal``) over the
importance ranking, P(truly causal | tier) with Wilson CIs, decoy resistance (the
fraction of correlational decoys that did NOT earn a ``measured_causal`` label --
post-2.3.0 this should be 1.0), and the Spearman correlation between planted and
reported effect sizes among true causals.

Honest caveats (also embedded in every report): synthetic worlds are not real
models. Planted effects are clean, independent, and schema-perfect; the harness
certifies the *grading machinery*, not the toolkit's behavior on messy real
activations.

Pure stdlib; no torch required. Default settings finish in a few seconds on CPU.
"""

from __future__ import annotations

import json
import math
import random
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from interp_lab import stats
from interp_lab.adapters.interventions import summarize_intervention_file
from interp_lab.adapters.records import ActivationRecordFeatureProvider
from interp_lab.match_validation import build_match_validation_report
from interp_lab.matching import has_intervention_provenance
from interp_lab.pipeline import match_reports
from interp_lab.schema import Criterion, FeatureCard, InspectionReport

CALIBRATION_SCHEMA = "interp-lab.calibration_report.v1"

KIND_CAUSAL = "causal"
KIND_DECOY = "decoy"
KIND_NOISE = "noise"

TIER_MEASURED_CAUSAL = "measured_causal"
TIER_INTERVENTION_NULL = "intervention_null"
TIER_CORRELATIONAL_ONLY = "correlational_only"
TIER_MISSING = "missing_from_report"

VERDICT_WELL_CALIBRATED = "well_calibrated"
VERDICT_OVERCLAIMS = "overclaims_causality"
VERDICT_UNDERPOWERED = "underpowered_or_misranked"
VERDICT_NO_CAUSAL_TRUTH = "no_causal_ground_truth"

DEFAULT_CRITERION = "the model exhibits the planted synthetic behavior"
DEFAULT_MIN_ABS_EFFECT = 0.05
DEFAULT_SEEDS = range(5)

# Noise scales for the planted intervention measurements. The decoy scale is kept
# well below DEFAULT_MIN_ABS_EFFECT so "decoy resistance == 1.0" is a property of
# the grading machinery, not a coin flip on measurement noise.
_CAUSAL_MEASUREMENT_SD = 0.02
_DECOY_MEASUREMENT_SD = 0.01
_CONTROL_MEASUREMENT_SD = 0.015

CAVEATS = [
    "Synthetic worlds are not real models: planted effects are clean, independent, "
    "and match the record schemas exactly; real SAE/probe features are messier and "
    "their interventions noisier.",
    "Only the records backend + intervention-record grading path is exercised; "
    "HF/SAELens/Neuronpedia backends and live-model interventions are not covered.",
    "P(truly causal | tier) is conditional on this planted generative process and "
    "does not transfer to arbitrary datasets; treat it as a check that the grading "
    "machinery cannot be fooled by clean correlational decoys, not as a field error rate.",
    "Self-match claim grades compare a report against itself, deliberately isolating "
    "the evidence-grading axis from cross-model matching noise.",
    "The match grader's causal axis is magnitude-sensitive, so truly causal features "
    "with small planted effects grade needs_more_evidence rather than "
    "validated_equivalent; P(truly causal | validated_equivalent) is the precision "
    "anchor, the evidence-tier table is the recall-sensitive axis.",
]


# ---------------------------------------------------------------------------
# Planted worlds
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlantedFeature:
    """Ground truth for one planted feature (never written into the artifacts)."""

    feature_id: str
    kind: str  # KIND_CAUSAL | KIND_DECOY | KIND_NOISE
    layer: int
    label: str
    planted_effect: float  # 0.0 for decoys and noise features
    intervention: str  # "ablate"/"amplify" for intervention-tested kinds, "" for noise


@dataclass(frozen=True)
class PlantedWorld:
    """A synthetic world plus the real JSONL artifacts that describe it."""

    seed: int
    model: str
    criterion: str
    n_prompts: int
    noise: float
    features: tuple[PlantedFeature, ...]
    activation_records: tuple[dict[str, Any], ...]
    intervention_records: tuple[dict[str, Any], ...]
    config: dict[str, Any] = field(default_factory=dict)

    def feature_ids(self, kind: str) -> set[str]:
        return {feature.feature_id for feature in self.features if feature.kind == kind}

    @property
    def causal_ids(self) -> set[str]:
        return self.feature_ids(KIND_CAUSAL)

    @property
    def decoy_ids(self) -> set[str]:
        return self.feature_ids(KIND_DECOY)

    @property
    def noise_ids(self) -> set[str]:
        return self.feature_ids(KIND_NOISE)

    def write(self, out_dir: str | Path, *, validate: bool = True) -> tuple[Path, Path]:
        """Write the activation/intervention JSONL artifacts; round-trip-check them.

        With ``validate=True`` (default) both files are re-loaded through the REAL
        loaders (``ActivationRecordFeatureProvider`` and the intervention-record
        loader), so the harness can never silently drift from the formats the
        pipeline actually parses.
        """
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        activation_path = out / "activation_records.jsonl"
        interventions_path = out / "interventions.jsonl"
        activation_path.write_text(_jsonl(self.activation_records), encoding="utf-8")
        interventions_path.write_text(_jsonl(self.intervention_records), encoding="utf-8")
        if validate:
            self._validate_artifacts(activation_path, interventions_path)
        return activation_path, interventions_path

    def _validate_artifacts(self, activation_path: Path, interventions_path: Path) -> None:
        provider = ActivationRecordFeatureProvider(activation_path)
        evidence = provider.features_for(self.model, Criterion(text=self.criterion))
        if len(evidence) != len(self.features):
            raise ValueError(
                f"planted world artifact drift: wrote {len(self.features)} features but the "
                f"records loader recovered {len(evidence)} from {activation_path}"
            )
        counts = summarize_intervention_file(interventions_path)
        loaded = sum(counts.values())
        if loaded != len(self.intervention_records):
            raise ValueError(
                f"planted world artifact drift: wrote {len(self.intervention_records)} "
                f"intervention records but the loader recovered {loaded} from {interventions_path}"
            )


def generate_planted_world(
    seed: int,
    *,
    n_features: int = 24,
    n_causal: int = 6,
    n_prompts: int = 64,
    noise: float = 0.3,
    effect_range: tuple[float, float] = (0.2, 0.8),
    n_decoys: int | None = None,
    intervention_repeats: int = 6,
    controls_per_type: int = 2,
    criterion: str = DEFAULT_CRITERION,
) -> PlantedWorld:
    """Deterministically build a planted world from ``random.Random(seed)``.

    ``n_causal`` features are truly causal (correlated activations + real planted
    signed intervention effects with controls), ``n_decoys`` (default: as many as
    fit, up to ``n_causal``) are correlational decoys (correlated activations,
    ~zero intervention effects), and the remainder are noise (independent
    activations, no intervention records).
    """
    _validate_world_args(
        n_features=n_features,
        n_causal=n_causal,
        n_prompts=n_prompts,
        noise=noise,
        effect_range=effect_range,
        n_decoys=n_decoys,
        intervention_repeats=intervention_repeats,
        controls_per_type=controls_per_type,
    )
    if n_decoys is None:
        n_decoys = min(n_causal, n_features - n_causal)
    rng = random.Random(seed)
    model = f"planted/world-{seed}"

    features: list[PlantedFeature] = []
    for index in range(n_features):
        if index < n_causal:
            kind = KIND_CAUSAL
            planted_effect = round(rng.uniform(*effect_range), 6)
            intervention = rng.choice(("ablate", "amplify"))
        elif index < n_causal + n_decoys:
            kind = KIND_DECOY
            planted_effect = 0.0
            intervention = rng.choice(("ablate", "amplify"))
        else:
            kind = KIND_NOISE
            planted_effect = 0.0
            intervention = ""
        layer = 4 + index % 12
        features.append(
            PlantedFeature(
                feature_id=f"L{layer}:F{100 + index}",
                kind=kind,
                layer=layer,
                # Blind label: ground truth must never leak into the artifacts.
                label=f"planted latent {index:02d}",
                planted_effect=planted_effect,
                intervention=intervention,
            )
        )

    activation_records = _build_activation_records(
        rng, model=model, features=features, n_prompts=n_prompts, noise=noise
    )
    intervention_records = _build_intervention_records(
        rng,
        model=model,
        criterion=criterion,
        features=features,
        intervention_repeats=intervention_repeats,
        controls_per_type=controls_per_type,
    )
    config = {
        "n_features": n_features,
        "n_causal": n_causal,
        "n_decoys": n_decoys,
        "n_noise": n_features - n_causal - n_decoys,
        "n_prompts": n_prompts,
        "noise": noise,
        "effect_range": [effect_range[0], effect_range[1]],
        "intervention_repeats": intervention_repeats,
        "controls_per_type": controls_per_type,
        "criterion": criterion,
    }
    return PlantedWorld(
        seed=seed,
        model=model,
        criterion=criterion,
        n_prompts=n_prompts,
        noise=noise,
        features=tuple(features),
        activation_records=tuple(activation_records),
        intervention_records=tuple(intervention_records),
        config=config,
    )


def _validate_world_args(
    *,
    n_features: int,
    n_causal: int,
    n_prompts: int,
    noise: float,
    effect_range: tuple[float, float],
    n_decoys: int | None,
    intervention_repeats: int,
    controls_per_type: int,
) -> None:
    if n_features < 1:
        raise ValueError("n_features must be >= 1")
    if not 0 <= n_causal <= n_features:
        raise ValueError(f"n_causal must be in [0, n_features={n_features}], got {n_causal}")
    if n_decoys is not None:
        if n_decoys < 0 or n_causal + n_decoys > n_features:
            raise ValueError(
                f"n_decoys must satisfy 0 <= n_decoys <= n_features - n_causal "
                f"({n_features - n_causal}), got {n_decoys}"
            )
    if n_prompts < 4:
        raise ValueError("n_prompts must be >= 4")
    if noise < 0:
        raise ValueError("noise must be >= 0")
    low, high = float(effect_range[0]), float(effect_range[1])
    if not (0.0 < low <= high <= 1.0):
        raise ValueError(f"effect_range must satisfy 0 < low <= high <= 1, got {effect_range!r}")
    if intervention_repeats < 2:
        # n=1 reports an insufficient_n CI; the tier gate needs a real interval.
        raise ValueError("intervention_repeats must be >= 2")
    if controls_per_type < 0:
        raise ValueError("controls_per_type must be >= 0")


def _build_activation_records(
    rng: random.Random,
    *,
    model: str,
    features: list[PlantedFeature],
    n_prompts: int,
    noise: float,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index in range(n_prompts):
        positive = index % 2 == 0
        score = rng.uniform(0.7, 1.0) if positive else rng.uniform(0.0, 0.3)
        prompt_id = f"p{index:03d}"
        text = (
            f"{prompt_id}: synthetic {'positive' if positive else 'negative'} probe "
            "for the planted behavior"
        )
        feature_entries: list[dict[str, Any]] = []
        for feature in features:
            if feature.kind in (KIND_CAUSAL, KIND_DECOY):
                # Correlated with the criterion for BOTH causal and decoy features:
                # the activation file alone cannot distinguish them.
                activation = score + noise * rng.gauss(0.0, 1.0)
            else:
                activation = rng.gauss(0.5, 0.5)
            feature_entries.append(
                {
                    "feature_id": feature.feature_id,
                    "activation": round(activation, 6),
                    "label": feature.label,
                    "layer": feature.layer,
                }
            )
        records.append(
            {
                "model": model,
                "prompt_id": prompt_id,
                "text": text,
                "criterion_score": round(score, 6),
                "features": feature_entries,
            }
        )
    return records


def _build_intervention_records(
    rng: random.Random,
    *,
    model: str,
    criterion: str,
    features: list[PlantedFeature],
    intervention_repeats: int,
    controls_per_type: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for feature in features:
        if feature.kind == KIND_NOISE:
            continue  # noise features get no intervention records at all
        effect = feature.planted_effect
        measurement_sd = _CAUSAL_MEASUREMENT_SD if feature.kind == KIND_CAUSAL else _DECOY_MEASUREMENT_SD
        for repeat in range(intervention_repeats):
            if feature.intervention == "ablate":
                baseline = effect + rng.uniform(0.05, 0.15)
                intervention_score = baseline - effect + rng.gauss(0.0, measurement_sd)
            else:  # amplify
                baseline = rng.uniform(0.05, 0.15)
                intervention_score = baseline + effect + rng.gauss(0.0, measurement_sd)
            records.append(
                {
                    "model": model,
                    "feature_id": feature.feature_id,
                    "criterion": criterion,
                    "intervention": feature.intervention,
                    "prompt_id": f"iv{repeat:03d}",
                    "baseline_score": round(baseline, 6),
                    "intervention_score": round(intervention_score, 6),
                    "side_effect_score": round(abs(rng.gauss(0.0, 0.01)), 6),
                }
            )
        for control_type in ("random_feature", "matched_frequency", "placebo"):
            for repeat in range(controls_per_type):
                baseline = rng.uniform(0.4, 0.7)
                delta = rng.gauss(0.0, _CONTROL_MEASUREMENT_SD)
                if feature.intervention == "ablate":
                    intervention_score = baseline - delta
                else:
                    intervention_score = baseline + delta
                records.append(
                    {
                        "model": model,
                        "feature_id": feature.feature_id,
                        "criterion": criterion,
                        "intervention": feature.intervention,
                        "prompt_id": f"ctl-{control_type}-{repeat:03d}",
                        "baseline_score": round(baseline, 6),
                        "intervention_score": round(intervention_score, 6),
                        "side_effect_score": round(abs(rng.gauss(0.0, 0.01)), 6),
                        "metadata": {"control_type": control_type},
                    }
                )
    return records


def _jsonl(records: Sequence[dict[str, Any]]) -> str:
    if not records:
        return ""
    return "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n"


# ---------------------------------------------------------------------------
# Running the real pipeline and grading against the planted truth
# ---------------------------------------------------------------------------


def run_calibration(
    out_dir: str | Path,
    *,
    seeds: int | Iterable[int] = DEFAULT_SEEDS,
    min_abs_effect: float = DEFAULT_MIN_ABS_EFFECT,
    **world_kwargs: Any,
) -> dict[str, Any]:
    """Generate planted worlds, run the REAL pipeline, and score it against truth.

    Per seed: world artifacts are written under ``out_dir/seed-<seed>/`` and fed to
    ``api.inspect(backend="records", interventions=...)`` -- the exact code path
    the CLI uses. Returns the calibration report dict (timestamp-free, so two runs
    with the same seeds produce identical payloads). ``seeds`` accepts an int N
    (meaning ``range(N)``) or an iterable of ints. Extra keyword arguments are
    forwarded to :func:`generate_planted_world`.
    """
    seed_list = _normalize_seeds(seeds)
    out = Path(out_dir)
    per_seed: list[dict[str, Any]] = []
    config: dict[str, Any] | None = None
    for seed in seed_list:
        world = generate_planted_world(seed, **world_kwargs)
        if config is None:
            config = dict(world.config)
        world_dir_name = f"seed-{seed}"
        activation_path, interventions_path = world.write(out / world_dir_name)
        report = _inspect_world(world, activation_path, interventions_path)
        per_seed.append(
            _evaluate_world(
                world,
                report,
                min_abs_effect=min_abs_effect,
                artifact_paths={
                    # Relative, '/'-joined names: deterministic across machines/out dirs.
                    "activation_records": f"{world_dir_name}/{activation_path.name}",
                    "interventions": f"{world_dir_name}/{interventions_path.name}",
                },
            )
        )
    pooled = _pool_metrics(per_seed)
    assessment = _build_assessment(pooled, config or {}, seed_count=len(seed_list))
    return {
        "schema_version": CALIBRATION_SCHEMA,
        "config": {
            "seeds": seed_list,
            "min_abs_effect": min_abs_effect,
            "world": config or {},
        },
        "per_seed": per_seed,
        "pooled": pooled,
        "assessment": assessment,
    }


def _normalize_seeds(seeds: int | Iterable[int]) -> list[int]:
    if isinstance(seeds, bool):
        raise ValueError("seeds must be an int or an iterable of ints")
    if isinstance(seeds, int):
        if seeds < 1:
            raise ValueError("seeds must be >= 1 when given as a count")
        return list(range(seeds))
    seed_list = [int(seed) for seed in seeds]
    if not seed_list:
        raise ValueError("seeds must be non-empty")
    return seed_list


def _inspect_world(world: PlantedWorld, records_path: Path, interventions_path: Path) -> InspectionReport:
    # Lazy import: after integration, api.py imports this module, so a top-level
    # `from interp_lab import api` here would create an import cycle. Importing it
    # inside the call (long after both modules finished initializing) is safe and
    # keeps this run on the exact code path the CLI uses.
    from interp_lab import api

    report = api.inspect(
        world.model,
        world.criterion,
        backend="records",
        records=records_path,
        interventions=interventions_path,
        top_k=len(world.features),
    )
    if not isinstance(report, InspectionReport):  # pragma: no cover - inspect(out=None) contract
        report = report.report
    return report


def evidence_tier(
    causal_effects: dict[str, Any],
    metadata: dict[str, Any] | None,
    *,
    min_abs_effect: float = DEFAULT_MIN_ABS_EFFECT,
) -> str:
    """Assign the evidence tier the pipeline's own outputs support for one card.

    Built entirely from real machinery outputs: the provenance gate
    (``matching.has_intervention_provenance``, the same gate scoring.py uses to
    zero out correlational ``causal_effect``), the intervention runner's Student-t
    CI on the mean directed effect, and its control-adjusted ``strong_causal_score``.
    """
    if not has_intervention_provenance(causal_effects, metadata):
        return TIER_CORRELATIONAL_ONLY
    signed = float(causal_effects.get("signed_causal_effect", 0.0) or 0.0)
    strong = float(causal_effects.get("strong_causal_score", 0.0) or 0.0)
    ci_low = causal_effects.get("criterion_ci_low")
    ci_high = causal_effects.get("criterion_ci_high")
    ci_excludes_zero = (
        ci_low is not None
        and ci_high is not None
        and (float(ci_low) > 0.0 or float(ci_high) < 0.0)
    )
    if strong > 0.0 and abs(signed) >= min_abs_effect and ci_excludes_zero:
        return TIER_MEASURED_CAUSAL
    return TIER_INTERVENTION_NULL


def _self_match_claim_grades(report: InspectionReport) -> dict[str, str]:
    """Grade each card's evidence with the real match grader via a self-match."""
    cards = report.cards
    if not cards:
        return {}
    matches = match_reports(report, report, top_k=len(cards) * len(cards), min_score=0.0)
    validation = build_match_validation_report(matches)
    grades: dict[str, str] = {}
    for item in validation["validations"]:
        if item["left_feature_id"] == item["right_feature_id"]:
            grades[str(item["left_feature_id"])] = str(item["claim_grade"])
    return grades


def _evaluate_world(
    world: PlantedWorld,
    report: InspectionReport,
    *,
    min_abs_effect: float,
    artifact_paths: dict[str, str],
) -> dict[str, Any]:
    cards_by_id: dict[str, FeatureCard] = {card.feature_id: card for card in report.cards}
    rank_by_id = {card.feature_id: index + 1 for index, card in enumerate(report.cards)}
    grades_by_id = _self_match_claim_grades(report)

    rows: list[dict[str, Any]] = []
    for feature in world.features:
        card = cards_by_id.get(feature.feature_id)
        if card is None:
            rows.append(
                {
                    "feature_id": feature.feature_id,
                    "kind": feature.kind,
                    "planted_effect": feature.planted_effect,
                    "rank": None,
                    "importance": None,
                    "association": None,
                    "causal_effect": None,
                    "strong_causal_score": None,
                    "signed_causal_effect": None,
                    "evidence_tier": TIER_MISSING,
                    "self_match_claim_grade": None,
                }
            )
            continue
        signed = card.causal_effects.get("signed_causal_effect")
        strong = card.causal_effects.get("strong_causal_score")
        rows.append(
            {
                "feature_id": feature.feature_id,
                "kind": feature.kind,
                "planted_effect": feature.planted_effect,
                "rank": rank_by_id[feature.feature_id],
                "importance": round(float(card.importance), 6),
                "association": round(float(card.association), 6),
                "causal_effect": round(float(card.causal_effect), 6),
                "strong_causal_score": round(float(strong), 6) if strong is not None else None,
                "signed_causal_effect": round(float(signed), 6) if signed is not None else None,
                "evidence_tier": evidence_tier(
                    card.causal_effects, card.metadata, min_abs_effect=min_abs_effect
                ),
                "self_match_claim_grade": grades_by_id.get(feature.feature_id),
            }
        )

    metrics = _seed_metrics(world, report, rows)
    return {
        "seed": world.seed,
        "world": {
            "model": world.model,
            "criterion": world.criterion,
            "n_features": len(world.features),
            "n_causal": len(world.causal_ids),
            "n_decoys": len(world.decoy_ids),
            "n_noise": len(world.noise_ids),
            "artifacts": artifact_paths,
        },
        "metrics": metrics,
        "features": rows,
    }


def _seed_metrics(
    world: PlantedWorld, report: InspectionReport, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    causal_ids = world.causal_ids
    k = len(causal_ids)
    top_ids = [card.feature_id for card in report.cards[:k]]
    true_positives = sum(1 for feature_id in top_ids if feature_id in causal_ids)

    decoy_rows = [row for row in rows if row["kind"] == KIND_DECOY]
    decoy_false = [row for row in decoy_rows if row["evidence_tier"] == TIER_MEASURED_CAUSAL]
    causal_rows = [row for row in rows if row["kind"] == KIND_CAUSAL]
    causal_measured = [row for row in causal_rows if row["evidence_tier"] == TIER_MEASURED_CAUSAL]

    spearman = _spearman(
        [row["planted_effect"] for row in causal_rows],
        [abs(row["signed_causal_effect"] or 0.0) for row in causal_rows],
    )
    return {
        "k": k,
        "true_positives_at_k": true_positives,
        "discovery_precision_at_k": _ratio(true_positives, k),
        "discovery_recall_at_k": _ratio(true_positives, k),
        "decoy_count": len(decoy_rows),
        "decoy_false_causal_count": len(decoy_false),
        "decoy_resistance": (
            _ratio(len(decoy_rows) - len(decoy_false), len(decoy_rows)) if decoy_rows else None
        ),
        "causal_count": len(causal_rows),
        "causal_measured_count": len(causal_measured),
        "causal_recovery": _ratio(len(causal_measured), len(causal_rows)),
        "effect_rank_correlation": spearman,
    }


def _pool_metrics(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for seed_result in per_seed for row in seed_result["features"]]
    metrics = [seed_result["metrics"] for seed_result in per_seed]

    total_k = sum(m["k"] for m in metrics)
    total_tp = sum(m["true_positives_at_k"] for m in metrics)
    total_decoys = sum(m["decoy_count"] for m in metrics)
    total_decoy_false = sum(m["decoy_false_causal_count"] for m in metrics)
    total_causal = sum(m["causal_count"] for m in metrics)
    total_causal_measured = sum(m["causal_measured_count"] for m in metrics)

    causal_rows = [row for row in rows if row["kind"] == KIND_CAUSAL]
    pooled_spearman = _spearman(
        [row["planted_effect"] for row in causal_rows],
        [abs(row["signed_causal_effect"] or 0.0) for row in causal_rows],
    )
    return {
        "discovery": {
            "k": metrics[0]["k"] if metrics else 0,
            "seed_count": len(metrics),
            "true_positives_at_k": total_tp,
            "precision_at_k": _ratio(total_tp, total_k),
            "recall_at_k": _ratio(total_tp, total_k),
            "per_seed_precision_at_k": [m["discovery_precision_at_k"] for m in metrics],
        },
        "grade_calibration": {
            "by_evidence_tier": _calibration_table(rows, "evidence_tier"),
            "by_self_match_claim_grade": _calibration_table(rows, "self_match_claim_grade"),
        },
        "decoy_resistance": {
            "decoy_count": total_decoys,
            "false_causal_count": total_decoy_false,
            "decoy_resistance": (
                _ratio(total_decoys - total_decoy_false, total_decoys) if total_decoys else None
            ),
            "false_causal_fraction": (
                _ratio(total_decoy_false, total_decoys) if total_decoys else None
            ),
        },
        "causal_recovery": {
            "causal_count": total_causal,
            "measured_causal_count": total_causal_measured,
            "recovery": _ratio(total_causal_measured, total_causal),
        },
        "effect_rank_correlation": {
            "n": len(causal_rows),
            "spearman": pooled_spearman,
        },
    }


def _calibration_table(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    """For each bucket of ``key``: P(truly causal | bucket) with a Wilson 95% CI."""
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        value = row.get(key)
        if value is None:
            value = "ungraded"
        buckets.setdefault(str(value), []).append(row)
    table: dict[str, dict[str, Any]] = {}
    for value in sorted(buckets):
        bucket_rows = buckets[value]
        truly_causal = sum(1 for row in bucket_rows if row["kind"] == KIND_CAUSAL)
        interval = wilson_interval(truly_causal, len(bucket_rows))
        table[value] = {
            "count": len(bucket_rows),
            "truly_causal_count": truly_causal,
            "p_truly_causal": _ratio(truly_causal, len(bucket_rows)),
            "ci_low": interval["low"],
            "ci_high": interval["high"],
            "ci_method": interval["method"],
        }
    return table


def wilson_interval(successes: int, total: int, *, confidence: float = 0.95) -> dict[str, Any]:
    """Wilson score interval for a binomial proportion (stdlib-only)."""
    if total <= 0:
        return {"low": None, "high": None, "method": "no_data", "confidence": confidence, "n": 0}
    if not 0 <= successes <= total:
        raise ValueError(f"successes must be in [0, total={total}], got {successes}")
    z = stats._inverse_normal_cdf(1.0 - (1.0 - confidence) / 2.0)
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half_width = (z / denominator) * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
    return {
        "low": round(max(0.0, center - half_width), 6),
        "high": round(min(1.0, center + half_width), 6),
        "method": "wilson",
        "confidence": confidence,
        "n": total,
    }


def _spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    rx = _ranks(x)
    ry = _ranks(y)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    cx = [value - mx for value in rx]
    cy = [value - my for value in ry]
    denominator = math.sqrt(sum(v * v for v in cx)) * math.sqrt(sum(v * v for v in cy))
    if denominator == 0.0:
        return None
    return round(sum(a * b for a, b in zip(cx, cy)) / denominator, 6)


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        tied_end = position
        while (
            tied_end + 1 < len(order)
            and values[order[tied_end + 1]] == values[order[position]]
        ):
            tied_end += 1
        average_rank = (position + tied_end) / 2.0 + 1.0
        for index in range(position, tied_end + 1):
            ranks[order[index]] = average_rank
        position = tied_end + 1
    return ranks


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 6)


def _build_assessment(
    pooled: dict[str, Any], world_config: dict[str, Any], *, seed_count: int
) -> dict[str, Any]:
    tier_table = pooled["grade_calibration"]["by_evidence_tier"]
    grade_table = pooled["grade_calibration"]["by_self_match_claim_grade"]
    measured = tier_table.get(TIER_MEASURED_CAUSAL, {})
    validated = grade_table.get("validated_equivalent", {})
    precision = pooled["discovery"]["precision_at_k"]
    recovery = pooled["causal_recovery"]["recovery"]
    decoy_resistance = pooled["decoy_resistance"]["decoy_resistance"]
    measured_count = int(measured.get("count", 0))
    measured_false = measured_count - int(measured.get("truly_causal_count", 0))
    planted_causal_total = pooled["causal_recovery"]["causal_count"]

    headline = {
        "discovery_precision_at_k": precision,
        "discovery_recall_at_k": pooled["discovery"]["recall_at_k"],
        "decoy_resistance": decoy_resistance,
        "p_truly_causal_given_measured_causal": measured.get("p_truly_causal"),
        "p_truly_causal_given_validated_match": validated.get("p_truly_causal"),
        "effect_rank_correlation": pooled["effect_rank_correlation"]["spearman"],
    }

    if planted_causal_total == 0:
        if measured_count > 0:
            verdict = VERDICT_OVERCLAIMS
            summary = (
                f"No causal features were planted across {seed_count} seed(s), but the pipeline "
                f"labeled {measured_count} feature(s) measured-causal. This is a calibration failure."
            )
        else:
            verdict = VERDICT_NO_CAUSAL_TRUTH
            summary = (
                f"No causal features were planted across {seed_count} seed(s), and the pipeline "
                "honestly reported no measured-causal evidence. Discovery precision/recall are "
                "undefined for a world with no causal ground truth."
            )
    elif measured_false > 0 or (decoy_resistance is not None and decoy_resistance < 1.0):
        verdict = VERDICT_OVERCLAIMS
        summary = (
            f"{measured_false} non-causal feature(s) earned a measured-causal label "
            f"(decoy resistance {decoy_resistance}). The grading machinery overclaimed "
            "causality on planted ground truth."
        )
    elif (precision or 0.0) >= 0.9 and (recovery or 0.0) >= 0.9:
        verdict = VERDICT_WELL_CALIBRATED
        summary = (
            f"Across {seed_count} seed(s): precision@k={precision}, causal recovery={recovery}, "
            f"decoy resistance={decoy_resistance}. No correlational feature earned causal-labeled "
            "evidence; planted causal features were discovered and graded as measured-causal."
        )
    else:
        verdict = VERDICT_UNDERPOWERED
        summary = (
            f"No false causal labels, but discovery or recovery fell short "
            f"(precision@k={precision}, causal recovery={recovery}). The pipeline is honest "
            "but underpowered or misranked on this world configuration."
        )
    return {
        "verdict": verdict,
        "summary": summary,
        "headline": headline,
        "caveats": list(CAVEATS),
    }


# ---------------------------------------------------------------------------
# Export: the publishable trust-anchor artifact
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationWriteResult:
    report: dict[str, Any]
    json_path: Path
    markdown_path: Path


def export_calibration_report(
    out: str | Path,
    markdown_out: str | Path | None = None,
    *,
    report: dict[str, Any] | None = None,
    work_dir: str | Path | None = None,
    seeds: int | Iterable[int] = DEFAULT_SEEDS,
    min_abs_effect: float = DEFAULT_MIN_ABS_EFFECT,
    created_at: str | None = None,
    **world_kwargs: Any,
) -> CalibrationWriteResult:
    """Write the calibration report as JSON plus readable Markdown.

    Pass a pre-built ``report`` (from :func:`run_calibration`) to just render it,
    or omit it to run a fresh calibration (world artifacts go to ``work_dir``, or a
    temporary directory that is cleaned up afterwards). ``created_at`` is
    injectable so callers/tests can produce byte-identical artifacts; the metric
    payload itself never contains timestamps.
    """
    if report is None:
        if work_dir is not None:
            report = run_calibration(
                work_dir, seeds=seeds, min_abs_effect=min_abs_effect, **world_kwargs
            )
        else:
            with tempfile.TemporaryDirectory(prefix="interp-lab-calibration-") as tmp:
                report = run_calibration(
                    tmp, seeds=seeds, min_abs_effect=min_abs_effect, **world_kwargs
                )
    payload = dict(report)
    payload["created_at"] = created_at or datetime.now(timezone.utc).isoformat()

    json_path = Path(out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path = Path(markdown_out) if markdown_out is not None else json_path.with_suffix(".md")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_calibration_markdown(payload), encoding="utf-8")
    return CalibrationWriteResult(report=payload, json_path=json_path, markdown_path=markdown_path)


def render_calibration_markdown(report: dict[str, Any]) -> str:
    config = report.get("config", {})
    world = config.get("world", {})
    pooled = report.get("pooled", {})
    assessment = report.get("assessment", {})
    headline = assessment.get("headline", {})
    lines = [
        "# interp-lab Calibration Report",
        "",
        "Synthetic planted-ground-truth audit of interp-lab's own claim grading: ",
        "planted causal features, correlational decoys, and noise features were run ",
        "through the real records + interventions pipeline and graded by the real ",
        "evidence machinery.",
        "",
        f"Schema: `{report.get('schema_version', CALIBRATION_SCHEMA)}`",
        f"Verdict: `{assessment.get('verdict', '')}`",
        "",
        f"{assessment.get('summary', '')}",
        "",
        "## Configuration",
        "",
        f"- Seeds: `{config.get('seeds', [])}`",
        f"- Features per world: `{world.get('n_features', '')}` "
        f"(causal `{world.get('n_causal', '')}`, decoys `{world.get('n_decoys', '')}`, "
        f"noise `{world.get('n_noise', '')}`)",
        f"- Prompts per world: `{world.get('n_prompts', '')}`, activation noise `{world.get('noise', '')}`",
        f"- Planted effect range: `{world.get('effect_range', '')}`, "
        f"intervention repeats `{world.get('intervention_repeats', '')}`, "
        f"controls per type `{world.get('controls_per_type', '')}`",
        f"- Minimum |signed effect| for a measured-causal tier: `{config.get('min_abs_effect', '')}`",
        "",
        "## Headline",
        "",
        f"- Discovery precision@k: `{_md_number(headline.get('discovery_precision_at_k'))}`",
        f"- Discovery recall@k: `{_md_number(headline.get('discovery_recall_at_k'))}`",
        f"- Decoy resistance: `{_md_number(headline.get('decoy_resistance'))}`",
        "- P(truly causal | measured_causal tier): "
        f"`{_md_number(headline.get('p_truly_causal_given_measured_causal'))}`",
        "- P(truly causal | validated_equivalent self-match): "
        f"`{_md_number(headline.get('p_truly_causal_given_validated_match'))}`",
        "- Planted-vs-reported effect rank correlation (Spearman): "
        f"`{_md_number(headline.get('effect_rank_correlation'))}`",
        "",
        "## Grade calibration: evidence tiers",
        "",
    ]
    lines.extend(
        _calibration_table_markdown(
            pooled.get("grade_calibration", {}).get("by_evidence_tier", {}), "Evidence tier"
        )
    )
    lines.extend(["", "## Grade calibration: self-match claim grades", ""])
    lines.extend(
        _calibration_table_markdown(
            pooled.get("grade_calibration", {}).get("by_self_match_claim_grade", {}),
            "Claim grade",
        )
    )
    lines.extend(
        [
            "",
            "## Per-seed metrics",
            "",
            "| Seed | Precision@k | Recall@k | Decoy resistance | Causal recovery | Spearman |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for seed_result in report.get("per_seed", []):
        metrics = seed_result.get("metrics", {})
        lines.append(
            f"| {seed_result.get('seed', '')} "
            f"| {_md_number(metrics.get('discovery_precision_at_k'))} "
            f"| {_md_number(metrics.get('discovery_recall_at_k'))} "
            f"| {_md_number(metrics.get('decoy_resistance'))} "
            f"| {_md_number(metrics.get('causal_recovery'))} "
            f"| {_md_number(metrics.get('effect_rank_correlation'))} |"
        )
    lines.extend(["", "## Caveats", ""])
    for caveat in assessment.get("caveats", []):
        lines.append(f"- {caveat}")
    lines.append("")
    return "\n".join(lines)


def _calibration_table_markdown(table: dict[str, dict[str, Any]], label: str) -> list[str]:
    if not table:
        return ["(no graded features)"]
    lines = [
        f"| {label} | n | Truly causal | P(causal) | 95% CI |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for value, cell in table.items():
        ci = (
            f"[{_md_number(cell.get('ci_low'))}, {_md_number(cell.get('ci_high'))}]"
            if cell.get("ci_low") is not None
            else "n/a"
        )
        lines.append(
            f"| `{value}` | {cell.get('count', 0)} | {cell.get('truly_causal_count', 0)} "
            f"| {_md_number(cell.get('p_truly_causal'))} | {ci} |"
        )
    return lines


def _md_number(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"
