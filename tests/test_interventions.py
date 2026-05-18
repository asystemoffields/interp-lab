import json
from pathlib import Path

from oracle_sae.adapters.interventions import InterventionRecordRunner, summarize_intervention_file
from oracle_sae.criteria import HeuristicCriterionCompiler
from oracle_sae.schema import FeatureEvidence


def test_intervention_runner_aggregates_causal_effects(tmp_path: Path):
    path = tmp_path / "interventions.jsonl"
    rows = [
        {
            "model": "m",
            "feature_id": "L1:F1",
            "criterion": "evaluation awareness",
            "intervention": "ablate",
            "prompt_id": "p1",
            "baseline_score": 0.9,
            "intervention_score": 0.3,
            "side_effect_score": 0.05,
        },
        {
            "model": "m",
            "feature_id": "L1:F1",
            "criterion": "evaluation awareness",
            "intervention": "amplify",
            "prompt_id": "p2",
            "baseline_score": 0.4,
            "intervention_score": 0.7,
            "side_effect_score": 0.03,
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    criterion = HeuristicCriterionCompiler().compile("evaluation awareness")
    evidence = FeatureEvidence(
        feature_id="L1:F1",
        model="m",
        layer=1,
        label="evaluation awareness",
    )

    runner = InterventionRecordRunner(path)
    effects = runner.estimate(evidence, criterion)
    metadata = runner.metadata_for(evidence, criterion)

    assert effects["criterion"] == 0.45
    assert effects["specificity"] == 0.41
    assert effects["side_effect"] == 0.04
    assert metadata["interventions"]["count"] == 2
    assert metadata["interventions"]["examples"][0].startswith("p1: ablate")


def test_intervention_runner_filters_criterion_by_default(tmp_path: Path):
    path = tmp_path / "interventions.jsonl"
    rows = [
        {
            "model": "m",
            "feature_id": "L1:F1",
            "criterion": "other criterion",
            "intervention": "ablate",
            "baseline_score": 1,
            "intervention_score": 0,
        }
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    evidence = FeatureEvidence(feature_id="L1:F1", model="m", layer=1, label="feature")

    runner = InterventionRecordRunner(path)
    effects = runner.estimate(evidence, HeuristicCriterionCompiler().compile("evaluation awareness"))

    assert effects == {}


def test_intervention_runner_can_zero_untested_features(tmp_path: Path):
    path = tmp_path / "interventions.jsonl"
    path.write_text(
        json.dumps(
            {
                "model": "m",
                "feature_id": "other",
                "criterion": "criterion",
                "intervention": "ablate",
                "baseline_score": 1,
                "intervention_score": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    evidence = FeatureEvidence(
        feature_id="L1:F1",
        model="m",
        layer=1,
        label="feature",
        causal_effects={"criterion": 0.9, "specificity": 0.8},
    )

    runner = InterventionRecordRunner(path, require_records=True)
    effects = runner.estimate(evidence, HeuristicCriterionCompiler().compile("criterion"))

    assert effects["criterion"] == 0.0
    assert effects["intervention_record_count"] == 0.0
    assert runner.should_keep(evidence, HeuristicCriterionCompiler().compile("criterion")) is False


def test_summarize_intervention_file_counts_types(tmp_path: Path):
    path = tmp_path / "interventions.jsonl"
    rows = [
        {
            "model": "m",
            "feature_id": "f",
            "intervention": "ablate",
            "baseline_score": 1,
            "intervention_score": 0,
        },
        {
            "model": "m",
            "feature_id": "f",
            "intervention": "amplify",
            "baseline_score": 0,
            "intervention_score": 1,
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    assert summarize_intervention_file(path) == {"ablate": 1, "amplify": 1}
