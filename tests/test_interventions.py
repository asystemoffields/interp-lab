import json
from pathlib import Path

from oracle_sae.adapters.interventions import InterventionRecordRunner, summarize_intervention_file
from oracle_sae.criteria import HeuristicCriterionCompiler
from oracle_sae.reporting import _intervention_lines, _sae_training_lines
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
    assert effects["strong_causal_score"] == 0.41
    assert effects["criterion_ci_low"] < effects["criterion_ci_high"]


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


def test_intervention_runner_tracks_control_records(tmp_path: Path):
    path = tmp_path / "interventions.jsonl"
    rows = [
        {
            "model": "m",
            "feature_id": "L1:F1",
            "criterion": "criterion",
            "intervention": "ablate",
            "baseline_score": 0.8,
            "intervention_score": 0.2,
            "side_effect_score": 0.05,
        },
        {
            "model": "m",
            "feature_id": "L1:F1",
            "criterion": "criterion",
            "intervention": "ablate",
            "baseline_score": 0.5,
            "intervention_score": 0.4,
            "metadata": {"control_type": "random_feature"},
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    evidence = FeatureEvidence(feature_id="L1:F1", model="m", layer=1, label="feature")

    runner = InterventionRecordRunner(path)
    effects = runner.estimate(evidence, HeuristicCriterionCompiler().compile("criterion"))
    metadata = runner.metadata_for(evidence, HeuristicCriterionCompiler().compile("criterion"))

    assert effects["criterion"] == 0.6
    assert effects["control_record_count"] == 1.0
    assert effects["control_mean_abs_effect"] == 0.1
    assert effects["strong_causal_score"] == 0.45
    assert metadata["interventions"]["controls"]["by_type"]["random_feature"]["count"] == 1


def test_intervention_metadata_flags_saturated_behavior_scores(tmp_path: Path):
    path = tmp_path / "interventions.jsonl"
    rows = [
        {
            "model": "m",
            "feature_id": "L1:F1",
            "criterion": "criterion",
            "intervention": "steer",
            "baseline_score": 0.93,
            "intervention_score": 0.94,
            "metadata": {
                "behavior_score": "target_token_probability_mass",
                "target_token_strategy": "auto",
                "target_tokens": [" and", "<turn|>"],
            },
        },
        {
            "model": "m",
            "feature_id": "L1:F1",
            "criterion": "criterion",
            "intervention": "steer",
            "baseline_score": 0.88,
            "intervention_score": 0.89,
            "metadata": {
                "behavior_score": "target_token_probability_mass",
                "target_token_strategy": "auto",
                "target_tokens": [" and", "<turn|>"],
            },
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    evidence = FeatureEvidence(feature_id="L1:F1", model="m", layer=1, label="feature")

    metadata = InterventionRecordRunner(path).metadata_for(
        evidence,
        HeuristicCriterionCompiler().compile("criterion"),
    )

    behavior_score = metadata["interventions"]["behavior_score"]
    assert behavior_score["diagnostic"] == "saturated_baseline"
    assert behavior_score["target_token_strategy"] == "auto"
    assert behavior_score["target_token_count"] == 2
    markdown = "\n".join(_intervention_lines(metadata["interventions"]))
    assert "Behavior score: target_token_probability_mass baseline mean=0.905" in markdown
    assert "Behavior note:" in markdown


def test_intervention_metadata_flags_near_zero_behavior_scores(tmp_path: Path):
    path = tmp_path / "interventions.jsonl"
    rows = [
        {
            "model": "m",
            "feature_id": "L1:F1",
            "criterion": "criterion",
            "intervention": "steer",
            "baseline_score": 0.002,
            "intervention_score": 0.01,
            "metadata": {
                "behavior_score": "target_token_probability_mass",
                "target_token_strategy": "explicit",
                "target_tokens": [" Python"],
            },
        },
        {
            "model": "m",
            "feature_id": "L1:F1",
            "criterion": "criterion",
            "intervention": "steer",
            "baseline_score": 0.004,
            "intervention_score": 0.012,
            "metadata": {
                "behavior_score": "target_token_probability_mass",
                "target_token_strategy": "explicit",
                "target_tokens": [" Python"],
            },
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    evidence = FeatureEvidence(feature_id="L1:F1", model="m", layer=1, label="feature")

    metadata = InterventionRecordRunner(path).metadata_for(
        evidence,
        HeuristicCriterionCompiler().compile("criterion"),
    )

    behavior_score = metadata["interventions"]["behavior_score"]
    assert behavior_score["diagnostic"] == "near_zero_baseline"
    assert behavior_score["target_token_strategy"] == "explicit"
    assert "raw tokenizer forms" in behavior_score["advisory"]


def test_sae_training_lines_surface_quality_notes():
    lines = _sae_training_lines(
        {
            "sample_count": 64,
            "latent_dim": 256,
            "active_latent_fraction": 0.125,
            "dead_latent_count": 224,
            "validation_reconstruction_mse": 45.094,
            "advisories": ["Training rows are fewer than latents; collect more activations."],
        }
    )

    assert lines[0] == "SAE training: rows=64, latents=256, active=0.125, dead=224, val MSE=45.094"
    assert lines[1].startswith("SAE training note:")
