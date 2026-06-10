import json
from pathlib import Path

import pytest

from interp_lab.adapters.interventions import InterventionRecordRunner, summarize_intervention_file
from interp_lab.criteria import HeuristicCriterionCompiler
from interp_lab.feature_interventions import intervene_on_features
from interp_lab.hf_interventions import export_hf_intervention_records, validate_hookable_feature_layers
from interp_lab.reporting import _intervention_lines, _sae_training_lines
from interp_lab.schema import FeatureEvidence


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


def test_intervention_metadata_gives_auto_specific_near_zero_advice(tmp_path: Path):
    path = tmp_path / "interventions.jsonl"
    rows = [
        {
            "model": "m",
            "feature_id": "L1:F1",
            "criterion": "criterion",
            "intervention": "steer",
            "baseline_score": 0.002,
            "intervention_score": 0.003,
            "metadata": {
                "behavior_score": "target_token_probability_mass",
                "target_token_strategy": "auto",
                "target_tokens": [" meters", " feet"],
            },
        },
        {
            "model": "m",
            "feature_id": "L1:F1",
            "criterion": "criterion",
            "intervention": "steer",
            "baseline_score": 0.004,
            "intervention_score": 0.005,
            "metadata": {
                "behavior_score": "target_token_probability_mass",
                "target_token_strategy": "auto",
                "target_tokens": [" meters", " feet"],
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
    assert "even with auto-derived targets" in behavior_score["advisory"]
    assert "use auto targets" not in behavior_score["advisory"]
    markdown = "\n".join(_intervention_lines(metadata["interventions"]))
    assert "sample=` meters`, ` feet`" in markdown


def test_validate_hookable_feature_layers_rejects_layer_zero():
    with pytest.raises(ValueError, match=r"L0:D1, L0:D7"):
        validate_hookable_feature_layers([(0, 1, "L0:D1"), (3, 2, "L3:D2"), (0, 7, "L0:D7")])

    validate_hookable_feature_layers([(1, 0, "L1:D0"), (12, 5, "L12:D5")])


def test_export_hf_interventions_rejects_layer_zero_before_model_load(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("interp_lab.hf_interventions._optional_import", lambda name, message: object())

    def _fail_load(**kwargs):
        raise AssertionError("load_hf_text_model must not be called for unhookable features")

    monkeypatch.setattr("interp_lab.hf_interventions.load_hf_text_model", _fail_load)
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "model": "m",
                "criterion": {"text": "criterion"},
                "cards": [_hidden_card("L0:D1"), _hidden_card("L2:D3")],
            }
        ),
        encoding="utf-8",
    )
    dataset = tmp_path / "prompts.jsonl"
    dataset.write_text(
        json.dumps({"prompt_id": "pos", "text": "a", "criterion_score": 1.0})
        + "\n"
        + json.dumps({"prompt_id": "neg", "text": "b", "criterion_score": 0.0})
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="L0:D1"):
        export_hf_intervention_records(
            model_name="m",
            report_path=report,
            dataset_path=dataset,
            out_path=tmp_path / "interventions.jsonl",
            criterion="criterion",
        )
    assert not (tmp_path / "interventions.jsonl").exists()


def test_intervene_on_features_rejects_layer_zero_before_model_load(tmp_path: Path):
    dataset = tmp_path / "prompts.jsonl"
    dataset.write_text(
        json.dumps({"prompt_id": "pos", "text": "a", "criterion_score": 1.0})
        + "\n"
        + json.dumps({"prompt_id": "neg", "text": "b", "criterion_score": 0.0})
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="L0:D3"):
        intervene_on_features(
            model_name="m",
            dataset_path=dataset,
            criterion="criterion",
            out_path=tmp_path / "interventions.jsonl",
            features=["L0:D3"],
            mode="ablate",
        )
    assert not (tmp_path / "interventions.jsonl").exists()

    # Dry runs still produce a plan (with the existing layer-0 advisory) so
    # agents can inspect why the live run would fail.
    result = intervene_on_features(
        model_name="m",
        dataset_path=dataset,
        criterion="criterion",
        out_path=tmp_path / "interventions.jsonl",
        features=["L0:D3"],
        mode="ablate",
        dry_run=True,
    )
    assert result.dry_run is True
    assert any("Layer 0" in advisory for advisory in result.plan["advisories"])


def _hidden_card(feature_id: str) -> dict:
    layer = int(feature_id.split(":")[0][1:])
    return {
        "feature_id": feature_id,
        "model": "m",
        "layer": layer,
        "label": "hidden dimension",
        "explanation": "",
        "importance": 1,
        "association": 1,
        "specificity": 1,
        "causal_effect": 1,
        "stability": 1,
        "examples": [],
        "source": "hf-hidden-state",
        "fingerprint": {
            "feature_id": feature_id,
            "model": "m",
            "layer": layer,
            "text": "",
            "text_vector": [],
            "activation_signature": [],
            "decoder_signature": [],
            "causal_vector": [],
        },
        "metadata": {},
        "causal_effects": {},
    }


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


def test_group_activation_record_labels_member_sign_provenance(tmp_path: Path):
    # Member signs are heuristic seeds for orienting the group aggregate: an
    # association-derived sign is acceptable for ordering the sum, but the output
    # artifact must label each member's provenance so downstream readers never
    # mistake a correlational sign for a measured causal direction.
    from interp_lab.hf_interventions import append_hf_group_activation_record

    def _report_card(feature_id: str, causal_effects: dict) -> dict:
        return {
            "feature_id": feature_id,
            "model": "m",
            "layer": 1,
            "label": feature_id,
            "explanation": "",
            "importance": 1,
            "association": 1,
            "specificity": 1,
            "causal_effect": 1,
            "stability": 1,
            "examples": [],
            "source": "hf-hidden-state",
            "fingerprint": {
                "feature_id": feature_id,
                "model": "m",
                "layer": 1,
                "text": "",
                "text_vector": [],
                "activation_signature": [],
                "decoder_signature": [],
                "causal_vector": [],
            },
            "metadata": {},
            "causal_effects": causal_effects,
        }

    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "model": "m",
                "criterion": {"text": "criterion"},
                "cards": [
                    _report_card("L1:D1", {"signed_causal_effect": 1}),
                    _report_card("L1:D2", {"signed_association": -1}),
                    _report_card("L1:D3", {}),
                ],
            }
        ),
        encoding="utf-8",
    )
    records = tmp_path / "records.jsonl"
    records.write_text(
        json.dumps(
            {
                "model": "m",
                "prompt_id": "p",
                "text": "text",
                "criterion_score": 1,
                "features": [
                    {"feature_id": "L1:D1", "activation": 2},
                    {"feature_id": "L1:D2", "activation": -4},
                    {"feature_id": "L1:D3", "activation": 1},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "grouped.jsonl"

    group_id = append_hf_group_activation_record(
        records_path=records,
        report_path=report,
        out_path=out,
        group_top_k=3,
    )

    row = json.loads(out.read_text(encoding="utf-8"))
    group_metadata = row["feature_metadata"][group_id]
    assert group_metadata["member_signs"] == {"L1:D1": 1.0, "L1:D2": -1.0, "L1:D3": 1.0}
    assert group_metadata["member_sign_provenance"] == {
        "L1:D1": "intervention",
        "L1:D2": "association",
        "L1:D3": "none",
    }
    # Orientation behavior is unchanged: (2*1 + (-4)*(-1) + 1*1) / 3.
    assert row["features"][-1]["activation"] == pytest.approx(7 / 3)
