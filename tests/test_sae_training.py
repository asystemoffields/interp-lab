import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from oracle_sae.adapters.records import ActivationRecordFeatureProvider
from oracle_sae.criteria import HeuristicCriterionCompiler
from oracle_sae.sae_training import (
    _split_prompt_indexes,
    _training_settings,
    build_train_sae_parser,
    encode_with_artifact,
    load_activation_matrix_from_records,
    run_train_sae_from_args,
    train_sae,
    train_sae_from_records,
)


def test_train_sae_from_records_writes_artifact_and_activation_records(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    rows = [
        _row("m", "pos-1", 1.0, {"raw-a": 2.0, "raw-b": 0.0, "raw-c": 0.1}),
        _row("m", "pos-2", 1.0, {"raw-a": 1.7, "raw-b": 0.1, "raw-c": 0.0}),
        _row("m", "neg-1", 0.0, {"raw-a": 0.0, "raw-b": 2.0, "raw-c": 0.1}),
        _row("m", "neg-2", 0.0, {"raw-a": 0.1, "raw-b": 1.8, "raw-c": 0.0}),
    ]
    source.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    artifact_path = tmp_path / "sae.json"
    records_out = tmp_path / "sae_records.jsonl"

    train_sae_from_records(
        records_path=source,
        out_path=artifact_path,
        records_out=records_out,
        model_name="m",
        latent_dim=4,
        method="fallback",
        top_k_features=2,
    )

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["format"] == "interp-lab.sae.v1"
    assert artifact["method"] == "fallback-dictionary"
    assert artifact["input_dim"] == 3
    assert artifact["latent_dim"] == 4

    generated_rows = [
        json.loads(line)
        for line in records_out.read_text(encoding="utf-8").splitlines()
    ]
    assert generated_rows[0]["model"] == "m"
    assert len(generated_rows[0]["features"]) == 2
    assert generated_rows[0]["features"][0]["feature_id"].startswith("SAE:")

    evidence = ActivationRecordFeatureProvider(records_out).features_for(
        "m",
        HeuristicCriterionCompiler().compile("positive raw-a pattern"),
    )
    assert evidence
    assert evidence[0].source == "trained-sae"


def test_encode_with_artifact_uses_relu_codes(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps(_row("m", "p", 1.0, {"raw-a": 2.0, "raw-b": 0.0})) + "\n",
        encoding="utf-8",
    )
    matrix = load_activation_matrix_from_records(source, model_name="m")
    artifact = {
        "mean": [1.0, 0.0],
        "encoder_weight": [[1.0, 0.0], [-1.0, 0.0]],
        "encoder_bias": [0.0, 0.0],
    }

    assert encode_with_artifact(matrix.values, artifact) == [[1.0, 0.0]]


def test_encode_with_artifact_applies_topk_sparsity():
    artifact = {
        "mean": [0.0, 0.0, 0.0],
        "encoder_weight": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        "encoder_bias": [0.0, 0.0, 0.0],
        "config": {"sparsity": "topk", "top_k": 1},
    }

    assert encode_with_artifact([[1.0, 3.0, 2.0]], artifact) == [[0.0, 3.0, 0.0]]


def test_encode_with_artifact_applies_jumprelu_sparsity():
    artifact = {
        "mean": [0.0, 0.0],
        "encoder_weight": [[1.0, 0.0], [0.0, 1.0]],
        "encoder_bias": [0.0, 0.0],
        "config": {"sparsity": "jumprelu", "jump_threshold": 1.5},
    }

    assert encode_with_artifact([[1.0, 2.0]], artifact) == [[0.0, 2.0]]


def test_train_sae_reports_validation_and_dead_latents(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    rows = [
        _row("m", "p1", 1.0, {"raw-a": 2.0, "raw-b": 0.0}),
        _row("m", "p2", 1.0, {"raw-a": 1.5, "raw-b": 0.0}),
        _row("m", "p3", 0.0, {"raw-a": 0.0, "raw-b": 1.5}),
        _row("m", "p4", 0.0, {"raw-a": 0.0, "raw-b": 2.0}),
    ]
    source.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    matrix = load_activation_matrix_from_records(source, model_name="m")

    artifact = train_sae(
        matrix,
        latent_dim=4,
        method="fallback",
        validation_fraction=0.25,
        dead_latent_threshold=1.0,
    )

    assert artifact["metrics"]["validation_reconstruction_mse"] is not None
    assert artifact["metrics"]["dead_latent_count"] == 4
    assert artifact["metrics"]["active_latent_fraction"] == 0.0


def test_training_presets_can_be_selected_and_overridden():
    parser = build_train_sae_parser()

    production = parser.parse_args(["--preset", "production", "--hf-model", "m", "--dataset", "d", "--out", "o"])
    production_settings = _training_settings(production)
    assert production_settings["token_mode"] == "all"
    assert production_settings["sparsity"] == "topk"

    minimal = parser.parse_args(["--preset", "minimal", "--hf-model", "m", "--dataset", "d", "--out", "o"])
    assert _training_settings(minimal)["epochs"] == 50

    custom = parser.parse_args(
        ["--preset", "production", "--hf-model", "m", "--dataset", "d", "--out", "o", "--epochs", "7"]
    )
    assert _training_settings(custom)["epochs"] == 7


def test_fallback_training_honors_topk_sparsity(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    rows = [
        _row("m", "p1", 1.0, {"raw-a": 3.0, "raw-b": 2.0, "raw-c": 1.0}),
        _row("m", "p2", 0.0, {"raw-a": 0.0, "raw-b": 2.5, "raw-c": 4.0}),
    ]
    source.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    matrix = load_activation_matrix_from_records(source, model_name="m")

    artifact = train_sae(
        matrix,
        latent_dim=3,
        method="fallback",
        sparsity="topk",
        top_k=1,
        l1_coefficient=0.0,
    )

    assert artifact["config"]["sparsity"] == "topk"
    assert artifact["config"]["top_k"] == 1
    encoded = encode_with_artifact(matrix.values, artifact)
    assert all(sum(1 for value in row if value > 1e-6) <= 1 for row in encoded)


def test_split_prompt_indexes_handles_duplicate_prompt_ids():
    prompts = [
        SimpleNamespace(prompt_id="", criterion_score=1.0),
        SimpleNamespace(prompt_id="", criterion_score=0.0),
    ]

    assert _split_prompt_indexes(prompts) == ({0}, {1})


def test_training_records_mode_rejects_causal_out(tmp_path: Path):
    parser = build_train_sae_parser()
    args = parser.parse_args(
        [
            "--records",
            str(tmp_path / "records.jsonl"),
            "--out",
            str(tmp_path / "sae.json"),
            "--causal-out",
            str(tmp_path / "interventions.jsonl"),
        ]
    )

    with pytest.raises(SystemExit, match="--causal-out"):
        run_train_sae_from_args(args)


def _row(model: str, prompt_id: str, score: float, features: dict[str, float]):
    return {
        "model": model,
        "prompt_id": prompt_id,
        "text": prompt_id,
        "criterion_score": score,
        "features": features,
    }
