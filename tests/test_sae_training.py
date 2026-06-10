import json
from inspect import signature
from pathlib import Path
from types import SimpleNamespace

import pytest
from interp_lab.adapters.records import ActivationRecordFeatureProvider
from interp_lab.criteria import HeuristicCriterionCompiler
from interp_lab.sae_training import (
    _sae_training_summary,
    _split_prompt_indexes,
    _select_latents,
    _training_settings,
    build_train_sae_parser,
    encode_with_artifact,
    load_activation_matrix_from_hf,
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
    first_feature_id = generated_rows[0]["features"][0]["feature_id"]
    assert generated_rows[0]["feature_metadata"][first_feature_id]["sae_training"]["sample_count"] == 4

    evidence = ActivationRecordFeatureProvider(records_out).features_for(
        "m",
        HeuristicCriterionCompiler().compile("positive raw-a pattern"),
    )
    assert evidence
    assert "sae_training" in evidence[0].metadata
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


def test_encode_with_artifact_rejects_shape_mismatch():
    artifact = {
        "input_dim": 3,
        "latent_dim": 2,
        "mean": [0.0, 0.0, 0.0],
        "encoder_weight": [[1.0, 0.0], [0.0, 1.0, 0.0]],
        "encoder_bias": [0.0, 0.0],
        "decoder_weight": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    }

    with pytest.raises(ValueError, match=r"encoder_weight\[0\] length"):
        encode_with_artifact([[1.0, 2.0, 3.0]], artifact)


def test_encode_with_artifact_rejects_activation_dimension_mismatch():
    artifact = {
        "input_dim": 2,
        "latent_dim": 1,
        "mean": [0.0, 0.0],
        "encoder_weight": [[1.0, 0.0]],
        "encoder_bias": [0.0],
        "decoder_weight": [[1.0, 0.0]],
    }

    with pytest.raises(ValueError, match="activation row 0 length 3"):
        encode_with_artifact([[1.0, 2.0, 3.0]], artifact)


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


def test_sae_training_summary_warns_on_sparse_rows_and_validation_drift():
    artifact = {
        "method": "torch",
        "latent_dim": 256,
        "config": {"sparsity": "topk", "top_k": 16},
        "metrics": {
            "active_latent_fraction": 0.9,
            "dead_latent_count": 25,
            "average_l0": 16,
            "train_reconstruction_mse": 1.0,
            "validation_reconstruction_mse": 3.0,
        },
    }

    summary = _sae_training_summary(artifact, sample_count=512)

    assert summary["rows_per_latent"] == 2.0
    assert summary["validation_train_mse_ratio"] == 3.0
    assert any("Training rows are sparse" in advisory for advisory in summary["advisories"])
    assert any("broaden training prompts" in advisory for advisory in summary["advisories"])


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


def test_fallback_artifact_reproduces_metric_latents_with_nonzero_l1(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    rows = [
        _row("m", "p1", 1.0, {"raw-a": 3.0, "raw-b": 0.5}),
        _row("m", "p2", 1.0, {"raw-a": 2.0, "raw-b": 0.0}),
        _row("m", "p3", 0.0, {"raw-a": 0.0, "raw-b": 2.5}),
        _row("m", "p4", 0.0, {"raw-a": 0.5, "raw-b": 3.0}),
    ]
    source.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    matrix = load_activation_matrix_from_records(source, model_name="m")

    artifact = train_sae(
        matrix,
        latent_dim=2,
        method="fallback",
        l1_coefficient=0.5,
        validation_fraction=0.0,
    )

    # The l1 soft shrinkage must live in the stored encoder bias so
    # encode_with_artifact reproduces the latents the metrics describe.
    assert artifact["encoder_bias"] == [-0.5, -0.5]
    encoded = encode_with_artifact(matrix.values, artifact)
    average_l0 = sum(sum(1 for value in row if value > 1e-6) for row in encoded) / len(encoded)
    firing_rates = [
        sum(1.0 for row in encoded if row[index] > 1e-6) / len(encoded)
        for index in range(2)
    ]
    latent_means = [sum(row[index] for row in encoded) / len(encoded) for index in range(2)]
    assert artifact["metrics"]["average_l0"] == round(average_l0, 6)
    assert artifact["metrics"]["latent_firing_rate"] == [round(rate, 8) for rate in firing_rates]
    assert artifact["metrics"]["latent_activation_mean"] == [round(value, 8) for value in latent_means]


def test_train_sae_auto_falls_back_with_advisory_when_torch_missing(tmp_path: Path, monkeypatch, capsys):
    source = tmp_path / "source.jsonl"
    rows = [
        _row("m", "p1", 1.0, {"raw-a": 2.0, "raw-b": 0.0}),
        _row("m", "p2", 0.0, {"raw-a": 0.0, "raw-b": 2.0}),
    ]
    source.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    matrix = load_activation_matrix_from_records(source, model_name="m")
    monkeypatch.setattr("interp_lab.sae_training._torch_is_available", lambda: False)

    artifact = train_sae(matrix, latent_dim=2, method="auto")

    assert artifact["method"] == "fallback-dictionary"
    assert "fallback dictionary trainer" in capsys.readouterr().err


def test_train_sae_torch_method_fails_fast_without_torch(tmp_path: Path, monkeypatch):
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(_row("m", "p", 1.0, {"raw-a": 1.0})) + "\n", encoding="utf-8")
    matrix = load_activation_matrix_from_records(source, model_name="m")
    monkeypatch.setattr("interp_lab.sae_training._torch_is_available", lambda: False)

    with pytest.raises(RuntimeError, match="PyTorch is required"):
        train_sae(matrix, latent_dim=1, method="torch")


def test_train_sae_auto_surfaces_import_errors_from_broken_torch(tmp_path: Path, monkeypatch):
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(_row("m", "p", 1.0, {"raw-a": 1.0})) + "\n", encoding="utf-8")
    matrix = load_activation_matrix_from_records(source, model_name="m")
    monkeypatch.setattr("interp_lab.sae_training._torch_is_available", lambda: True)

    def _broken_torch_training(*args, **kwargs):
        raise ImportError("torch is installed but broken")

    monkeypatch.setattr("interp_lab.sae_training._train_torch_sae", _broken_torch_training)

    with pytest.raises(ImportError, match="installed but broken"):
        train_sae(matrix, latent_dim=1, method="auto")


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


def test_hf_activation_loader_accepts_hf_loading_options():
    parameters = signature(load_activation_matrix_from_hf).parameters

    for name in [
        "model_class",
        "trust_remote_code",
        "local_files_only",
        "torch_dtype",
        "device_map",
        "model_kwargs",
        "tokenizer_kwargs",
    ]:
        assert name in parameters


def test_training_parser_accepts_separate_causal_dataset():
    parser = build_train_sae_parser()

    args = parser.parse_args(
        [
            "--hf-model",
            "m",
            "--dataset",
            "train.jsonl",
            "--causal-dataset",
            "eval.jsonl",
            "--out",
            "sae.json",
            "--causal-out",
            "interventions.jsonl",
            "--criterion",
            "criterion",
        ]
    )

    assert args.dataset == "train.jsonl"
    assert args.causal_dataset == "eval.jsonl"


def test_select_latents_keeps_forced_causal_latents():
    selected = _select_latents(
        [0.9, 0.8, 0.1, 0.0],
        2,
        force_latent_indexes={3},
    )

    assert selected == [(0, 0.9), (1, 0.8), (3, 0.0)]


def _row(model: str, prompt_id: str, score: float, features: dict[str, float]):
    return {
        "model": model,
        "prompt_id": prompt_id,
        "text": prompt_id,
        "criterion_score": score,
        "features": features,
    }
