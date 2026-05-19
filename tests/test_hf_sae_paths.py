from pathlib import Path

import pytest

from oracle_sae.hf_sae_paths import (
    build_hf_sae_paths_parser,
    parse_path_pair,
    parse_path_pairs,
    parse_sae_feature_ref,
    _random_source_control_refs,
    resolve_sae_feature_refs,
)
from oracle_sae.reporting import write_inspection_report
from oracle_sae.schema import Criterion, FeatureCard, FeatureFingerprint, InspectionReport


def test_parse_sae_feature_ref_validates_layer_and_latent_bounds():
    artifact = _artifact(layer=12, latent_dim=3)

    ref = parse_sae_feature_ref("SAE:L12:F2", artifact=artifact, role="source")

    assert ref.feature_id == "SAE:L12:F2"
    assert ref.layer == 12
    assert ref.latent_index == 2
    with pytest.raises(ValueError, match="layer 13"):
        parse_sae_feature_ref("SAE:L13:F2", artifact=artifact, role="source")
    with pytest.raises(ValueError, match="outside latent_dim"):
        parse_sae_feature_ref("SAE:L12:F3", artifact=artifact, role="source")


def test_resolve_sae_feature_refs_uses_report_then_explicit_features(tmp_path: Path):
    artifact = _artifact(layer=24, latent_dim=8)
    report = InspectionReport(
        model="m",
        criterion=Criterion(text="criterion"),
        cards=[
            _card("SAE:L24:F3", 24, "reported target"),
            _card("SAE:L24:F5", 24, "second target"),
        ],
    )
    report_path, _ = write_inspection_report(report, tmp_path / "report")

    refs = resolve_sae_feature_refs(
        explicit_features=["SAE:L24:F6"],
        report_path=report_path,
        artifact=artifact,
        top_k=1,
        role="target",
    )

    assert [ref.feature_id for ref in refs] == ["SAE:L24:F3", "SAE:L24:F6"]
    assert refs[0].label == "reported target"


def test_hf_sae_paths_parser_accepts_core_options():
    parser = build_hf_sae_paths_parser()

    args = parser.parse_args(
        [
            "--model",
            "m",
            "--dataset",
            "prompts.jsonl",
            "--criterion",
            "criterion",
            "--source-sae",
            "source.json",
            "--target-sae",
            "target.json",
            "--out",
            "paths.jsonl",
            "--source-feature",
            "SAE:L12:F1",
            "--target-feature",
            "SAE:L24:F8",
            "--strength-sweep=-2,2",
            "--random-source-controls",
            "2",
            "--control-seed",
            "17",
            "--path-pair",
            "SAE:L12:F1=SAE:L24:F8",
        ]
    )

    assert args.source_feature == ["SAE:L12:F1"]
    assert args.target_feature == ["SAE:L24:F8"]
    assert args.random_source_controls == 2
    assert args.control_seed == 17
    assert parse_path_pairs(args.path_pair) == [("SAE:L12:F1", "SAE:L24:F8")]


def test_parse_path_pair_accepts_common_separators():
    assert parse_path_pair("SAE:L1:F2=SAE:L2:F3") == ("SAE:L1:F2", "SAE:L2:F3")
    assert parse_path_pair("SAE:L1:F2, SAE:L2:F3") == ("SAE:L1:F2", "SAE:L2:F3")
    assert parse_path_pair("SAE:L1:F2->SAE:L2:F3") == ("SAE:L1:F2", "SAE:L2:F3")
    with pytest.raises(ValueError, match="Path pairs"):
        parse_path_pair("SAE:L1:F2")


def test_random_source_controls_are_deterministic_and_exclude_source_latent():
    artifact = _artifact(layer=12, latent_dim=5)
    source_ref = parse_sae_feature_ref("SAE:L12:F2", artifact=artifact, role="source")

    left = _random_source_control_refs(source_ref, artifact=artifact, count=4, seed=11)
    right = _random_source_control_refs(source_ref, artifact=artifact, count=4, seed=11)

    assert [ref.latent_index for ref in left] == [ref.latent_index for ref in right]
    assert all(ref.latent_index != 2 for ref in left)
    assert all(ref.feature_id.startswith("SAE:L12:F") for ref in left)


def _artifact(*, layer: int, latent_dim: int) -> dict:
    return {
        "format": "interp-lab.sae.v1",
        "layer": layer,
        "latent_dim": latent_dim,
        "decoder_weight": [[0.0, 1.0] for _ in range(latent_dim)],
        "encoder_weight": [[1.0, 0.0] for _ in range(latent_dim)],
        "encoder_bias": [0.0 for _ in range(latent_dim)],
        "mean": [0.0, 0.0],
    }


def _card(feature_id: str, layer: int, label: str) -> FeatureCard:
    return FeatureCard(
        feature_id=feature_id,
        model="m",
        layer=layer,
        label=label,
        explanation="",
        importance=0.5,
        association=0.5,
        specificity=0.1,
        causal_effect=0.1,
        stability=1.0,
        examples=[],
        source="trained-sae",
        fingerprint=FeatureFingerprint(
            feature_id=feature_id,
            model="m",
            layer=layer,
            text=label,
            text_vector=[],
            activation_signature=[],
            decoder_signature=[],
            causal_vector=[],
        ),
        causal_effects={"signed_causal_effect": 0.1, "strong_causal_score": 0.1},
    )
