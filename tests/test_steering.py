"""Tests for steering.py: export provenance gating, direction resolution, stubbed apply."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import interp_lab.steering as steering
from interp_lab import __version__
from interp_lab.schema import Criterion, FeatureCard, FeatureFingerprint, InspectionReport
from interp_lab.steering import (
    STEERING_GENERATION_SCHEMA,
    STEERING_SCHEMA,
    apply_steering,
    build_apply_steering_parser,
    build_export_steering_parser,
    export_steering_vector,
    load_steering_artifact,
    run_export_steering_from_args,
)

CRITERION = "the model is aware it is being evaluated"

VALIDATED_EFFECTS = {
    "criterion": 0.21,
    "signed_causal_effect": 0.21,
    "strong_causal_score": 0.15,
    "intervention_record_count": 3.0,
}


def _fingerprint(feature_id: str, model: str = "toy/m") -> FeatureFingerprint:
    return FeatureFingerprint(
        feature_id=feature_id,
        model=model,
        layer=2,
        text="evaluation awareness",
        text_vector=[0.5, 0.5],
        activation_signature=[1.0, 0.0],
        decoder_signature=[],
        causal_vector=[],
    )


def _card(
    feature_id: str,
    *,
    causal_effects: dict | None = None,
    metadata: dict | None = None,
    layer: int = 2,
    label: str = "eval awareness feature",
) -> FeatureCard:
    return FeatureCard(
        feature_id=feature_id,
        model="toy/m",
        layer=layer,
        label=label,
        explanation="",
        importance=0.5,
        association=0.4,
        specificity=0.1,
        causal_effect=0.2,
        stability=0.5,
        examples=["the assistant suspects a test"],
        source="test",
        fingerprint=_fingerprint(feature_id),
        metadata=metadata or {},
        causal_effects=causal_effects if causal_effects is not None else dict(VALIDATED_EFFECTS),
    )


def _report(cards: list[FeatureCard]) -> InspectionReport:
    return InspectionReport(model="toy/m", criterion=Criterion(text=CRITERION), cards=cards)


def _sae_artifact(layer: int = 2) -> dict:
    return {
        "format": "interp-lab.sae.v1",
        "model": "toy/m",
        "layer": layer,
        "input_dim": 3,
        "latent_dim": 2,
        "source_feature_ids": ["H2:D0", "H2:D1", "H2:D2"],
        "mean": [0.0, 0.0, 0.0],
        "encoder_weight": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        "encoder_bias": [0.0, 0.0],
        "decoder_weight": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        "method": "fallback-dictionary",
        "metrics": {},
        "config": {},
    }


# --- export -------------------------------------------------------------------


def test_export_sae_decoder_row_matches_artifact(tmp_path: Path):
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(_report([_card("SAE:L2:F1")]).to_dict()), encoding="utf-8")
    sae_path = tmp_path / "sae.json"
    sae_path.write_text(json.dumps(_sae_artifact()), encoding="utf-8")
    out = tmp_path / "steer.json"

    payload = export_steering_vector(report_path, "SAE:L2:F1", sae=sae_path, out=out)

    assert payload["schema_version"] == STEERING_SCHEMA
    assert payload["direction"] == {"kind": "vector", "values": [4.0, 5.0, 6.0], "dim": 3}
    assert payload["layer"] == 2
    assert payload["model"] == "toy/m"
    assert payload["criterion"] == CRITERION
    assert payload["provenance"] == "intervention"
    assert payload["measured_signed_effect"] == 0.21
    assert payload["signed_effect_provenance"] == "intervention"
    assert payload["tool"] == {"name": "interp-lab", "version": __version__}
    assert payload["source"]["report_path"] == str(report_path)
    assert payload["source"]["report_sha256"] == hashlib.sha256(report_path.read_bytes()).hexdigest()
    assert payload["source"]["sae_sha256"] == hashlib.sha256(sae_path.read_bytes()).hexdigest()
    # The written file is the same payload and round-trips through the loader.
    assert json.loads(out.read_text(encoding="utf-8")) == payload
    assert load_steering_artifact(out)["feature_id"] == "SAE:L2:F1"
    # An SAE latent without the artifact is not exportable.
    with pytest.raises(ValueError, match="--sae|SAE artifact"):
        export_steering_vector(report_path, "SAE:L2:F1", out=tmp_path / "other.json")


def test_export_hidden_dimension_one_hot(tmp_path: Path):
    report = _report([_card("L3:D7", layer=3), _card("L0:D1", layer=0)])
    out = tmp_path / "steer.json"

    payload = export_steering_vector(report, "L3:D7", out=out)

    assert payload["direction"] == {"kind": "hidden_dim", "index": 7, "dim": None}
    assert payload["layer"] == 3
    assert payload["source"]["report_path"] is None  # in-memory report: nothing to hash
    with pytest.raises(ValueError, match="[Ll]ayer 0"):
        export_steering_vector(report, "L0:D1", out=tmp_path / "l0.json")
    with pytest.raises(ValueError, match="not in this report"):
        export_steering_vector(report, "L9:D9", out=tmp_path / "missing.json")


def test_export_refuses_association_only_evidence(tmp_path: Path):
    report = _report(
        [_card("L3:D7", layer=3, causal_effects={"criterion": 0.5, "signed_association": 0.4})]
    )
    out = tmp_path / "steer.json"

    with pytest.raises(ValueError, match="--allow-unvalidated"):
        export_steering_vector(report, "L3:D7", out=out)
    assert not out.exists()

    # Intervention provenance without a signed effect is still not steerable evidence.
    unsigned = _report(
        [_card("L3:D7", layer=3, causal_effects={"criterion": 0.5, "intervention_record_count": 2.0})]
    )
    with pytest.raises(ValueError, match="--allow-unvalidated"):
        export_steering_vector(unsigned, "L3:D7", out=out)


def test_export_allow_unvalidated_stamps_provenance(tmp_path: Path):
    report = _report(
        [_card("L3:D7", layer=3, causal_effects={"criterion": 0.5, "signed_association": 0.4})]
    )
    out = tmp_path / "steer.json"

    payload = export_steering_vector(report, "L3:D7", out=out, allow_unvalidated=True)

    assert payload["provenance"] == "unvalidated"
    assert "UNVALIDATED" in payload["unvalidated_warning"]
    assert payload["measured_signed_effect"] == 0.4
    assert payload["signed_effect_provenance"] == "association"
    assert json.loads(out.read_text(encoding="utf-8"))["provenance"] == "unvalidated"


def test_export_recommended_strength_derivation_and_null_reason(tmp_path: Path):
    with_strength = _card(
        "L3:D7",
        layer=3,
        metadata={"interventions": {"count": 2, "mean_directed_effect": 0.1, "selected_strength": -3.0}},
    )
    sweep_only = _card(
        "L4:D1",
        layer=4,
        metadata={
            "interventions": {
                "count": 2,
                "mean_directed_effect": 0.1,
                "strength_sweep": [
                    {"strength": 1.0, "specificity": 0.02},
                    {"strength": 10.0, "specificity": 0.08},
                    {"strength": 30.0, "specificity": 0.01},
                ],
            }
        },
    )
    bare = _card("L5:D2", layer=5)
    report = _report([with_strength, sweep_only, bare])

    selected = export_steering_vector(report, "L3:D7", out=tmp_path / "a.json")
    assert selected["recommended_strength"] == -3.0
    assert selected["recommended_strength_source"] == "interventions.selected_strength"
    assert selected["recommended_strength_reason"] is None

    swept = export_steering_vector(report, "L4:D1", out=tmp_path / "b.json")
    assert swept["recommended_strength"] == 10.0
    assert swept["recommended_strength_source"] == "interventions.strength_sweep_best_specificity"

    unknown = export_steering_vector(report, "L5:D2", out=tmp_path / "c.json")
    assert unknown["recommended_strength"] is None
    assert unknown["recommended_strength_source"] is None
    assert "--strength" in unknown["recommended_strength_reason"]

    explicit = export_steering_vector(report, "L5:D2", out=tmp_path / "d.json", strength=5.0)
    assert explicit["recommended_strength"] == 5.0
    assert explicit["recommended_strength_source"] == "explicit"


def test_export_steering_parser_runs_end_to_end(tmp_path: Path):
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(_report([_card("SAE:L2:F0")]).to_dict()), encoding="utf-8")
    sae_path = tmp_path / "sae.json"
    sae_path.write_text(json.dumps(_sae_artifact()), encoding="utf-8")
    out = tmp_path / "steer.json"
    args = build_export_steering_parser().parse_args(
        ["--report", str(report_path), "--feature", "SAE:L2:F0", "--sae", str(sae_path), "--out", str(out)]
    )

    payload = run_export_steering_from_args(args)

    assert out.exists()
    assert payload["direction"]["values"] == [1.0, 2.0, 3.0]
    assert args.allow_unvalidated is False and args.strength is None


# --- apply (stubbed model/generate, real register_hidden_steering) -------------


class _StubHandle:
    def __init__(self, layer: "_StubLayer"):
        self._layer = layer

    def remove(self) -> None:
        self._layer.active = None
        self._layer.removed += 1


class _StubLayer:
    def __init__(self):
        self.registered = []
        self.active = None
        self.removed = 0

    def register_forward_hook(self, hook):
        self.registered.append(hook)
        self.active = hook
        return _StubHandle(self)


class _StubModel:
    """Decoder-shaped stub: transformer.h list so register_hidden_steering hooks it."""

    def __init__(self, n_layers: int = 3, hidden_size: int = 4):
        self.transformer = SimpleNamespace(
            h=[_StubLayer() for _ in range(n_layers)],
            ln_f=_StubLayer(),
        )
        self.config = SimpleNamespace(hidden_size=hidden_size)


def _stub_generate_for(model: _StubModel, layer: int):
    def generate(*, model: _StubModel, tokenizer, prompt, device, max_new_tokens):
        steered = model.transformer.h[layer - 1].active is not None
        return ("steered:" if steered else "baseline:") + prompt

    return generate


def _steering_artifact(direction: dict, *, layer: int = 2, recommended: float | None = None) -> dict:
    return {
        "schema_version": STEERING_SCHEMA,
        "model": "toy/m",
        "criterion": CRITERION,
        "feature_id": "SAE:L2:F1",
        "label": "eval awareness feature",
        "layer": layer,
        "direction": direction,
        "recommended_strength": recommended,
        "recommended_strength_source": None if recommended is None else "interventions.selected_strength",
        "recommended_strength_reason": None,
        "measured_signed_effect": 0.21,
        "signed_effect_provenance": "intervention",
        "provenance": "intervention",
        "source": {"report_path": None, "report_sha256": None, "sae_path": None, "sae_sha256": None},
        "created_at": "2026-06-10T00:00:00+00:00",
        "tool": {"name": "interp-lab", "version": __version__},
    }


def test_apply_steering_writes_jsonl_and_routes_register_hidden_steering(tmp_path: Path, monkeypatch):
    model = _StubModel()
    recorded = []
    real_register = steering.register_hidden_steering

    def recording_register(target_model, layer, direction, strength):
        recorded.append((layer, direction, strength))
        return real_register(target_model, layer, direction, strength)

    monkeypatch.setattr(steering, "register_hidden_steering", recording_register)
    artifact = _steering_artifact({"kind": "vector", "values": [4.0, 5.0, 6.0], "dim": 3})
    out = tmp_path / "generations.jsonl"

    summary = apply_steering(
        artifact,
        prompts=["alpha", "beta"],
        out=out,
        strength=4.0,
        model_loader=lambda: (None, model, "cpu"),
        generate_fn=_stub_generate_for(model, layer=2),
    )

    assert summary == {
        "rows": 2,
        "out": str(out),
        "strength_used": 4.0,
        "provenance": "intervention",
        "feature_id": "SAE:L2:F1",
        "model": "toy/m",
    }
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [row["prompt"] for row in rows] == ["alpha", "beta"]
    assert rows[0]["baseline_text"] == "baseline:alpha"
    assert rows[0]["steered_text"] == "steered:alpha"
    assert all(row["schema_version"] == STEERING_GENERATION_SCHEMA for row in rows)
    assert all(row["strength"] == 4.0 for row in rows)
    assert all(row["feature_id"] == "SAE:L2:F1" for row in rows)
    assert all(row["provenance"] == "intervention" for row in rows)
    # The decoder-row direction went through register_hidden_steering, once per prompt,
    # on the artifact's layer, and every hook was removed afterwards.
    assert recorded == [(2, [4.0, 5.0, 6.0], 4.0), (2, [4.0, 5.0, 6.0], 4.0)]
    hooked_layer = model.transformer.h[1]
    assert len(hooked_layer.registered) == 2
    assert hooked_layer.removed == 2
    assert hooked_layer.active is None
    assert model.transformer.h[0].registered == []


def test_apply_steering_hidden_dim_uses_recommended_strength_and_prompt_file(tmp_path: Path, monkeypatch):
    model = _StubModel(hidden_size=4)
    recorded = []
    monkeypatch.setattr(
        steering,
        "register_hidden_steering",
        lambda target_model, layer, direction, strength: (
            recorded.append((layer, direction, strength)),
            _StubHandle(model.transformer.h[layer - 1]),
        )[1],
    )
    artifact = _steering_artifact({"kind": "hidden_dim", "index": 2, "dim": None}, layer=3, recommended=-3.0)
    prompts_path = tmp_path / "prompts.jsonl"
    prompts_path.write_text(
        json.dumps({"text": "alpha"}) + "\n\n" + json.dumps({"prompt": "beta"}) + "\ngamma\n",
        encoding="utf-8",
    )
    out = tmp_path / "generations.jsonl"

    summary = apply_steering(
        artifact,
        prompts=prompts_path,
        out=out,
        model_loader=lambda: (None, model, "cpu"),
        generate_fn=lambda **kwargs: "text",
    )

    assert summary["strength_used"] == -3.0
    assert summary["rows"] == 3
    # One-hot direction resolved from the stub model's hidden size at apply time.
    assert recorded[0] == (3, [0.0, 0.0, 1.0, 0.0], -3.0)
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [row["prompt"] for row in rows] == ["alpha", "beta", "gamma"]


def test_apply_steering_rejects_bad_artifacts_and_missing_strength(tmp_path: Path):
    model = _StubModel()
    loader = lambda: (None, model, "cpu")  # noqa: E731
    with pytest.raises(ValueError, match="schema_version"):
        apply_steering({"schema_version": "nope"}, prompts=["a"], out=tmp_path / "x.jsonl")
    # recommended_strength is null and no --strength was given: refuse, never invent one.
    artifact = _steering_artifact({"kind": "vector", "values": [1.0], "dim": 1})
    with pytest.raises(ValueError, match="[Ss]trength"):
        apply_steering(
            artifact,
            prompts=["a"],
            out=tmp_path / "y.jsonl",
            model_loader=loader,
            generate_fn=lambda **kwargs: "text",
        )
    assert not (tmp_path / "y.jsonl").exists()
    # The apply parser carries the documented defaults for the CLI wiring.
    args = build_apply_steering_parser().parse_args(
        ["--artifact", "a.json", "--prompts", "p.jsonl", "--out", "o.jsonl"]
    )
    assert args.max_new_tokens == 32 and args.device == "cpu" and args.strength is None
