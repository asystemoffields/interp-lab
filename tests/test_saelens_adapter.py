import pytest

from interp_lab.adapters.saelens import SAELensFeatureProvider, parse_feature_indices
from interp_lab.criteria import HeuristicCriterionCompiler


class FakeSAE:
    W_dec = [
        [0.1, 0.2, 0.3],
        [0.4, 0.0, -0.4],
        [1.0, 0.0, 0.0],
    ]


def fake_loader(release, sae_id, *, device, force_download):
    assert release == "fake-release"
    assert sae_id == "blocks.2.hook_resid_post"
    assert device == "cpu"
    assert force_download is False
    cfg = {"hook_name": "blocks.2.hook_resid_post", "hook_layer": 2}
    sparsity = [0.01, 0.9, 0.2]
    return FakeSAE(), cfg, sparsity


def test_parse_feature_indices_supports_ranges():
    assert parse_feature_indices("0,2,4-6") == [0, 2, 4, 5, 6]
    assert parse_feature_indices(None) is None
    with pytest.raises(ValueError):
        parse_feature_indices("5-3")


def test_saelens_provider_loads_selected_features_from_injected_loader():
    provider = SAELensFeatureProvider(
        release="fake-release",
        sae_id="blocks.2.hook_resid_post",
        feature_indices=[1],
        sae_loader=fake_loader,
        feature_metadata={1: {"label": "selected feature", "examples": ["example"]}},
    )

    evidence = provider.features_for(
        "fake/model",
        HeuristicCriterionCompiler().compile("criterion"),
    )

    assert len(evidence) == 1
    assert evidence[0].feature_id == "fake-release@blocks.2.hook_resid_post:1"
    assert evidence[0].layer == 2
    assert evidence[0].label == "selected feature"
    assert evidence[0].decoder_signature == [0.4, 0.0, -0.4]
    assert evidence[0].metadata["sparsity"] == 0.9


def test_saelens_provider_defaults_to_high_sparsity_features():
    provider = SAELensFeatureProvider(
        release="fake-release",
        sae_id="blocks.2.hook_resid_post",
        max_features=2,
        sae_loader=fake_loader,
    )

    evidence = provider.features_for(
        "fake/model",
        HeuristicCriterionCompiler().compile("criterion"),
    )

    assert [item.metadata["feature_index"] for item in evidence] == [1, 2]


def test_saelens_provider_reports_missing_dependency(monkeypatch):
    from interp_lab.adapters import saelens as saelens_adapter

    def missing_module(name):
        raise ImportError("missing")

    monkeypatch.setattr(saelens_adapter.importlib, "import_module", missing_module)
    provider = SAELensFeatureProvider(release="release", sae_id="id", feature_indices=[0])

    with pytest.raises(RuntimeError, match="SAELens is not installed"):
        provider.features_for("model", HeuristicCriterionCompiler().compile("criterion"))
