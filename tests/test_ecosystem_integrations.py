import json
from pathlib import Path

from oracle_sae.adapters.goodfire import GoodfireFeatureProvider
from oracle_sae.adapters.scope import ScopeFeatureProvider
from oracle_sae.criteria import HeuristicCriterionCompiler
from oracle_sae.graphs import build_attribution_graph, export_attribution_graph
from oracle_sae.hf_publish import publish_hf_artifact, render_hf_card
from oracle_sae.nnsight_records import _resolve_activation_path, parse_activation_paths
from oracle_sae.reporting import write_inspection_report
from oracle_sae.scaling import ScalePlan, parse_bytes, parse_count_float, parse_count_int
from oracle_sae.transformerlens_records import parse_hook_names


def test_goodfire_provider_uses_feature_search():
    provider = GoodfireFeatureProvider(
        client=_FakeGoodfireClient(),
        variant_factory=lambda model: f"variant:{model}",
        top_k=2,
    )

    evidence = provider.features_for(
        "meta-llama/Llama-3.1-8B-Instruct",
        HeuristicCriterionCompiler().compile("formal writing style"),
    )

    assert len(evidence) == 1
    assert evidence[0].feature_id == "goodfire:42"
    assert evidence[0].label == "formal writing"
    assert evidence[0].metadata["searched_model"] == "variant:meta-llama/Llama-3.1-8B-Instruct"


def test_scope_provider_adds_scope_metadata():
    provider = ScopeFeatureProvider(
        source="gemma-scope",
        release="fake-release",
        sae_id="blocks.1.hook_resid_post",
        feature_indices=[0],
        sae_loader=_fake_sae_loader,
    )

    evidence = provider.features_for("google/gemma-2-2b", HeuristicCriterionCompiler().compile("math"))

    assert evidence[0].source == "gemma-scope"
    assert evidence[0].metadata["scope_label"] == "Gemma Scope"
    assert evidence[0].metadata["scope_homepage"].startswith("https://")


def test_parse_transformerlens_and_nnsight_paths():
    assert parse_hook_names(["blocks.0.hook_resid_post, blocks.1.hook_mlp_out"]) == [
        "blocks.0.hook_resid_post",
        "blocks.1.hook_mlp_out",
    ]
    assert parse_activation_paths(["transformer.h[6].output[0], model.layers[2].output"]) == [
        "transformer.h[6].output[0]",
        "model.layers[2].output",
    ]


def test_nnsight_path_resolver_supports_indexed_paths():
    root = _PathRoot()

    value = _resolve_activation_path(root, "transformer.h[1].output[0]")

    assert value == "layer-1-output-0"


def test_hf_publish_dry_run_validates_paths(tmp_path: Path):
    artifact = tmp_path / "report.json"
    artifact.write_text("{}", encoding="utf-8")

    result = publish_hf_artifact(
        repo_id="user/interp-lab-demo",
        paths=[artifact],
        path_in_repo="reports/report.json",
        dry_run=True,
    )

    assert result.dry_run is True
    assert result.uploaded == ["reports/report.json"]
    assert "interp-lab" in render_hf_card(repo_id="user/repo", repo_type="dataset", title=None, tags=[])


def test_attribution_graph_export(tmp_path: Path):
    report = _toy_report()
    report_path, _ = write_inspection_report(report, tmp_path / "report")
    out = tmp_path / "graph.json"
    markdown_out = tmp_path / "graph.md"

    path = export_attribution_graph(
        report_path=report_path,
        out_path=out,
        markdown_out_path=markdown_out,
        include_similarity_edges=True,
    )
    graph = json.loads(path.read_text(encoding="utf-8"))

    assert graph["schema_version"] == "interp-lab.attribution_graph.v1"
    assert any(edge["type"] == "causal_effect" for edge in graph["edges"])
    assert "# Attribution Graph" in markdown_out.read_text(encoding="utf-8")


def test_scale_plan_marks_trillion_parameter_remote_path():
    plan = ScalePlan(
        model_params=1e12,
        tokens=1_000_000,
        d_model=8192,
        selected_layers=4,
        latent_dim=131072,
        dtype="bf16",
        shards=256,
    ).to_dict()

    assert plan["schema_version"] == "interp-lab.scale_plan.v2"
    assert plan["activation_storage_bytes"] > 0
    assert plan["profile"] == "frontier-lab"
    assert plan["estimates"]["causal_validation"]["estimated_forward_passes"] > 0
    assert plan["agent_next_actions"]
    assert any("1T+" in item for item in plan["recommendations"])


def test_scale_plan_accounts_for_model_weight_load():
    plan = ScalePlan(
        model_params=5_000_000_000,
        tokens=12,
        d_model=1536,
        selected_layers=1,
        latent_dim=512,
        dtype="bf16",
    ).to_dict()

    assert plan["profile"] == "single-gpu"
    assert plan["estimates"]["model_weight_storage"]["bytes"] == 10_000_000_000
    assert plan["model_weight_bytes"] == 10_000_000_000


def test_scale_plan_accepts_human_friendly_values():
    assert parse_count_float("1T") == 1_000_000_000_000
    assert parse_count_int("1.5B") == 1_500_000_000
    assert parse_bytes("64GB") == 64 * 1024**3


class _FakeGoodfireClient:
    def __init__(self):
        self.features = _FakeGoodfireFeatures()


class _FakeGoodfireFeatures:
    def search(self, query, *, model, top_k):
        assert query == "formal writing style"
        assert top_k == 2
        return [
            {
                "index": 42,
                "label": "formal writing",
                "description": "formal style",
                "layer": 19,
                "examples": ["Dear committee"],
                "searched_model": model,
            }
        ]


class _FakeSAE:
    W_dec = [[0.2, 0.1]]


def _fake_sae_loader(_release, _sae_id, *, device, force_download):
    return _FakeSAE(), {"hook_name": "blocks.1.hook_resid_post", "hook_layer": 1}, [0.5]


class _PathRoot:
    def __init__(self):
        self.transformer = _Transformer()


class _Transformer:
    def __init__(self):
        self.h = [_Layer(0), _Layer(1)]


class _Layer:
    def __init__(self, index):
        self.output = [f"layer-{index}-output-0"]


def _toy_report():
    from oracle_sae.adapters.toy import ToyFeatureProvider, ToyInterventionRunner, ToyVerbalizer
    from oracle_sae.pipeline import inspect_model

    return inspect_model(
        model="toy/model-a",
        criterion_text="benchmark awareness",
        feature_provider=ToyFeatureProvider(),
        verbalizer=ToyVerbalizer(),
        intervention_runner=ToyInterventionRunner(),
        top_k=2,
    )
