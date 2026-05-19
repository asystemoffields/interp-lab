from oracle_sae.graphs import build_attribution_graph, render_attribution_graph_markdown
from oracle_sae.schema import Criterion, FeatureCard, FeatureFingerprint, InspectionReport


def test_attribution_graph_includes_candidate_mechanism_paths_and_groups():
    report = InspectionReport(
        model="m",
        criterion=Criterion(text="code-oriented completions"),
        cards=[
            _card(
                "SAE:L12:F1",
                12,
                "trained SAE latent 1",
                [0.0, 1.0, 2.0, 3.0],
                examples=["p1: activation=3.0 | Write a Python function | token[3]='Python'"],
                signed=0.01,
                strong=0.0,
            ),
            _card(
                "SAE:L24:F8",
                24,
                "trained SAE latent 8",
                [0.0, 1.1, 2.1, 3.2],
                examples=["p1: activation=3.2 | Write a Python function | token[4]='function'"],
                signed=0.08,
                strong=0.07,
            ),
        ],
    )

    graph = build_attribution_graph(report)

    assert any(node["type"] == "supernode" for node in graph["nodes"])
    path_edges = [edge for edge in graph["edges"] if edge["type"] == "coactivation" and edge["candidate_path"]]
    assert path_edges
    assert path_edges[0]["source"] == "SAE:L12:F1"
    assert path_edges[0]["target"] == "SAE:L24:F8"
    assert graph["mechanism_summary"]["candidate_paths"]
    assert graph["mechanism_summary"]["validation_plan"]


def test_attribution_graph_accepts_measured_path_patch_edges():
    report = InspectionReport(
        model="m",
        criterion=Criterion(text="code-oriented completions"),
        cards=[
            _card("SAE:L12:F1", 12, "source", [0.0, 1.0], examples=[], signed=0.01, strong=0.0),
            _card("SAE:L24:F8", 24, "target", [0.0, 1.0], examples=[], signed=0.08, strong=0.07),
        ],
    )
    path_records = [
        {
            "source_feature_id": "SAE:L12:F1",
            "target_feature_id": "SAE:L24:F8",
            "target_activation_delta": 0.25,
            "score_delta": 0.03,
            "strength": 2.0,
            "prompt_id": "p1",
        },
        {
            "source_feature_id": "SAE:L12:F1",
            "target_feature_id": "SAE:L24:F8",
            "target_activation_delta": 0.15,
            "score_delta": 0.01,
            "strength": 2.0,
            "prompt_id": "p2",
        },
        {
            "source_feature_id": "SAE:L12:F1",
            "target_feature_id": "SAE:L24:F8",
            "target_activation_delta": 0.05,
            "score_delta": 0.005,
            "strength": 2.0,
            "prompt_id": "p1",
            "metadata": {
                "control_type": "random_source",
                "control_source_feature_id": "SAE:L12:F9",
            },
        },
    ]

    graph = build_attribution_graph(report, path_records=path_records)

    path_edges = [edge for edge in graph["edges"] if edge["type"] == "path_patch"]
    assert path_edges
    assert path_edges[0]["mean_target_activation_delta"] == 0.2
    assert path_edges[0]["record_count"] == 2
    assert path_edges[0]["control_record_count"] == 1
    assert path_edges[0]["control_mean_abs_target_activation_delta"] == 0.05
    assert path_edges[0]["path_specificity_score"] == 0.15
    assert path_edges[0]["best_strength"]["strength"] == 2.0
    assert path_edges[0]["by_strength"][0]["mean_score_delta"] == 0.02
    assert path_edges[0]["by_strength"][0]["control_record_count"] == 1
    assert graph["mechanism_summary"]["candidate_paths"][0]["evidence"] == "path_patch"
    assert graph["mechanism_summary"]["candidate_paths"][0]["best_strength"]["strength"] == 2.0
    assert graph["mechanism_summary"]["candidate_paths"][0]["path_specificity_score"] == 0.15
    assert "held-out prompts" in graph["mechanism_summary"]["validation_plan"][0]


def test_attribution_graph_markdown_summarizes_validated_paths():
    graph = {
        "model": "m",
        "criterion": {"text": "code-oriented completions"},
        "nodes": [{"id": "criterion"}],
        "edges": [{"source": "SAE:L12:F1", "target": "SAE:L24:F8"}],
        "mechanism_summary": {
            "strong_causal_features": [
                {
                    "feature_id": "SAE:L24:F8",
                    "label": "target",
                    "layer": 24,
                    "role": "criterion_promoter",
                    "signed_effect": 0.08,
                    "strong_causal_score": 0.07,
                }
            ],
            "candidate_paths": [
                {
                    "source_feature_id": "SAE:L12:F1",
                    "target_feature_id": "SAE:L24:F8",
                    "evidence": "path_patch",
                    "mean_abs_target_activation_delta": 0.2,
                    "control_mean_abs_target_activation_delta": 0.05,
                    "path_specificity_score": 0.15,
                    "record_count": 2,
                    "validation": {
                        "status": "robust",
                        "interpretation": "The path replicated with a target-latent effect that beat controls.",
                        "reason_codes": ["passed_effect_control_and_sign_thresholds"],
                    },
                }
            ],
            "candidate_feature_groups": [],
            "path_validation_status_counts": {"robust": 1},
            "validation_plan": ["Repeat on held-out prompts."],
        },
    }

    markdown = render_attribution_graph_markdown(graph)

    assert "# Attribution Graph" in markdown
    assert "Path validation: `robust=1`" in markdown
    assert "`SAE:L12:F1 -> SAE:L24:F8`" in markdown
    assert "robust" in markdown
    assert "### Path Notes" in markdown
    assert "passed_effect_control_and_sign_thresholds" in markdown


def _card(
    feature_id: str,
    layer: int,
    label: str,
    activation_signature: list[float],
    *,
    examples: list[str],
    signed: float,
    strong: float,
) -> FeatureCard:
    fingerprint = FeatureFingerprint(
        feature_id=feature_id,
        model="m",
        layer=layer,
        text=label,
        text_vector=[],
        activation_signature=activation_signature,
        decoder_signature=[],
        causal_vector=[],
    )
    return FeatureCard(
        feature_id=feature_id,
        model="m",
        layer=layer,
        label=label,
        explanation="",
        importance=0.5,
        association=0.3,
        specificity=strong,
        causal_effect=strong,
        stability=1.0,
        examples=examples,
        source="trained-sae",
        fingerprint=fingerprint,
        causal_effects={
            "signed_causal_effect": signed,
            "strong_causal_score": strong,
        },
    )
