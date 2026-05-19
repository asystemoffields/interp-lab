import json
from pathlib import Path

from oracle_sae.adapters.records import ActivationRecordFeatureProvider
from oracle_sae.criteria import HeuristicCriterionCompiler


def test_records_provider_aggregates_ranked_feature_evidence(tmp_path: Path):
    path = tmp_path / "records.jsonl"
    rows = [
        {
            "model": "m",
            "prompt_id": "pos-1",
            "text": "evaluation-looking prompt",
            "criterion_score": 1.0,
            "features": [
                {"feature_id": "L2:F_eval", "activation": 0.9, "label": "evaluation awareness", "layer": 2},
                {"feature_id": "L3:F_other", "activation": 0.1, "label": "unrelated", "layer": 3},
            ],
        },
        {
            "model": "m",
            "prompt_id": "neg-1",
            "text": "ordinary user prompt",
            "criterion_score": 0.0,
            "features": [
                {"feature_id": "L2:F_eval", "activation": 0.1, "label": "evaluation awareness", "layer": 2},
                {"feature_id": "L3:F_other", "activation": 0.8, "label": "unrelated", "layer": 3},
            ],
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    criterion = HeuristicCriterionCompiler().compile("evaluation awareness")

    provider = ActivationRecordFeatureProvider(path)
    evidence = provider.features_for("m", criterion)

    assert evidence[0].feature_id == "L2:F_eval"
    assert evidence[0].causal_effects["criterion"] > 0.9
    assert evidence[0].layer == 2
    assert evidence[0].examples[0].startswith("pos-1:")
    assert provider.report_metadata()["evidence"]["record_count"] == 2
    assert provider.report_metadata()["evidence"]["positive_record_count"] == 1
    suppressor = next(item for item in evidence if item.feature_id == "L3:F_other")
    assert suppressor.metadata["signed_association"] < 0


def test_records_provider_accepts_mapping_features(tmp_path: Path):
    path = tmp_path / "records.jsonl"
    rows = [
        {
            "model": "m",
            "prompt_id": "p1",
            "text": "positive",
            "criterion_score": 1,
            "features": {"L4:F1": 1.0},
            "feature_metadata": {"L4:F1": {"label": "mapped feature"}},
        },
        {
            "model": "m",
            "prompt_id": "p2",
            "text": "negative",
            "criterion_score": 0,
            "features": {"L4:F1": 0.0},
            "feature_metadata": {"L4:F1": {"label": "mapped feature"}},
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    evidence = ActivationRecordFeatureProvider(path).features_for(
        "m",
        HeuristicCriterionCompiler().compile("criterion"),
    )

    assert evidence[0].label == "mapped feature"
    assert evidence[0].layer == 4
