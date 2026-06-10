import json
import re
from pathlib import Path

import pytest

from interp_lab.adapters.goodfire import goodfire_feature_to_evidence
from interp_lab.adapters.nla import load_nla_explanation_records
from interp_lab.adapters.toy import ToyFeatureProvider
from interp_lab.schema import Criterion


def test_goodfire_evidence_has_no_fabricated_causal_effects():
    evidence = goodfire_feature_to_evidence(
        {"index": 7, "label": "formal writing"},
        model="meta-llama/Llama-3.1-8B-Instruct",
    )

    # Goodfire semantic search returns no causal measurements; reporting a
    # constant specificity would render as a measured value.
    assert evidence.causal_effects == {}


def test_toy_provider_handles_criterion_without_examples():
    provider = ToyFeatureProvider(feature_count=2, dimensions=4)
    criterion = Criterion(text="benchmark awareness")

    features = provider.features_for("toy/model-a", criterion)

    assert len(features) == 2
    assert "benchmark awareness" in features[0].examples[0]
    assert "unrelated neutral text" in features[0].examples[1]


def test_toy_provider_still_uses_provided_examples():
    provider = ToyFeatureProvider(feature_count=1, dimensions=4)
    criterion = Criterion(
        text="benchmark awareness",
        positive_examples=["the model notices the test"],
        negative_examples=["a recipe for soup"],
    )

    features = provider.features_for("toy/model-a", criterion)

    assert "the model notices the test" in features[0].examples[0]
    assert "a recipe for soup" in features[0].examples[1]


def test_nla_jsonl_loader_reports_path_and_line_for_malformed_rows(tmp_path: Path):
    path = tmp_path / "explanations.jsonl"
    path.write_text(
        json.dumps({"feature_id": "L1:F1", "explanation": "good row"})
        + "\n\n{not valid json\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=rf"{re.escape(str(path))}:3: invalid JSON"):
        load_nla_explanation_records(path)


def test_nla_jsonl_loader_still_loads_valid_rows(tmp_path: Path):
    path = tmp_path / "explanations.jsonl"
    path.write_text(
        json.dumps({"feature_id": "L1:F1", "explanation": "tracks units"}) + "\n",
        encoding="utf-8",
    )

    records = load_nla_explanation_records(path)

    assert records["L1:F1"].explanation == "tracks units"
