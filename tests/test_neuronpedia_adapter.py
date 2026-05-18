import json
from pathlib import Path

from oracle_sae.adapters.neuronpedia import (
    NeuronpediaClient,
    NeuronpediaFeatureProvider,
    NeuronpediaFeatureRef,
    load_neuronpedia_feature_refs,
    neuronpedia_payload_to_evidence,
)
from oracle_sae.criteria import HeuristicCriterionCompiler


def test_neuronpedia_ref_parses_feature_ids_and_urls():
    ref = NeuronpediaFeatureRef.parse("gpt2-small@6-res_scefr-ajt:650")
    url_ref = NeuronpediaFeatureRef.parse(
        "https://www.neuronpedia.org/api/feature/gpt2-small/6-res_scefr-ajt/650"
    )

    assert ref == url_ref
    assert ref.feature_id == "gpt2-small@6-res_scefr-ajt:650"


def test_neuronpedia_payload_maps_to_feature_evidence():
    payload = {
        "modelId": "gpt2-small",
        "layer": "6-res_scefr-ajt",
        "index": "650",
        "sourceSetName": "res_scefr-ajt",
        "maxActApprox": 22.735,
        "frac_nonzero": 0.00886,
        "vector": [0.1, -0.2],
        "pos_str": [" diameter", " radius"],
        "pos_values": [1.23, 1.1],
        "neg_str": [" Peb"],
        "neg_values": [-0.99],
        "explanations": [
            {
                "description": "measurements in meters or feet",
                "explanationModelName": "gpt-3.5-turbo",
                "scores": [{"value": 0.96}],
            }
        ],
        "activations": [
            {
                "tokens": ["The", " radius", " is", " 10", " meters"],
                "values": [0, 3, 0, 1, 5],
            }
        ],
    }

    evidence = neuronpedia_payload_to_evidence(payload)

    assert evidence.feature_id == "gpt2-small@6-res_scefr-ajt:650"
    assert evidence.layer == 6
    assert evidence.label == "measurements in meters or feet"
    assert evidence.decoder_signature == [0.1, -0.2]
    assert evidence.causal_effects["specificity"] == 0.96
    assert evidence.examples[0].startswith("max_activation=5.000")


def test_neuronpedia_provider_uses_client_and_filters_model():
    payloads = {
        "https://np.test/api/feature/gpt2-small/6-res_scefr-ajt/650": {
            "modelId": "gpt2-small",
            "layer": "6-res_scefr-ajt",
            "index": "650",
            "explanations": [{"description": "measurements"}],
        },
        "https://np.test/api/feature/other/1-res/2": {
            "modelId": "other",
            "layer": "1-res",
            "index": "2",
            "explanations": [{"description": "other"}],
        },
    }
    client = NeuronpediaClient(base_url="https://np.test", fetch_json=lambda url: payloads[url])
    provider = NeuronpediaFeatureProvider(
        ["gpt2-small@6-res_scefr-ajt:650", "other@1-res:2"],
        client=client,
    )

    evidence = provider.features_for(
        "gpt2-small",
        HeuristicCriterionCompiler().compile("measurements"),
    )

    assert [item.feature_id for item in evidence] == ["gpt2-small@6-res_scefr-ajt:650"]


def test_load_neuronpedia_feature_refs_supports_text_and_json(tmp_path: Path):
    text_path = tmp_path / "features.txt"
    text_path.write_text("# comment\ngpt2-small@6-res_scefr-ajt:650\n", encoding="utf-8")
    json_path = tmp_path / "features.json"
    json_path.write_text(json.dumps({"features": ["gpt2-small@6-res_scefr-ajt:650"]}), encoding="utf-8")

    assert load_neuronpedia_feature_refs(text_path) == ["gpt2-small@6-res_scefr-ajt:650"]
    assert load_neuronpedia_feature_refs(json_path) == ["gpt2-small@6-res_scefr-ajt:650"]
