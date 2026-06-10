import json
import re
import sys
from pathlib import Path

import pytest

import interp_lab
from interp_lab.adapters.records import ActivationRecordFeatureProvider
from interp_lab.criteria import HeuristicCriterionCompiler
from interp_lab.gguf_records import (
    GGUF_INSTALL_MESSAGE,
    build_gguf_export_parser,
    build_hidden_dump_convert_parser,
    convert_hidden_state_dump,
    export_gguf_records,
    run_hidden_dump_convert_from_args,
    summarize_records,
)


class _FakeLlama:
    """Implements the duck-typed seam documented in gguf_records.py."""

    def __init__(self, embeddings_by_text, *, n_layers=4, metadata=None, use_method=True):
        self._embeddings_by_text = embeddings_by_text
        self._n_layers = n_layers
        self._use_method = use_method
        if metadata is not None:
            self.metadata = metadata
        self.embed_calls = []

    def n_layer(self):
        if not self._use_method:
            raise AttributeError("n_layer disabled for this fake")
        return self._n_layers

    def embed(self, text):
        self.embed_calls.append(text)
        return self._embeddings_by_text[text]


def _write_prompts(path: Path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _scored_prompts(path: Path):
    _write_prompts(
        path,
        [
            {"prompt_id": "pos-1", "text": "an evaluation-looking prompt", "criterion_score": 1.0},
            {"prompt_id": "pos-2", "text": "another synthetic test prompt", "criterion_score": 1.0},
            {"prompt_id": "neg-1", "text": "an ordinary user request", "criterion_score": 0.0},
        ],
    )
    return path


def _read_rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_export_gguf_records_writes_final_layer_rows(tmp_path: Path):
    dataset = _scored_prompts(tmp_path / "prompts.jsonl")
    # Per-token vectors: pool="last" must pick the final token's vector.
    fake = _FakeLlama(
        {
            "an evaluation-looking prompt": [[9.0, 9.0, 9.0], [1.0, 0.9, 0.1]],
            "another synthetic test prompt": [[9.0, 9.0, 9.0], [0.8, 1.0, 0.2]],
            "an ordinary user request": [[9.0, 9.0, 9.0], [0.1, 0.0, 0.9]],
        },
        n_layers=30,
    )

    out = export_gguf_records(
        model_path=tmp_path / "tiny.Q8_0.gguf",
        dataset_path=dataset,
        out_path=tmp_path / "records.jsonl",
        top_features=3,
        llama_factory=lambda model_path, **kwargs: fake,
    )

    rows = _read_rows(out)
    assert len(rows) == 3
    row = rows[0]
    assert row["model"] == "tiny.Q8_0.gguf"
    assert row["prompt_id"] == "pos-1"
    assert row["criterion_score"] == 1.0
    # Final-norm hidden state = hidden_states[n_layers] in the HF convention.
    feature_ids = {feature["feature_id"] for feature in row["features"]}
    assert feature_ids == {"L30:D0", "L30:D1", "L30:D2"}
    assert all(feature["layer"] == 30 for feature in row["features"])
    metadata = row["feature_metadata"]["L30:D0"]
    assert metadata["layer_convention"] == "hidden_state_index"
    assert metadata["source"] == "llama_cpp_embeddings"
    assert metadata["layers_available"] == "final_only"
    assert row["metadata"]["layers_available"] == "final_only"
    # pool="last": activation comes from the last token vector, not the first.
    activations = {feature["feature_id"]: feature["activation"] for feature in row["features"]}
    assert activations["L30:D0"] == 1.0


def test_export_gguf_records_selects_dimensions_by_variance(tmp_path: Path):
    dataset = _scored_prompts(tmp_path / "prompts.jsonl")
    # D0 and D2 are constant across prompts; D1 and D3 vary.
    fake = _FakeLlama(
        {
            "an evaluation-looking prompt": [[5.0, 0.0, -2.0, 10.0]],
            "another synthetic test prompt": [[5.0, 1.0, -2.0, 30.0]],
            "an ordinary user request": [[5.0, 2.0, -2.0, 20.0]],
        },
        n_layers=4,
    )

    out = export_gguf_records(
        model_path=tmp_path / "m.gguf",
        dataset_path=dataset,
        out_path=tmp_path / "records.jsonl",
        top_features=2,
        llama_factory=lambda model_path, **kwargs: fake,
    )

    rows = _read_rows(out)
    feature_ids = {feature["feature_id"] for feature in rows[0]["features"]}
    assert feature_ids == {"L4:D1", "L4:D3"}
    variances = {
        feature_id: metadata["selection_variance"]
        for feature_id, metadata in rows[0]["feature_metadata"].items()
    }
    assert variances["L4:D3"] > variances["L4:D1"] > 0


def test_export_gguf_records_mean_pool_and_flat_vectors(tmp_path: Path):
    dataset = tmp_path / "prompts.jsonl"
    _write_prompts(
        dataset,
        [
            {"prompt_id": "a", "text": "alpha", "criterion_score": 1.0},
            {"prompt_id": "b", "text": "beta", "criterion_score": 0.0},
        ],
    )
    fake = _FakeLlama(
        {
            "alpha": [[0.0, 2.0], [2.0, 4.0]],  # mean -> [1.0, 3.0]
            "beta": [4.0, 0.0],  # already-pooled flat vector is accepted as-is
        },
        n_layers=2,
    )

    out = export_gguf_records(
        model_path=tmp_path / "m.gguf",
        dataset_path=dataset,
        out_path=tmp_path / "records.jsonl",
        top_features=2,
        pool="mean",
        llama_factory=lambda model_path, **kwargs: fake,
    )

    rows = _read_rows(out)
    activations = {feature["feature_id"]: feature["activation"] for feature in rows[0]["features"]}
    assert activations == {"L2:D0": 1.0, "L2:D1": 3.0}
    activations = {feature["feature_id"]: feature["activation"] for feature in rows[1]["features"]}
    assert activations == {"L2:D0": 4.0, "L2:D1": 0.0}


def test_export_gguf_records_reads_layer_count_from_gguf_metadata(tmp_path: Path):
    dataset = tmp_path / "prompts.jsonl"
    _write_prompts(dataset, [{"prompt_id": "a", "text": "alpha", "criterion_score": 1.0}])
    fake = _FakeLlama(
        {"alpha": [[1.0, 2.0]]},
        metadata={"general.architecture": "llama", "llama.block_count": "6"},
        use_method=False,
    )
    fake.n_layer = None  # not callable: forces the metadata path

    out = export_gguf_records(
        model_path=tmp_path / "m.gguf",
        dataset_path=dataset,
        out_path=tmp_path / "records.jsonl",
        top_features=2,
        llama_factory=lambda model_path, **kwargs: fake,
    )

    rows = _read_rows(out)
    assert {feature["feature_id"] for feature in rows[0]["features"]} == {"L6:D0", "L6:D1"}


def test_export_gguf_records_requires_layer_count_when_undiscoverable(tmp_path: Path):
    dataset = tmp_path / "prompts.jsonl"
    _write_prompts(dataset, [{"prompt_id": "a", "text": "alpha", "criterion_score": 1.0}])
    fake = _FakeLlama({"alpha": [[1.0]]})
    fake.n_layer = None

    with pytest.raises(ValueError, match="n_layers"):
        export_gguf_records(
            model_path=tmp_path / "m.gguf",
            dataset_path=dataset,
            out_path=tmp_path / "records.jsonl",
            llama_factory=lambda model_path, **kwargs: fake,
        )

    # The explicit override unblocks the export.
    out = export_gguf_records(
        model_path=tmp_path / "m.gguf",
        dataset_path=dataset,
        out_path=tmp_path / "records.jsonl",
        n_layers=12,
        llama_factory=lambda model_path, **kwargs: fake,
    )
    assert _read_rows(out)[0]["features"][0]["feature_id"] == "L12:D0"


def test_export_gguf_records_errors_cleanly_without_llama_cpp(tmp_path: Path, monkeypatch):
    dataset = _scored_prompts(tmp_path / "prompts.jsonl")
    monkeypatch.setitem(sys.modules, "llama_cpp", None)  # force ImportError even if installed

    with pytest.raises(RuntimeError, match=re.escape("interp-lab[gguf]")) as excinfo:
        export_gguf_records(
            model_path=tmp_path / "m.gguf",
            dataset_path=dataset,
            out_path=tmp_path / "records.jsonl",
        )
    assert str(excinfo.value) == GGUF_INSTALL_MESSAGE


def test_gguf_records_flow_through_real_loader_and_pipeline(tmp_path: Path):
    dataset = _scored_prompts(tmp_path / "prompts.jsonl")
    # D0 tracks the criterion score; D1 anti-tracks it.
    fake = _FakeLlama(
        {
            "an evaluation-looking prompt": [[0.9, 0.1]],
            "another synthetic test prompt": [[0.8, 0.2]],
            "an ordinary user request": [[0.1, 0.9]],
        },
        n_layers=4,
    )
    out = export_gguf_records(
        model_path=tmp_path / "tiny.gguf",
        dataset_path=dataset,
        out_path=tmp_path / "records.jsonl",
        top_features=2,
        llama_factory=lambda model_path, **kwargs: fake,
    )

    provider = ActivationRecordFeatureProvider(out)
    evidence = provider.features_for(
        "tiny.gguf", HeuristicCriterionCompiler().compile("evaluation awareness")
    )
    assert evidence[0].feature_id == "L4:D0"
    assert evidence[0].layer == 4
    assert evidence[0].metadata["layers_available"] == "final_only"

    report = interp_lab.inspect(
        "tiny.gguf",
        "evaluation awareness",
        backend="records",
        records=str(out),
    )
    assert report.cards
    assert report.cards[0].feature_id == "L4:D0"

    summary = summarize_records(out)
    assert summary.record_count == 3
    assert summary.feature_count == 2
    assert summary.layers == [4]
    assert summary.layers_available == "final_only"
    assert summary.to_dict()["source"] == "llama_cpp_embeddings"


def _write_dump(path: Path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_convert_hidden_state_dump_multi_layer_round_trip(tmp_path: Path):
    prompts = tmp_path / "prompts.jsonl"
    _write_prompts(
        prompts,
        [
            {"prompt_id": "pos-1", "text": "synthetic eval prompt", "criterion_score": 1.0},
            {"prompt_id": "neg-1", "text": "ordinary prompt", "criterion_score": 0.0},
        ],
    )
    dump = _write_dump(
        tmp_path / "dump.jsonl",
        [
            {"prompt_index": 0, "layer": 0, "tokens": ["a", "b"], "hidden": [[0.0, 0.0], [0.9, 0.1]]},
            {"prompt_index": 0, "layer": 3, "tokens": ["a", "b"], "hidden": [[0.0, 0.0], [0.7, 0.3]]},
            {"prompt_index": 1, "layer": 0, "tokens": ["c"], "hidden": [[0.1, 0.8]]},
            {"prompt_index": 1, "layer": 3, "tokens": ["c"], "hidden": [[0.2, 0.9]]},
        ],
    )

    out = convert_hidden_state_dump(
        dump_path=dump,
        out_path=tmp_path / "records.jsonl",
        prompts_path=prompts,
        model_name="parcae-140m",
    )

    rows = _read_rows(out)
    assert [row["prompt_id"] for row in rows] == ["pos-1", "neg-1"]
    assert rows[0]["criterion_score"] == 1.0
    assert rows[1]["criterion_score"] == 0.0
    assert rows[0]["text"] == "synthetic eval prompt"
    feature_ids = {feature["feature_id"] for feature in rows[0]["features"]}
    assert feature_ids == {"L0:D0", "L0:D1", "L3:D0", "L3:D1"}
    assert rows[0]["feature_metadata"]["L3:D1"]["layer"] == 3
    assert rows[0]["feature_metadata"]["L3:D1"]["layer_convention"] == "hidden_state_index"
    assert rows[0]["metadata"]["dump_format"] == "interp-lab.hidden_state_dump.v1"

    evidence = ActivationRecordFeatureProvider(out).features_for(
        "parcae-140m", HeuristicCriterionCompiler().compile("evaluation awareness")
    )
    assert {item.layer for item in evidence} == {0, 3}
    report = interp_lab.inspect(
        "parcae-140m", "evaluation awareness", backend="records", records=str(out)
    )
    assert report.cards


def test_convert_hidden_dump_accepts_inline_scores_and_flat_vectors(tmp_path: Path):
    dump = _write_dump(
        tmp_path / "dump.jsonl",
        [
            {"prompt_index": 0, "layer": 2, "hidden": [1.0, 0.0], "text": "yes", "criterion_score": 1.0},
            {"prompt_index": 1, "layer": 2, "hidden": [0.0, 1.0], "text": "no", "criterion_score": 0.0},
        ],
    )

    out = convert_hidden_state_dump(dump_path=dump, out_path=tmp_path / "records.jsonl")

    rows = _read_rows(out)
    assert rows[0]["model"] == "hidden-state-dump"
    assert rows[0]["prompt_id"] == "dump-000"  # stable fallback id from the dump file name
    assert rows[0]["criterion_score"] == 1.0
    assert {feature["feature_id"] for feature in rows[0]["features"]} == {"L2:D0", "L2:D1"}


def test_convert_hidden_dump_stamps_custom_layer_convention(tmp_path: Path):
    dump = _write_dump(
        tmp_path / "dump.jsonl",
        [{"prompt_index": 0, "layer": 5, "hidden": [1.0], "text": "t", "criterion_score": 1.0}],
    )

    out = convert_hidden_state_dump(
        dump_path=dump,
        out_path=tmp_path / "records.jsonl",
        layer_convention="block_output_index",
    )

    rows = _read_rows(out)
    assert rows[0]["feature_metadata"]["L5:D0"]["layer_convention"] == "block_output_index"
    assert rows[0]["metadata"]["layer_convention"] == "block_output_index"


def test_convert_hidden_dump_limits_features_per_layer_by_variance(tmp_path: Path):
    dump = _write_dump(
        tmp_path / "dump.jsonl",
        [
            {"prompt_index": 0, "layer": 1, "hidden": [7.0, 0.0, 3.0], "text": "a", "criterion_score": 1.0},
            {"prompt_index": 1, "layer": 1, "hidden": [7.0, 5.0, 3.1], "text": "b", "criterion_score": 0.0},
        ],
    )

    out = convert_hidden_state_dump(
        dump_path=dump,
        out_path=tmp_path / "records.jsonl",
        features_per_layer=1,
    )

    rows = _read_rows(out)
    assert {feature["feature_id"] for feature in rows[0]["features"]} == {"L1:D1"}


def test_convert_hidden_dump_reports_file_and_line_for_malformed_lines(tmp_path: Path):
    dump = tmp_path / "dump.jsonl"
    good = json.dumps({"prompt_index": 0, "layer": 0, "hidden": [1.0], "text": "t", "criterion_score": 1.0})
    dump.write_text(good + "\n{not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape(f"{dump}:2") + ".*invalid JSON"):
        convert_hidden_state_dump(dump_path=dump, out_path=tmp_path / "records.jsonl")

    _write_dump(dump, [{"prompt_index": 0, "layer": 0, "text": "t", "criterion_score": 1.0}])
    with pytest.raises(ValueError, match=re.escape(f"{dump}:1") + ".*missing hidden"):
        convert_hidden_state_dump(dump_path=dump, out_path=tmp_path / "records.jsonl")

    _write_dump(
        dump,
        [
            {
                "prompt_index": 0,
                "layer": 0,
                "hidden": [[1.0, 2.0], [3.0]],
                "text": "t",
                "criterion_score": 1.0,
            }
        ],
    )
    with pytest.raises(ValueError, match=re.escape(f"{dump}:1") + ".*inconsistent widths"):
        convert_hidden_state_dump(dump_path=dump, out_path=tmp_path / "records.jsonl")

    _write_dump(
        dump,
        [
            {
                "prompt_index": 0,
                "layer": 0,
                "tokens": ["a", "b", "c"],
                "hidden": [[1.0], [2.0]],
                "text": "t",
                "criterion_score": 1.0,
            }
        ],
    )
    with pytest.raises(ValueError, match=re.escape(f"{dump}:1") + ".*tokens has 3 entries"):
        convert_hidden_state_dump(dump_path=dump, out_path=tmp_path / "records.jsonl")


def test_convert_hidden_dump_requires_scores_from_somewhere(tmp_path: Path):
    dump = _write_dump(tmp_path / "dump.jsonl", [{"prompt_index": 0, "layer": 0, "hidden": [1.0], "text": "t"}])

    with pytest.raises(ValueError, match=re.escape(f"{dump}:1") + ".*criterion_score"):
        convert_hidden_state_dump(dump_path=dump, out_path=tmp_path / "records.jsonl")


def test_convert_hidden_dump_validates_prompt_index_and_coverage(tmp_path: Path):
    prompts = tmp_path / "prompts.jsonl"
    _write_prompts(prompts, [{"prompt_id": "p", "text": "t", "criterion_score": 1.0}])

    dump = _write_dump(tmp_path / "dump.jsonl", [{"prompt_index": 5, "layer": 0, "hidden": [1.0]}])
    with pytest.raises(ValueError, match=re.escape(f"{dump}:1") + ".*prompt_index 5"):
        convert_hidden_state_dump(dump_path=dump, out_path=tmp_path / "records.jsonl", prompts_path=prompts)

    _write_prompts(
        prompts,
        [
            {"prompt_id": "p0", "text": "t0", "criterion_score": 1.0},
            {"prompt_id": "p1", "text": "t1", "criterion_score": 0.0},
        ],
    )
    dump = _write_dump(
        tmp_path / "dump.jsonl",
        [
            {"prompt_index": 0, "layer": 0, "hidden": [1.0]},
            {"prompt_index": 0, "layer": 2, "hidden": [1.0]},
            {"prompt_index": 1, "layer": 0, "hidden": [0.0]},
        ],
    )
    with pytest.raises(ValueError, match="prompt_index 1 has no line for layer 2"):
        convert_hidden_state_dump(dump_path=dump, out_path=tmp_path / "records.jsonl", prompts_path=prompts)

    dump = _write_dump(
        tmp_path / "dump.jsonl",
        [
            {"prompt_index": 0, "layer": 0, "hidden": [1.0]},
            {"prompt_index": 0, "layer": 0, "hidden": [2.0]},
        ],
    )
    with pytest.raises(ValueError, match=re.escape(f"{dump}:2") + ".*duplicate dump line"):
        convert_hidden_state_dump(dump_path=dump, out_path=tmp_path / "records.jsonl", prompts_path=prompts)


def test_parser_defaults_and_convert_from_args(tmp_path: Path):
    parser = build_gguf_export_parser()
    args = parser.parse_args(["--model", "m.gguf", "--dataset", "d.jsonl", "--out", "r.jsonl"])
    assert args.n_ctx == 2048
    assert args.top_features == 64
    assert args.threads is None
    assert args.json is False

    dump = _write_dump(
        tmp_path / "dump.jsonl",
        [{"prompt_index": 0, "layer": 1, "hidden": [1.0, 2.0], "text": "t", "criterion_score": 1.0}],
    )
    convert_parser = build_hidden_dump_convert_parser()
    convert_args = convert_parser.parse_args(
        [
            "--dump",
            str(dump),
            "--out",
            str(tmp_path / "records.jsonl"),
            "--features-per-layer",
            "0",
            "--model-name",
            "m",
        ]
    )
    out = run_hidden_dump_convert_from_args(convert_args)
    rows = _read_rows(out)
    # --features-per-layer 0 means "keep all dimensions".
    assert {feature["feature_id"] for feature in rows[0]["features"]} == {"L1:D0", "L1:D1"}
    assert rows[0]["model"] == "m"
