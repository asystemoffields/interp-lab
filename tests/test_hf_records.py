import contextlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from interp_lab.hf_records import (
    PromptRecord,
    build_prompt_dataset,
    build_prompt_records,
    export_hf_activation_records,
    load_prompt_records,
    parse_layers,
    prepare_sae_prompt_datasets,
    split_prompt_record_indexes,
)
from interp_lab.hf_contrast import (
    _contrast_direction,
    _register_gpt2_steering,
    _select_best_strength,
    parse_strength_sweep,
)
from interp_lab.hf_interventions import (
    _register_gpt2_hidden_ablations,
    _target_token_ids,
    append_hf_group_activation_record,
    parse_target_tokens,
    resolve_target_token_ids,
    target_token_strategy,
)
from interp_lab.nnsight_records import _layer_for_path, export_nnsight_activation_records
from interp_lab.transformerlens_records import _layer_for_hook, export_transformerlens_activation_records


def test_parse_layers_supports_ranges():
    assert parse_layers("0,2,4-6") == [0, 2, 4, 5, 6]
    assert parse_layers("all") == "all"
    assert parse_layers("*") == "all"
    assert parse_layers(None) is None
    with pytest.raises(ValueError):
        parse_layers("3-1")


def test_load_prompt_records(tmp_path: Path):
    path = tmp_path / "prompts.jsonl"
    path.write_text(
        json.dumps({"prompt_id": "p1", "text": "hello", "criterion_score": 1.0}) + "\n",
        encoding="utf-8",
    )

    records = load_prompt_records(path)

    assert records[0].prompt_id == "p1"
    assert records[0].criterion_score == 1.0


def test_build_prompt_dataset_from_user_written_files(tmp_path: Path):
    positive = tmp_path / "positive.txt"
    positive.write_text(
        "Write a Python function that\n\nReturn JSON with status\n",
        encoding="utf-8",
    )
    negative = tmp_path / "negative.txt"
    negative.write_text("The museum opened on Tuesday\n\nA good dinner menu includes", encoding="utf-8")
    out = tmp_path / "prompts.jsonl"

    summary = build_prompt_dataset(
        positive_paths=[positive],
        negative_paths=[negative],
        positive_prompts=["Inline positive"],
        negative_prompts=["Inline negative"],
        out_path=out,
        id_prefix="criterion",
    )
    records = load_prompt_records(out)

    assert summary.record_count == 6
    assert summary.positive_count == 3
    assert summary.negative_count == 3
    assert [record.prompt_id for record in records] == [
        "criterion-positive-001",
        "criterion-positive-002",
        "criterion-positive-003",
        "criterion-negative-004",
        "criterion-negative-005",
        "criterion-negative-006",
    ]
    assert records[0].text == "Inline positive"
    assert records[-1].criterion_score == 0.0


def test_build_prompt_records_supports_delimited_multiline_prompts(tmp_path: Path):
    positive = tmp_path / "positive.txt"
    positive.write_text("User: A\nAssistant:\n---\nUser: B\nAssistant:", encoding="utf-8")

    records = build_prompt_records(
        positive_paths=[positive],
        split="lines",
        delimiter="\n---\n",
        id_prefix="chat",
    )

    assert [record.text for record in records] == ["User: A\nAssistant:", "User: B\nAssistant:"]


def test_prepare_sae_prompt_datasets_stratifies_and_keeps_duplicates_together(tmp_path: Path):
    dataset = tmp_path / "prompts.jsonl"
    rows = []
    for index in range(1, 7):
        rows.append(
            {
                "prompt_id": f"pos-{index}",
                "text": "duplicate tool call prompt" if index in {1, 2} else f"tool call prompt {index}",
                "criterion_score": 1.0,
            }
        )
    for index in range(1, 7):
        rows.append(
            {
                "prompt_id": f"neg-{index}",
                "text": f"ordinary answer prompt {index}",
                "criterion_score": 0.0,
            }
        )
    dataset.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    summary = prepare_sae_prompt_datasets(
        dataset_path=dataset,
        out_dir=tmp_path / "pack",
        train_ratio=0.5,
        causal_ratio=0.25,
        validation_ratio=0.25,
        seed="unit",
        latent_dim=64,
        max_length=8,
    )

    assert summary.train_path.exists()
    assert summary.causal_path.exists()
    assert summary.validation_path.exists()
    manifest = json.loads(summary.manifest_path.read_text(encoding="utf-8"))
    assert manifest["format"] == "interp-lab.sae_prompt_pack.v1"
    assert manifest["counts"]["total"]["record_count"] == 12
    assert manifest["counts"]["splits"]["causal"]["positive_count"] > 0
    assert manifest["counts"]["splits"]["causal"]["negative_count"] > 0
    assert any("Duplicate prompt text" in advisory for advisory in manifest["advisories"])

    split_texts = {
        split: {record.text for record in load_prompt_records(path)}
        for split, path in {
            "train": summary.train_path,
            "causal": summary.causal_path,
            "validation": summary.validation_path,
        }.items()
    }
    assert split_texts["train"].isdisjoint(split_texts["causal"])
    assert split_texts["train"].isdisjoint(split_texts["validation"])
    assert split_texts["causal"].isdisjoint(split_texts["validation"])


def test_parse_target_tokens_adds_leading_spaces():
    assert parse_target_tokens(["meters, feet", " kilogram"]) == [
        " meters",
        " feet",
        " kilogram",
    ]
    assert parse_target_tokens(None) is None


def test_parse_target_tokens_supports_raw_prefix():
    assert parse_target_tokens(["raw:def, return", "space:class"]) == [
        "def",
        " return",
        " class",
    ]


def test_parse_target_tokens_supports_auto():
    assert parse_target_tokens(["auto"]) == ["auto"]


def test_target_token_strategy_tracks_default_auto_and_explicit():
    assert target_token_strategy(None) == "default"
    assert target_token_strategy(["auto"]) == "auto"
    assert target_token_strategy([" Python"]) == "explicit"


def test_parse_strength_sweep():
    assert parse_strength_sweep("3, 10, -30") == [3.0, 10.0, -30.0]
    assert parse_strength_sweep(None) is None


def test_split_prompt_record_indexes():
    prompts = [
        PromptRecord(prompt_id="positive", text="a", criterion_score=1.0),
        PromptRecord(prompt_id="negative", text="b", criterion_score=0.0),
    ]

    assert split_prompt_record_indexes(prompts) == ({0}, {1})


def test_select_best_strength_uses_specificity():
    selected, summary = _select_best_strength(
        {
            3.0: [
                {"baseline_score": 0.1, "intervention_score": 0.2},
                {"baseline_score": 0.1, "intervention_score": 0.0},
            ],
            10.0: [
                {"baseline_score": 0.1, "intervention_score": 0.3},
                {"baseline_score": 0.1, "intervention_score": 0.2},
            ],
        },
        {
            3.0: [0.0],
            10.0: [0.2],
        },
    )

    assert selected == 3.0
    assert summary == [
        {
            "steer_strength": 3.0,
            "mean_directed_effect": 0.0,
            "mean_side_effect": 0.0,
            "specificity": 0.0,
        },
        {
            "steer_strength": 10.0,
            "mean_directed_effect": 0.15,
            "mean_side_effect": 0.2,
            "specificity": -0.05,
        },
    ]


def test_append_hf_group_activation_record_orients_by_signed_effect(tmp_path: Path):
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "model": "m",
                "criterion": {"text": "criterion"},
                "cards": [
                    {
                        "feature_id": "L1:D1",
                        "model": "m",
                        "layer": 1,
                        "label": "positive",
                        "explanation": "",
                        "importance": 1,
                        "association": 1,
                        "specificity": 1,
                        "causal_effect": 1,
                        "stability": 1,
                        "examples": [],
                        "source": "hf-hidden-state",
                        "fingerprint": {
                            "feature_id": "L1:D1",
                            "model": "m",
                            "layer": 1,
                            "text": "",
                            "text_vector": [],
                            "activation_signature": [],
                            "decoder_signature": [],
                            "causal_vector": [],
                        },
                        "metadata": {},
                        "causal_effects": {"signed_association": 1},
                    },
                    {
                        "feature_id": "L1:D2",
                        "model": "m",
                        "layer": 1,
                        "label": "negative",
                        "explanation": "",
                        "importance": 1,
                        "association": 1,
                        "specificity": 1,
                        "causal_effect": 1,
                        "stability": 1,
                        "examples": [],
                        "source": "hf-hidden-state",
                        "fingerprint": {
                            "feature_id": "L1:D2",
                            "model": "m",
                            "layer": 1,
                            "text": "",
                            "text_vector": [],
                            "activation_signature": [],
                            "decoder_signature": [],
                            "causal_vector": [],
                        },
                        "metadata": {},
                        "causal_effects": {"signed_association": -1},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    records = tmp_path / "records.jsonl"
    records.write_text(
        json.dumps(
            {
                "model": "m",
                "prompt_id": "p",
                "text": "text",
                "criterion_score": 1,
                "features": [
                    {"feature_id": "L1:D1", "activation": 2},
                    {"feature_id": "L1:D2", "activation": -4},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "grouped.jsonl"

    group_id = append_hf_group_activation_record(
        records_path=records,
        report_path=report,
        out_path=out,
        group_top_k=2,
    )

    row = json.loads(out.read_text(encoding="utf-8"))
    group_feature = row["features"][-1]
    assert group_feature["feature_id"] == group_id
    assert group_feature["activation"] == 3


def test_contrast_direction_points_from_negative_to_positive():
    direction = _contrast_direction(
        vectors=[[2.0, 0.0], [4.0, 0.0], [-2.0, 0.0], [-4.0, 0.0]],
        scores=[1.0, 1.0, 0.0, 0.0],
    )

    assert direction == [1.0, 0.0]


def test_final_hidden_layer_interventions_hook_final_layer_norm():
    model = _FakeGpt2Model(block_count=2)

    handle = _register_gpt2_hidden_ablations(model, [(2, 7, 0.0)])

    assert len(model.transformer.h[1].hooks) == 0
    assert len(model.transformer.ln_f.hooks) == 1
    handle.remove()
    assert model.transformer.ln_f.removed == 1


def test_final_hidden_layer_steering_hooks_final_layer_norm():
    model = _FakeGpt2Model(block_count=2)

    handle = _register_gpt2_steering(model, 2, direction=_FakeDirection(), strength=1.0)

    assert len(model.transformer.h[1].hooks) == 0
    assert len(model.transformer.ln_f.hooks) == 1
    handle.remove()
    assert model.transformer.ln_f.removed == 1


def test_target_token_ids_score_the_first_content_piece():
    tokenizer = _MultiPieceTokenizer()

    # " centimeters" splits into [" cent", "imeters"]; the next-token behavior
    # score can only observe the FIRST piece.
    assert _target_token_ids(tokenizer, [" centimeters"]) == [101]
    # SentencePiece-style lone-space first piece is skipped for the content piece.
    assert _target_token_ids(tokenizer, [" metres"]) == [202]
    # Single-piece targets are unchanged.
    assert _target_token_ids(tokenizer, [" meters"]) == [7]


def test_resolve_target_token_ids_records_id_to_token_mapping():
    tokenizer = _MultiPieceTokenizer()

    target_ids, resolved_tokens, token_map = resolve_target_token_ids(
        model=None,
        tokenizer=tokenizer,
        prompts=[],
        target_tokens=[" centimeters", " meters"],
        device="cpu",
        max_length=8,
    )

    assert target_ids == [7, 101]
    assert resolved_tokens == [" centimeters", " meters"]
    assert token_map == {"7": " meters", "101": " cent"}


def test_nnsight_and_transformerlens_layers_use_hidden_state_convention():
    # Block i's output is hidden_states[i + 1] in HF terms.
    assert _layer_for_path("transformer.h[6].output[0]") == 7
    assert _layer_for_path("model.layers[2].output") == 3
    assert _layer_for_path("lm_head.output") is None
    assert _layer_for_hook("blocks.6.hook_resid_post") == 7
    assert _layer_for_hook("blocks.6.hook_resid_pre") == 6
    assert _layer_for_hook("hook_embed") is None


def test_export_nnsight_records_normalizes_layers_to_hidden_state_convention(tmp_path: Path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    monkeypatch.setitem(sys.modules, "nnsight", types.ModuleType("nnsight"))
    dataset = _write_prompts(tmp_path)
    out = tmp_path / "nnsight.jsonl"
    model = _FakeNnsightModel([[[2.0, 0.0]], [[0.0, 1.0]]])

    export_nnsight_activation_records(
        model_name="m",
        dataset_path=dataset,
        out_path=out,
        activation_paths=["transformer.h[0].output[0]"],
        features_per_path=1,
        model_factory=lambda name: model,
    )

    row = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    feature = row["features"][0]
    assert feature["layer"] == 1
    metadata = row["feature_metadata"][feature["feature_id"]]
    assert metadata["layer"] == 1
    assert metadata["layer_convention"] == "hidden_state_index"


def test_export_transformerlens_records_normalizes_layers_to_hidden_state_convention(tmp_path: Path, monkeypatch):
    fake_torch = types.ModuleType("torch")
    fake_torch.no_grad = contextlib.nullcontext
    fake_tl = types.ModuleType("transformer_lens")
    fake_tl.HookedTransformer = _FakeHookedTransformer
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformer_lens", fake_tl)
    dataset = _write_prompts(tmp_path)
    out = tmp_path / "tl.jsonl"

    export_transformerlens_activation_records(
        model_name="m",
        dataset_path=dataset,
        out_path=out,
        hook_names=["blocks.1.hook_resid_post"],
        features_per_hook=1,
    )

    row = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    feature = row["features"][0]
    assert feature["layer"] == 2
    metadata = row["feature_metadata"][feature["feature_id"]]
    assert metadata["layer"] == 2
    assert metadata["layer_convention"] == "hidden_state_index"


def test_export_hf_activation_records_marks_layer_convention(tmp_path: Path, monkeypatch):
    fake_torch = SimpleNamespace(no_grad=contextlib.nullcontext)
    monkeypatch.setattr("interp_lab.hf_records._optional_import", lambda name, message: fake_torch)
    monkeypatch.setattr(
        "interp_lab.hf_records.load_hf_text_model",
        lambda **kwargs: (_FakeHfTokenizer(), _FakeHfModel(), "cpu"),
    )
    dataset = _write_prompts(tmp_path)
    out = tmp_path / "hf.jsonl"

    export_hf_activation_records(
        model_name="m",
        dataset_path=dataset,
        out_path=out,
        layers=[2],
        features_per_layer=1,
    )

    row = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    feature = row["features"][0]
    assert feature["feature_id"].startswith("L2:D")
    metadata = row["feature_metadata"][feature["feature_id"]]
    assert metadata["layer"] == 2
    assert metadata["layer_convention"] == "hidden_state_index"


def _write_prompts(tmp_path: Path) -> Path:
    dataset = tmp_path / "prompts.jsonl"
    dataset.write_text(
        json.dumps({"prompt_id": "pos", "text": "a", "criterion_score": 1.0})
        + "\n"
        + json.dumps({"prompt_id": "neg", "text": "b", "criterion_score": 0.0})
        + "\n",
        encoding="utf-8",
    )
    return dataset


class _MultiPieceTokenizer:
    _pieces = {
        " centimeters": [101, 102],
        " metres": [201, 202],
        " meters": [7],
    }
    _decoded = {7: " meters", 101: " cent", 102: "imeters", 201: " ", 202: "metres"}

    def encode(self, text, add_special_tokens=False):
        return list(self._pieces.get(text, []))

    def decode(self, ids):
        return "".join(self._decoded.get(token_id, "") for token_id in ids)


class _FakeActivation:
    """Minimal (seq, dim) tensor stand-in for nnsight/TL pooling."""

    def __init__(self, rows):
        self._rows = rows

    def detach(self):
        return self

    def cpu(self):
        return self

    @property
    def shape(self):
        return (len(self._rows), len(self._rows[0]))

    def reshape(self, first, second):
        assert second == -1
        return self

    def __getitem__(self, index):
        return _FakeVector(self._rows[index])


class _FakeVector:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return list(self._values)

    def detach(self):
        return self

    def cpu(self):
        return self


class _FakeSaved:
    def __init__(self):
        self.value = None

    def save(self):
        return self


class _FakeNnsightModel:
    def __init__(self, values_per_prompt):
        self._values = list(values_per_prompt)
        self._saved = _FakeSaved()
        self.transformer = SimpleNamespace(h=[SimpleNamespace(output=[self._saved])])

    def trace(self, text, **kwargs):
        self._saved.value = _FakeActivation(self._values.pop(0))
        return contextlib.nullcontext()


class _FakeHookedTransformer:
    def __init__(self):
        self.cfg = SimpleNamespace(n_layers=2)
        self._activations = [_FakeActivation([[2.0, 0.0]]), _FakeActivation([[0.0, 1.0]])]

    @classmethod
    def from_pretrained(cls, model_name, device="cpu"):
        return cls()

    def eval(self):
        return self

    def to_tokens(self, text, prepend_bos=True):
        return text

    def run_with_cache(self, tokens, names_filter=None, remove_batch_dim=False):
        return None, {"blocks.1.hook_resid_post": self._activations.pop(0)}


class _FakeHfTokenizer:
    def __call__(self, text, return_tensors="pt", truncation=True, max_length=128):
        return {"input_ids": _FakeEncodedTensor()}


class _FakeEncodedTensor:
    def to(self, device):
        return self


class _FakeHfModel:
    def __init__(self):
        self._hidden_per_call = [
            [_FakeHidden([[0.0, 0.0]]), _FakeHidden([[1.0, 0.0]]), _FakeHidden([[2.0, 0.0]])],
            [_FakeHidden([[0.0, 0.0]]), _FakeHidden([[0.0, 1.0]]), _FakeHidden([[0.0, 2.0]])],
        ]

    def __call__(self, output_hidden_states=True, use_cache=False, **encoded):
        return SimpleNamespace(hidden_states=self._hidden_per_call.pop(0))


class _FakeHidden:
    """(batch=1, seq, dim) tensor stand-in for hf_records pooling."""

    def __init__(self, rows):
        self._rows = rows

    @property
    def shape(self):
        return (1, len(self._rows), len(self._rows[0]))

    def __getitem__(self, index):
        batch, token_index = index
        assert batch == 0
        return _FakeVector(self._rows[token_index])


class _FakeGpt2Model:
    def __init__(self, block_count: int):
        self.transformer = _FakeTransformer(block_count)


class _FakeTransformer:
    def __init__(self, block_count: int):
        self.h = [_FakeHookable() for _ in range(block_count)]
        self.ln_f = _FakeHookable()


class _FakeHookable:
    def __init__(self):
        self.hooks = []
        self.removed = 0

    def register_forward_hook(self, hook):
        self.hooks.append(hook)
        return _FakeHandle(self)


class _FakeHandle:
    def __init__(self, module: _FakeHookable):
        self.module = module

    def remove(self):
        self.module.removed += 1


class _FakeDirection:
    def to(self, _dtype):
        return self
