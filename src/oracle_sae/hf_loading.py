from __future__ import annotations

import argparse
import json
from typing import Any


MODEL_CLASS_CHOICES = [
    "auto-causal-lm",
    "auto-image-text-to-text",
    "gemma4-conditional",
]


def add_hf_loading_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model-class",
        choices=MODEL_CLASS_CHOICES,
        default="auto-causal-lm",
        help="Transformers model class to load.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Optional dtype passed to from_pretrained.",
    )
    parser.add_argument(
        "--device-map",
        help="Optional device_map passed to from_pretrained, e.g. auto.",
    )
    parser.add_argument(
        "--model-kwargs-json",
        help="Extra JSON object passed to model from_pretrained.",
    )
    parser.add_argument(
        "--tokenizer-kwargs-json",
        help="Extra JSON object passed to tokenizer from_pretrained.",
    )


def hf_loading_options_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model_class": args.model_class,
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
        "torch_dtype": args.torch_dtype,
        "device_map": args.device_map,
        "model_kwargs": parse_json_object(args.model_kwargs_json, "--model-kwargs-json"),
        "tokenizer_kwargs": parse_json_object(args.tokenizer_kwargs_json, "--tokenizer-kwargs-json"),
    }


def parse_json_object(value: str | None, label: str) -> dict[str, Any]:
    if not value:
        return {}
    try:
        data = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} must be a JSON object: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{label} must be a JSON object")
    return data


def load_hf_text_model(
    *,
    transformers: Any,
    torch: Any,
    model_name: str,
    device: str,
    model_class: str = "auto-causal-lm",
    trust_remote_code: bool = False,
    local_files_only: bool = False,
    torch_dtype: str | None = None,
    device_map: str | None = None,
    model_kwargs: dict[str, Any] | None = None,
    tokenizer_kwargs: dict[str, Any] | None = None,
) -> tuple[Any, Any, str]:
    tokenizer = load_hf_tokenizer(
        transformers=transformers,
        model_name=model_name,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        tokenizer_kwargs=tokenizer_kwargs or {},
    )
    model_kwargs = dict(model_kwargs or {})
    if trust_remote_code:
        model_kwargs["trust_remote_code"] = True
    if local_files_only:
        model_kwargs["local_files_only"] = True
    resolved_dtype = _resolve_torch_dtype(torch, torch_dtype) if torch_dtype else None
    if resolved_dtype is not None:
        model_kwargs["torch_dtype"] = resolved_dtype
    if device_map:
        model_kwargs["device_map"] = device_map
    model_cls = _model_class(transformers, model_class)
    try:
        model = model_cls.from_pretrained(model_name, **model_kwargs)
    except TypeError:
        if resolved_dtype is None or "torch_dtype" not in model_kwargs:
            raise
        model_kwargs = dict(model_kwargs)
        model_kwargs.pop("torch_dtype")
        model_kwargs["dtype"] = resolved_dtype
        model = model_cls.from_pretrained(model_name, **model_kwargs)
    if device_map is None:
        model.to(device)
        runtime_device = device
    else:
        runtime_device = _model_input_device(model, fallback=device)
    model.eval()
    _ensure_pad_token(tokenizer)
    return tokenizer, model, runtime_device


def load_hf_tokenizer(
    *,
    transformers: Any,
    model_name: str,
    trust_remote_code: bool,
    local_files_only: bool,
    tokenizer_kwargs: dict[str, Any],
) -> Any:
    kwargs = dict(tokenizer_kwargs)
    if trust_remote_code:
        kwargs["trust_remote_code"] = True
    if local_files_only:
        kwargs["local_files_only"] = True
    try:
        return transformers.AutoTokenizer.from_pretrained(model_name, **kwargs)
    except Exception:
        processor_cls = getattr(transformers, "AutoProcessor", None)
        if processor_cls is None:
            raise
        processor = processor_cls.from_pretrained(model_name, **kwargs)
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise
        return tokenizer


def _model_class(transformers: Any, model_class: str) -> Any:
    if model_class == "auto-causal-lm":
        return transformers.AutoModelForCausalLM
    if model_class == "auto-image-text-to-text":
        model_cls = getattr(transformers, "AutoModelForImageTextToText", None)
        if model_cls is None:
            raise RuntimeError("This transformers version does not expose AutoModelForImageTextToText")
        return model_cls
    if model_class == "gemma4-conditional":
        model_cls = getattr(transformers, "Gemma4ForConditionalGeneration", None)
        if model_cls is None:
            raise RuntimeError("This transformers version does not expose Gemma4ForConditionalGeneration")
        return model_cls
    raise ValueError(f"unknown model class: {model_class}")


def _resolve_torch_dtype(torch: Any, dtype: str) -> Any:
    if dtype == "auto":
        return "auto"
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype]


def _model_input_device(model: Any, *, fallback: str) -> str:
    device = getattr(model, "device", None)
    if device is not None:
        return str(device)
    try:
        return str(next(model.parameters()).device)
    except Exception:
        return fallback


def _ensure_pad_token(tokenizer: Any) -> None:
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
