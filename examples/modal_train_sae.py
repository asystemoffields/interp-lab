from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import modal

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oracle_sae.modal_workflows import (  # noqa: E402
    ModalSaeWorkflow,
    build_modal_sae_commands,
    expected_modal_sae_outputs,
    summarize_modal_result,
)


APP_NAME = "interp-lab-sae-train"
GPU = os.environ.get("INTERP_LAB_MODAL_GPU", "A10G")

app = modal.App(APP_NAME)
cache_volume = modal.Volume.from_name("interp-lab-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.0",
        "transformers>=5.5.0",
        "accelerate>=1.0",
        "safetensors>=0.4",
        "huggingface-hub>=1.0",
        "PyYAML>=6.0",
    )
    .env(
        {
            "PYTHONPATH": "/app/src",
            "HF_HOME": "/cache/huggingface",
            "HF_HUB_CACHE": "/cache/huggingface/hub",
            "TRANSFORMERS_CACHE": "/cache/huggingface/transformers",
        }
    )
    .workdir("/app")
    .add_local_dir(ROOT / "src", remote_path="/app/src")
)


@app.function(
    image=image,
    gpu=GPU,
    timeout=2 * 60 * 60,
    volumes={"/cache": cache_volume},
)
def run_workflow(config_data: dict, dataset_text: str) -> dict:
    causal_dataset_text = str(config_data.pop("_causal_dataset_text", dataset_text))
    config = ModalSaeWorkflow(**config_data)
    work_dir = Path(config.work_dir)
    dataset_path = Path(config.dataset_path)
    causal_dataset_path = Path(config.causal_dataset_path)
    work_dir.mkdir(parents=True, exist_ok=True)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    causal_dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text(dataset_text, encoding="utf-8")
    causal_dataset_path.write_text(causal_dataset_text, encoding="utf-8")

    logs: list[dict[str, str | int | list[str]]] = []
    for command in build_modal_sae_commands(config):
        completed = subprocess.run(
            command,
            cwd="/app",
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        logs.append(
            {
                "command": command,
                "returncode": completed.returncode,
                "output": completed.stdout,
            }
        )
        if completed.returncode != 0:
            return {"ok": False, "logs": logs, "files": {}}

    files = {}
    for relative in expected_modal_sae_outputs(config):
        path = work_dir / relative
        if path.exists():
            files[relative] = path.read_text(encoding="utf-8")
    cache_volume.commit()
    return {"ok": True, "logs": logs, "files": files}


@app.local_entrypoint()
def main(
    dataset: str = "examples/gemma4_code_prompts.jsonl",
    causal_dataset: str = "examples/gemma4_code_prompts.jsonl",
    out_dir: str = "reports/modal-sae",
    model: str = "HuggingFaceTB/SmolLM3-3B",
    criterion: str = "code-oriented completions",
    layers: str = "12,24",
    preset: str = "production",
    latent_dim: int = 512,
    epochs: int = 40,
    batch_size: int = 128,
    sparsity: str = "",
    l1: float = 0.0,
    lr: float = 0.0,
    max_records: int = 512,
    max_length: int = 96,
    top_k: int = 16,
    top_k_features: int = 32,
    report_top_k: int = 8,
    causal_top_k: int = 8,
    causal_strength_sweep: str = "-2,2,4",
    target_token: str = "auto",
    device: str = "cuda",
    torch_dtype: str = "bfloat16",
    model_class: str = "auto-causal-lm",
    trust_remote_code: bool = False,
) -> None:
    dataset_path = Path(dataset)
    dataset_text = dataset_path.read_text(encoding="utf-8")
    causal_dataset_text = Path(causal_dataset).read_text(encoding="utf-8")
    output_root = Path(out_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    config = ModalSaeWorkflow(
        model=model,
        criterion=criterion,
        layers=layers,
        preset=preset,
        latent_dim=latent_dim,
        epochs=epochs,
        batch_size=batch_size,
        sparsity=sparsity,
        l1=l1,
        lr=lr,
        max_records=max_records,
        max_length=max_length,
        top_k=top_k,
        top_k_features=top_k_features,
        report_top_k=report_top_k,
        causal_top_k=causal_top_k,
        causal_strength_sweep=causal_strength_sweep,
        target_token=target_token,
        device=device,
        torch_dtype=torch_dtype,
        model_class=model_class,
        trust_remote_code=trust_remote_code,
    )
    config_data = dict(config.__dict__)
    config_data["_causal_dataset_text"] = causal_dataset_text
    result = run_workflow.remote(config_data, dataset_text)
    for relative, content in result.get("files", {}).items():
        path = output_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    (output_root / "modal-result.json").write_text(
        _json_dump(summarize_modal_result(result)),
        encoding="utf-8",
    )
    if not result.get("ok"):
        raise SystemExit(f"Modal SAE workflow failed; see {output_root / 'modal-result.json'}")
    print(f"Wrote Modal SAE results to {output_root}")


def _json_dump(value: object) -> str:
    import json

    return json.dumps(value, indent=2, sort_keys=True)
