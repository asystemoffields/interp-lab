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

from oracle_sae.modal_workflows import (
    ModalGemmaWorkflow,
    build_modal_gemma_commands,
    expected_modal_gemma_outputs,
    summarize_modal_result,
)
from oracle_sae.reporting import load_inspection_report, write_inspection_html


APP_NAME = "interp-lab-gemma4"
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
    timeout=60 * 60,
    volumes={"/cache": cache_volume},
)
def run_workflow(config_data: dict, dataset_text: str) -> dict:
    config = ModalGemmaWorkflow(**config_data)
    work_dir = Path(config.work_dir)
    dataset_path = Path(config.dataset_path)
    work_dir.mkdir(parents=True, exist_ok=True)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text(dataset_text, encoding="utf-8")

    logs: list[dict[str, str | int | list[str]]] = []
    for command in build_modal_gemma_commands(config):
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
            return {
                "ok": False,
                "workflow": config.workflow,
                "logs": logs,
                "files": {},
            }

    files = {}
    for relative in expected_modal_gemma_outputs(config.workflow):
        path = work_dir / relative
        if path.exists():
            files[relative] = path.read_text(encoding="utf-8")
    cache_volume.commit()
    return {
        "ok": True,
        "workflow": config.workflow,
        "logs": logs,
        "files": files,
    }


@app.local_entrypoint()
def main(
    dataset: str = "examples/gemma4_code_prompts.jsonl",
    out_dir: str = "reports/gemma4-modal",
    model: str = "google/gemma-4-E2B-it",
    criterion: str = "code-oriented completions",
    workflow: str = "contrast",
    layer: str = "",
    layers: str = "20,28,35",
    features_per_layer: int = 32,
    top_k: int = 16,
    group_top_k: int = 8,
    target_token: str = "auto",
    strength_sweep: str = "-4,-2,2,4",
    max_length: int = 64,
) -> None:
    dataset_path = Path(dataset)
    dataset_text = dataset_path.read_text(encoding="utf-8")
    output_root = Path(out_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    config = ModalGemmaWorkflow(
        model=model,
        criterion=criterion,
        workflow=workflow,
        layer=layer,
        layers=layers,
        features_per_layer=features_per_layer,
        top_k=top_k,
        group_top_k=group_top_k,
        target_token=target_token,
        strength_sweep=strength_sweep,
        max_length=max_length,
    )
    result = run_workflow.remote(config.__dict__, dataset_text)
    for relative, content in result.get("files", {}).items():
        path = output_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    _write_report_html(output_root)
    (output_root / "modal-result.json").write_text(
        _json_dump(summarize_modal_result(result)),
        encoding="utf-8",
    )
    if not result.get("ok"):
        raise SystemExit(f"Modal workflow failed; see {output_root / 'modal-result.json'}")
    print(f"Wrote Modal {workflow} results to {output_root}")


def _json_dump(value: object) -> str:
    import json

    return json.dumps(value, indent=2, sort_keys=True)


def _write_report_html(output_root: Path) -> None:
    for report_path in output_root.glob("**/report.json"):
        report = load_inspection_report(report_path)
        write_inspection_html(report, report_path.with_suffix(".html"))
