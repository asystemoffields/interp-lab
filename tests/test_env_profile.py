import json
from pathlib import Path

import pytest

from interp_lab.cli import main
from interp_lab.env_profile import build_environment_routing, collect_environment_profile
from interp_lab.scaling import ScalePlan


def test_collect_environment_profile_is_sanitized_and_routable(tmp_path: Path):
    profile = collect_environment_profile(path=tmp_path, env={}, probe_accelerators=False)

    assert profile["schema_version"] == "interp-lab.env_profile.v1"
    assert profile["tool"] == "interp-lab"
    assert profile["disk"]["path"] == str(tmp_path.resolve())
    assert profile["routing"]["suggested_profile"] in {
        "local-cpu",
        "remote-api",
        "single-gpu",
        "cluster",
        "frontier-lab",
    }
    assert profile["environment_flags"]["GOODFIRE_API_KEY"] == {"present": False}
    assert profile["routing"]["options"]
    assert profile["agent_next_actions"]


def test_environment_routing_prefers_detected_gpu():
    profile = _fake_profile(
        gpu_count=1,
        total_gpu_memory=24 * 1024**3,
        max_gpu_memory=24 * 1024**3,
    )

    routing = build_environment_routing(profile)

    assert routing["suggested_profile"] == "single-gpu"
    assert any(option["profile"] == "single-gpu" and option["status"] == "available" for option in routing["options"])


def test_environment_routing_can_suggest_remote_api():
    profile = _fake_profile(
        gpu_count=0,
        total_gpu_memory=0,
        max_gpu_memory=0,
        env_flags={"GOODFIRE_API_KEY": {"present": True}},
    )

    routing = build_environment_routing(profile)

    assert routing["suggested_profile"] == "remote-api"
    assert any(option["profile"] == "remote-api" and option["status"] == "available" for option in routing["options"])


def test_profile_env_cli_writes_json(tmp_path: Path, capsys):
    out = tmp_path / "env-profile.json"

    exit_code = main(["profile-env", "--path", str(tmp_path), "--out", str(out), "--json"])

    printed = json.loads(capsys.readouterr().out)
    written = json.loads(out.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert printed["schema_version"] == "interp-lab.env_profile.v1"
    assert written["schema_version"] == "interp-lab.env_profile.v1"


def test_scale_plan_uses_env_profile_as_advisory_route():
    env_profile = _fake_profile(
        gpu_count=1,
        total_gpu_memory=24 * 1024**3,
        max_gpu_memory=24 * 1024**3,
    )
    env_profile["routing"] = build_environment_routing(env_profile)

    plan = ScalePlan(
        model_params=2_000_000_000,
        tokens=1000,
        d_model=1024,
        selected_layers=1,
        latent_dim=4096,
        dtype="bf16",
        environment_profile=env_profile,
    ).to_dict()

    assert plan["profile"] == "single-gpu"
    assert plan["environment_profile"]["routing"]["suggested_profile"] == "single-gpu"
    assert plan["shard_plan"]["target_shard_size_source"] == "environment profile recommendation"
    assert any(action["id"] == "review_environment_route" for action in plan["agent_next_actions"])


def test_scale_plan_flags_model_weights_that_exceed_local_memory():
    env_profile = _fake_profile(
        gpu_count=0,
        total_gpu_memory=0,
        max_gpu_memory=0,
    )
    env_profile["routing"] = build_environment_routing(env_profile)

    plan = ScalePlan(
        model_params=5_000_000_000,
        tokens=12,
        d_model=1536,
        selected_layers=1,
        latent_dim=512,
        dtype="bf16",
        profile="local-cpu",
        model_weight_bytes=96 * 1024**3,
        environment_profile=env_profile,
    ).to_dict()

    assert any("model weights exceed local available RAM" in item["message"] for item in plan["risk_flags"])


def test_scale_plan_rejects_negative_core_sizes():
    with pytest.raises(ValueError, match="model_params must be positive"):
        ScalePlan(
            model_params=-1,
            tokens=1000,
            d_model=1024,
            selected_layers=1,
            latent_dim=4096,
            dtype="bf16",
        ).to_dict()


def test_scale_plan_env_profile_path_cli(tmp_path: Path):
    env_profile = _fake_profile(
        gpu_count=0,
        total_gpu_memory=0,
        max_gpu_memory=0,
        env_flags={"GOODFIRE_API_KEY": {"present": True}},
    )
    env_profile["routing"] = build_environment_routing(env_profile)
    path = tmp_path / "env-profile.json"
    path.write_text(json.dumps(env_profile), encoding="utf-8")
    plan_path = tmp_path / "scale-plan.json"

    exit_code = main(
        [
            "plan-scale",
            "--model-params",
            "1B",
            "--tokens",
            "1K",
            "--d-model",
            "1024",
            "--env-profile",
            str(path),
            "--out",
            str(plan_path),
            "--json",
        ]
    )

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert plan["profile"] == "remote-api"
    assert plan["environment_profile"]["routing"]["suggested_profile"] == "remote-api"


def _fake_profile(
    *,
    gpu_count: int,
    total_gpu_memory: int,
    max_gpu_memory: int,
    env_flags: dict | None = None,
) -> dict:
    flags = {name: {"present": False} for name in ["GOODFIRE_API_KEY", "NNSIGHT_API_KEY", "INTERP_LAB_PROFILE"]}
    flags.update(env_flags or {})
    return {
        "schema_version": "interp-lab.env_profile.v1",
        "platform": {"system": "TestOS", "machine": "x86_64"},
        "cpu": {"count": 8},
        "memory": {
            "total_bytes": 64 * 1024**3,
            "available_bytes": 48 * 1024**3,
            "total_human": "64.00 GB",
            "available_human": "48.00 GB",
        },
        "disk": {
            "path": "/tmp",
            "free_bytes": 512 * 1024**3,
            "free_human": "512.00 GB",
        },
        "accelerators": [],
        "optional_modules": {
            "torch": {"available": gpu_count > 0},
            "goodfire": {"available": False},
            "nnsight": {"available": False},
        },
        "environment_flags": flags,
        "capabilities": {
            "gpu_count": gpu_count,
            "total_gpu_memory_bytes": total_gpu_memory,
            "max_gpu_memory_bytes": max_gpu_memory,
            "has_cuda": gpu_count > 0,
            "has_mps": False,
            "has_local_accelerator": gpu_count > 0,
            "has_cluster_environment": False,
        },
    }
