from __future__ import annotations

import argparse
import ctypes
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from oracle_sae import __version__


SCHEMA_VERSION = "interp-lab.env_profile.v1"
BYTE_SUFFIXES = ["B", "KB", "MB", "GB", "TB", "PB", "EB"]
PROFILE_ORDER = {
    "local-cpu": 0,
    "remote-api": 1,
    "single-gpu": 2,
    "cluster": 3,
    "frontier-lab": 4,
}
OPTIONAL_MODULES = {
    "torch": "HF activation export and SAE training",
    "transformers": "Hugging Face model adapters",
    "sae_lens": "SAE Lens adapter",
    "transformer_lens": "TransformerLens activation export",
    "nnsight": "NNsight activation export",
    "goodfire": "Goodfire feature adapter",
    "huggingface_hub": "Hugging Face artifact publishing",
}
PACKAGE_NAMES = {
    "sae_lens": "sae-lens",
    "transformer_lens": "transformer-lens",
    "huggingface_hub": "huggingface-hub",
}
ENV_FLAGS = [
    "GOODFIRE_API_KEY",
    "HF_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "NNSIGHT_API_KEY",
    "CUDA_VISIBLE_DEVICES",
    "SLURM_JOB_ID",
    "SLURM_CPUS_ON_NODE",
    "SLURM_GPUS",
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "PBS_JOBID",
    "LSB_JOBID",
    "INTERP_LAB_PROFILE",
]


def collect_environment_profile(
    path: str | Path = ".",
    *,
    env: Mapping[str, str] | None = None,
    probe_accelerators: bool = True,
) -> dict[str, Any]:
    """Return a sanitized capability profile for the current environment."""
    env = os.environ if env is None else env
    probe_path = _existing_probe_path(Path(path))
    disk_usage = shutil.disk_usage(probe_path)
    optional_modules = _optional_modules()
    accelerators = _accelerators(probe_accelerators=probe_accelerators)
    profile: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "tool": "interp-lab",
        "version": __version__,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "platform": {
            "python": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "cpu": {
            "count": os.cpu_count() or 1,
        },
        "memory": _memory_info(),
        "disk": {
            "path": str(probe_path),
            "total_bytes": disk_usage.total,
            "used_bytes": disk_usage.used,
            "free_bytes": disk_usage.free,
            "total_human": _format_bytes(disk_usage.total),
            "free_human": _format_bytes(disk_usage.free),
        },
        "accelerators": accelerators,
        "optional_modules": optional_modules,
        "environment_flags": _environment_flags(env),
    }
    profile["capabilities"] = _capabilities(profile)
    profile["routing"] = build_environment_routing(profile)
    profile["risk_flags"] = environment_risk_flags(profile)
    profile["agent_next_actions"] = environment_agent_next_actions(profile)
    return profile


def load_environment_profile(path: str | Path) -> dict[str, Any]:
    profile = json.loads(Path(path).read_text(encoding="utf-8"))
    if profile.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"expected {SCHEMA_VERSION}, got {profile.get('schema_version')!r}")
    return profile


def build_environment_profile_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile local compute, storage, and routing options.")
    parser.add_argument(
        "--path",
        default=".",
        help="Path whose filesystem capacity should be inspected.",
    )
    parser.add_argument("--out", help="Optional JSON path to write the profile.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser


def run_environment_profile_from_args(args: argparse.Namespace) -> dict[str, Any]:
    profile = collect_environment_profile(path=args.path)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(profile, indent=2, sort_keys=True), encoding="utf-8")
    if args.json:
        print(json.dumps(profile, indent=2, sort_keys=True))
    else:
        print(render_environment_profile(profile))
    return profile


def render_environment_profile(profile: dict[str, Any]) -> str:
    summary = environment_summary(profile)
    routing = profile["routing"]
    lines = [
        "interp-lab environment profile",
        "",
        f"Looks like: {summary}",
        f"Suggested starting route: {routing['suggested_profile']} ({routing['suggested_reason']})",
        "Override any suggestion with `interp-lab plan-scale --profile <profile>`.",
        "",
        "Route options:",
    ]
    for option in routing["options"]:
        lines.append(
            f"- {option['profile']} [{option['status']}]: {option['reason']}"
        )
    if profile.get("risk_flags"):
        lines.extend(["", "Alerts:"])
        for item in profile["risk_flags"]:
            lines.append(f"- [{item['level']}] {item['message']} Next: {item['next_step']}")
    lines.extend(["", "Agent next actions:"])
    for item in profile["agent_next_actions"]:
        command = f" Command: {item['command']}" if item.get("command") else ""
        lines.append(f"- {item['id']}: {item['title']}.{command}")
    return "\n".join(lines)


def build_environment_routing(profile: dict[str, Any]) -> dict[str, Any]:
    capabilities = profile.get("capabilities", {})
    env_flags = profile.get("environment_flags", {})
    modules = profile.get("optional_modules", {})
    gpu_count = int(capabilities.get("gpu_count") or 0)
    total_gpu_memory = int(capabilities.get("total_gpu_memory_bytes") or 0)
    max_gpu_memory = int(capabilities.get("max_gpu_memory_bytes") or 0)
    has_mps = bool(capabilities.get("has_mps"))
    has_cluster_env = bool(capabilities.get("has_cluster_environment"))
    remote_ready = _flag_present(env_flags, "GOODFIRE_API_KEY") or _flag_present(env_flags, "NNSIGHT_API_KEY")
    remote_possible = remote_ready or _module_available(modules, "goodfire") or _module_available(modules, "nnsight")
    explicit_profile = _explicit_env_profile(env_flags)

    options = [
        {
            "id": "local_cpu",
            "profile": "local-cpu",
            "status": "available",
            "reason": "CPU execution and small evidence files are available on this machine.",
            "best_for": "smoke tests, records-only inspection, small local pilots",
        },
        {
            "id": "single_gpu",
            "profile": "single-gpu",
            "status": "available" if gpu_count or has_mps else "candidate",
            "reason": _gpu_route_reason(gpu_count, max_gpu_memory, has_mps),
            "best_for": "local activation export, SAE training, medium pilots",
        },
        {
            "id": "cluster",
            "profile": "cluster",
            "status": "available" if has_cluster_env or gpu_count >= 4 or total_gpu_memory >= 160 * 1024**3 else "candidate",
            "reason": _cluster_route_reason(has_cluster_env, gpu_count, total_gpu_memory),
            "best_for": "distributed activation harvests and larger SAE training runs",
        },
        {
            "id": "remote_api",
            "profile": "remote-api",
            "status": "available" if remote_ready else "candidate",
            "reason": _remote_route_reason(remote_ready, remote_possible),
            "best_for": "models served behind Goodfire, NNsight, hosted inference, or internal APIs",
        },
        {
            "id": "frontier_lab",
            "profile": "frontier-lab",
            "status": "candidate",
            "reason": "Use colocated harvesting when the model is too large to move to this environment.",
            "best_for": "1T+ models, lab clusters, sharded activation services",
        },
    ]
    if explicit_profile:
        suggested = explicit_profile
        reason = "selected by INTERP_LAB_PROFILE"
    elif has_cluster_env or gpu_count >= 4 or total_gpu_memory >= 160 * 1024**3:
        suggested = "cluster"
        reason = "cluster signals or multi-GPU capacity were detected"
    elif gpu_count or has_mps:
        suggested = "single-gpu"
        reason = "a local accelerator was detected"
    elif remote_ready:
        suggested = "remote-api"
        reason = "remote API credentials were detected"
    else:
        suggested = "local-cpu"
        reason = "CPU-only local execution was detected"
    return {
        "suggested_profile": suggested,
        "suggested_reason": reason,
        "recommended_target_shard_size_bytes": recommended_target_shard_size(profile),
        "recommended_target_shard_size_human": _format_bytes(recommended_target_shard_size(profile)),
        "options": options,
        "override_examples": [
            "interp-lab plan-scale --profile local-cpu ...",
            "interp-lab plan-scale --profile single-gpu ...",
            "interp-lab plan-scale --env-profile other-machine.json ...",
        ],
    }


def recommended_target_shard_size(profile: dict[str, Any]) -> int:
    memory = profile.get("memory", {})
    disk = profile.get("disk", {})
    capabilities = profile.get("capabilities", {})
    available_memory = int(memory.get("available_bytes") or memory.get("total_bytes") or 0)
    free_disk = int(disk.get("free_bytes") or 0)
    max_gpu_memory = int(capabilities.get("max_gpu_memory_bytes") or 0)
    candidates = [512 * 1024**2]
    if available_memory:
        candidates.append(max(256 * 1024**2, available_memory // 4))
    if free_disk:
        candidates.append(max(256 * 1024**2, free_disk // 64))
    if max_gpu_memory:
        candidates.append(max(512 * 1024**2, max_gpu_memory // 4))
    return int(min(max(candidates), 64 * 1024**3))


def environment_summary(profile: dict[str, Any]) -> str:
    platform_info = profile.get("platform", {})
    cpu = profile.get("cpu", {})
    memory = profile.get("memory", {})
    disk = profile.get("disk", {})
    accelerators = profile.get("accelerators", [])
    parts = [
        f"{platform_info.get('system', 'Unknown')} {platform_info.get('machine', '').strip()}".strip(),
        f"{cpu.get('count', 0)} CPU cores",
    ]
    if memory.get("total_bytes"):
        parts.append(f"{memory['total_human']} RAM")
    if disk.get("free_bytes"):
        parts.append(f"{disk['free_human']} free at {disk.get('path')}")
    if accelerators:
        parts.append(_accelerator_summary(accelerators))
    else:
        parts.append("no local accelerator detected")
    return ", ".join(parts)


def environment_risk_flags(profile: dict[str, Any]) -> list[dict[str, str]]:
    flags: list[dict[str, str]] = []
    modules = profile.get("optional_modules", {})
    capabilities = profile.get("capabilities", {})
    disk = profile.get("disk", {})
    if not _module_available(modules, "torch"):
        flags.append(
            {
                "level": "info",
                "message": "Torch is not installed in this environment.",
                "next_step": 'Install "interp-lab[hf,train]" before local model export or SAE training.',
            }
        )
    if not capabilities.get("has_local_accelerator"):
        flags.append(
            {
                "level": "info",
                "message": "Local accelerator support was not detected.",
                "next_step": "Use records-only workflows, a remote route, or profile another machine.",
            }
        )
    free_disk = int(disk.get("free_bytes") or 0)
    if free_disk and free_disk < 10 * 1024**3:
        flags.append(
            {
                "level": "medium",
                "message": "The inspected filesystem has less than 10 GB free.",
                "next_step": "Use a larger artifact path or write sharded records to remote storage.",
            }
        )
    return flags


def environment_agent_next_actions(profile: dict[str, Any]) -> list[dict[str, str]]:
    suggested = profile["routing"]["suggested_profile"]
    return [
        {
            "id": "plan_with_profile",
            "title": "Use this profile as advisory routing context",
            "command": "interp-lab plan-scale --from-env --model-params <size> --tokens <tokens> --d-model <width>",
        },
        {
            "id": "choose_route",
            "title": f"Review route options before launching; suggested starting route is {suggested}",
            "command": "interp-lab plan-scale --profile <profile> --model-params <size> --tokens <tokens> --d-model <width>",
        },
        {
            "id": "save_profile",
            "title": "Save this profile when planning from another machine or an agent workflow",
            "command": "interp-lab profile-env --out reports/env-profile.json --json",
        },
    ]


def _capabilities(profile: dict[str, Any]) -> dict[str, Any]:
    accelerators = profile.get("accelerators", [])
    gpu_devices = [item for item in accelerators if item.get("kind") in {"cuda", "nvidia-smi"}]
    total_gpu_memory = sum(int(item.get("total_memory_bytes") or 0) for item in gpu_devices)
    max_gpu_memory = max([int(item.get("total_memory_bytes") or 0) for item in gpu_devices] or [0])
    env_flags = profile.get("environment_flags", {})
    has_mps = any(item.get("kind") == "mps" for item in accelerators)
    has_cluster_environment = any(
        _flag_present(env_flags, name)
        for name in ["SLURM_JOB_ID", "SLURM_CPUS_ON_NODE", "SLURM_GPUS", "WORLD_SIZE", "PBS_JOBID", "LSB_JOBID"]
    )
    return {
        "gpu_count": len(gpu_devices),
        "total_gpu_memory_bytes": total_gpu_memory,
        "total_gpu_memory_human": _format_bytes(total_gpu_memory) if total_gpu_memory else None,
        "max_gpu_memory_bytes": max_gpu_memory,
        "max_gpu_memory_human": _format_bytes(max_gpu_memory) if max_gpu_memory else None,
        "has_cuda": bool(gpu_devices),
        "has_mps": has_mps,
        "has_local_accelerator": bool(gpu_devices or has_mps),
        "has_cluster_environment": has_cluster_environment,
    }


def _optional_modules() -> dict[str, dict[str, Any]]:
    modules: dict[str, dict[str, Any]] = {}
    for module_name, purpose in OPTIONAL_MODULES.items():
        package_name = PACKAGE_NAMES.get(module_name, module_name)
        found = importlib.util.find_spec(module_name) is not None
        version = None
        if found:
            try:
                version = importlib.metadata.version(package_name)
            except importlib.metadata.PackageNotFoundError:
                version = "available"
        modules[module_name] = {
            "available": found,
            "package": package_name,
            "version": version,
            "purpose": purpose,
        }
    return modules


def _accelerators(*, probe_accelerators: bool) -> list[dict[str, Any]]:
    if not probe_accelerators:
        return []
    devices = _torch_accelerators()
    if not any(item.get("kind") == "cuda" for item in devices):
        devices.extend(_nvidia_smi_accelerators())
    return devices


def _torch_accelerators() -> list[dict[str, Any]]:
    if importlib.util.find_spec("torch") is None:
        return []
    try:
        import torch  # type: ignore[import-not-found]
    except Exception:
        return []
    devices: list[dict[str, Any]] = []
    try:
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                free_memory = None
                if hasattr(torch.cuda, "mem_get_info"):
                    try:
                        free_memory, _ = torch.cuda.mem_get_info(index)
                    except Exception:
                        free_memory = None
                devices.append(
                    {
                        "kind": "cuda",
                        "index": index,
                        "name": getattr(props, "name", f"cuda:{index}"),
                        "total_memory_bytes": int(getattr(props, "total_memory", 0) or 0),
                        "total_memory_human": _format_bytes(int(getattr(props, "total_memory", 0) or 0)),
                        "available_memory_bytes": int(free_memory) if free_memory is not None else None,
                        "available_memory_human": _format_bytes(int(free_memory)) if free_memory is not None else None,
                        "source": "torch",
                    }
                )
    except Exception:
        pass
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            devices.append(
                {
                    "kind": "mps",
                    "index": 0,
                    "name": "Apple Metal Performance Shaders",
                    "total_memory_bytes": None,
                    "total_memory_human": None,
                    "available_memory_bytes": None,
                    "available_memory_human": None,
                    "source": "torch",
                }
            )
    except Exception:
        pass
    return devices


def _nvidia_smi_accelerators() -> list[dict[str, Any]]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return []
    try:
        result = subprocess.run(
            [
                executable,
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return []
    if result.returncode != 0:
        return []
    devices = []
    for index, line in enumerate(result.stdout.splitlines()):
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        total = _megabytes_to_bytes(parts[1])
        free = _megabytes_to_bytes(parts[2])
        devices.append(
            {
                "kind": "nvidia-smi",
                "index": index,
                "name": parts[0],
                "total_memory_bytes": total,
                "total_memory_human": _format_bytes(total),
                "available_memory_bytes": free,
                "available_memory_human": _format_bytes(free),
                "source": "nvidia-smi",
            }
        )
    return devices


def _memory_info() -> dict[str, Any]:
    total, available = _platform_memory_bytes()
    return {
        "total_bytes": total,
        "available_bytes": available,
        "total_human": _format_bytes(total) if total is not None else None,
        "available_human": _format_bytes(available) if available is not None else None,
    }


def _platform_memory_bytes() -> tuple[int | None, int | None]:
    system = platform.system()
    if system == "Windows":
        return _windows_memory_bytes()
    if system == "Linux":
        proc = _linux_proc_memory_bytes()
        if proc != (None, None):
            return proc
    total = _sysconf_total_memory()
    return total, None


def _windows_memory_bytes() -> tuple[int | None, int | None]:
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    try:
        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys), int(status.ullAvailPhys)
    except Exception:
        return None, None
    return None, None


def _linux_proc_memory_bytes() -> tuple[int | None, int | None]:
    path = Path("/proc/meminfo")
    if not path.exists():
        return None, None
    values: dict[str, int] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if ":" not in line:
                continue
            key, rest = line.split(":", 1)
            amount = rest.strip().split()[0]
            values[key] = int(amount) * 1024
    except Exception:
        return None, None
    return values.get("MemTotal"), values.get("MemAvailable")


def _sysconf_total_memory() -> int | None:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return None
    try:
        return int(pages) * int(page_size)
    except Exception:
        return None


def _environment_flags(env: Mapping[str, str]) -> dict[str, dict[str, Any]]:
    flags: dict[str, dict[str, Any]] = {}
    for name in ENV_FLAGS:
        value = env.get(name)
        entry: dict[str, Any] = {"present": bool(value)}
        if name == "CUDA_VISIBLE_DEVICES" and value:
            entry["visible_device_count"] = _visible_cuda_device_count(value)
        if name == "INTERP_LAB_PROFILE" and value in PROFILE_ORDER:
            entry["profile"] = value
        flags[name] = entry
    return flags


def _explicit_env_profile(env_flags: dict[str, dict[str, Any]]) -> str | None:
    value = env_flags.get("INTERP_LAB_PROFILE", {}).get("profile")
    return str(value) if value in PROFILE_ORDER else None


def _flag_present(env_flags: dict[str, dict[str, Any]], name: str) -> bool:
    return bool(env_flags.get(name, {}).get("present"))


def _module_available(modules: dict[str, dict[str, Any]], name: str) -> bool:
    return bool(modules.get(name, {}).get("available"))


def _gpu_route_reason(gpu_count: int, max_gpu_memory: int, has_mps: bool) -> str:
    if gpu_count:
        return f"{gpu_count} CUDA device(s) detected; largest VRAM is {_format_bytes(max_gpu_memory)}."
    if has_mps:
        return "Apple MPS acceleration is available."
    return "Choose this route for another machine with a local accelerator."


def _cluster_route_reason(has_cluster_env: bool, gpu_count: int, total_gpu_memory: int) -> str:
    if has_cluster_env:
        return "Cluster scheduler or distributed environment variables were detected."
    if gpu_count >= 4 or total_gpu_memory >= 160 * 1024**3:
        return f"{gpu_count} GPU device(s) with {_format_bytes(total_gpu_memory)} total VRAM were detected."
    return "Choose this route for a scheduler, multi-GPU node, or distributed training environment."


def _remote_route_reason(remote_ready: bool, remote_possible: bool) -> str:
    if remote_ready:
        return "Remote API credentials were detected."
    if remote_possible:
        return "Remote adapter packages are installed; add credentials when you want to use this route."
    return "Choose this route for hosted model execution or an internal activation service."


def _accelerator_summary(accelerators: list[dict[str, Any]]) -> str:
    first = accelerators[0]
    name = first.get("name") or first.get("kind")
    if len(accelerators) == 1:
        memory = first.get("total_memory_human")
        return f"{name}" + (f" with {memory} VRAM" if memory else "")
    total = sum(int(item.get("total_memory_bytes") or 0) for item in accelerators)
    memory = f" with {_format_bytes(total)} total VRAM" if total else ""
    return f"{len(accelerators)} accelerators{memory}"


def _existing_probe_path(path: Path) -> Path:
    candidate = path.expanduser()
    if candidate.exists():
        return candidate.resolve()
    for parent in [candidate, *candidate.parents]:
        if parent.exists():
            return parent.resolve()
    return Path.cwd()


def _visible_cuda_device_count(value: str) -> int | None:
    text = value.strip()
    if not text or text.lower() in {"none", "no", "-1"}:
        return 0
    return len([item for item in text.split(",") if item.strip()])


def _megabytes_to_bytes(value: str) -> int:
    try:
        return int(float(value.strip())) * 1024**2
    except ValueError:
        return 0


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    amount = float(value)
    for unit in BYTE_SUFFIXES:
        if amount < 1024.0 or unit == BYTE_SUFFIXES[-1]:
            return f"{amount:.2f} {unit}"
        amount /= 1024.0
    return f"{amount:.2f} EB"
