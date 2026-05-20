from __future__ import annotations

import argparse
import importlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PublishResult:
    repo_id: str
    repo_type: str
    uploaded: list[str]
    dry_run: bool = False


def publish_hf_artifact(
    *,
    repo_id: str,
    paths: list[str | Path],
    repo_type: str = "dataset",
    private: bool = False,
    path_in_repo: str | None = None,
    revision: str | None = None,
    commit_message: str = "Upload interp-lab artifact",
    card_title: str | None = None,
    tags: list[str] | None = None,
    dry_run: bool = False,
) -> PublishResult:
    source_paths = [Path(path) for path in paths]
    if not source_paths:
        raise ValueError("At least one --path is required")
    for path in source_paths:
        if not path.exists():
            raise ValueError(f"{path}: path does not exist")
    if path_in_repo and len(source_paths) != 1:
        raise ValueError("--path-in-repo can only be used with one --path")
    uploaded = [
        _path_in_repo_for(path, path_in_repo=path_in_repo, multiple=len(source_paths) > 1)
        for path in source_paths
    ]
    if dry_run:
        return PublishResult(repo_id=repo_id, repo_type=repo_type, uploaded=uploaded, dry_run=True)

    huggingface_hub = _optional_import(
        "huggingface_hub",
        "Install `interp-lab[publish]` to publish artifacts to Hugging Face.",
    )
    api = huggingface_hub.HfApi()
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)
    for path, remote_path in zip(source_paths, uploaded):
        remote_arg = None if remote_path == "." else remote_path
        if path.is_dir():
            kwargs = {
                "repo_id": repo_id,
                "repo_type": repo_type,
                "folder_path": str(path),
                "revision": revision,
                "commit_message": commit_message,
            }
            if remote_arg is not None:
                kwargs["path_in_repo"] = remote_arg
            api.upload_folder(**kwargs)
        else:
            api.upload_file(
                repo_id=repo_id,
                repo_type=repo_type,
                path_or_fileobj=str(path),
                path_in_repo=remote_arg or path.name,
                revision=revision,
                commit_message=commit_message,
            )
    card = render_hf_card(repo_id=repo_id, repo_type=repo_type, title=card_title, tags=tags or [])
    api.upload_file(
        repo_id=repo_id,
        repo_type=repo_type,
        path_or_fileobj=io.BytesIO(card.encode("utf-8")),
        path_in_repo="README.md",
        revision=revision,
        commit_message="Update interp-lab artifact card",
    )
    return PublishResult(repo_id=repo_id, repo_type=repo_type, uploaded=uploaded, dry_run=False)


def render_hf_card(*, repo_id: str, repo_type: str, title: str | None, tags: list[str]) -> str:
    tag_lines = ["mechanistic-interpretability", "sparse-autoencoders", "interp-lab", *tags]
    yaml_tags = "\n".join(f"- {tag}" for tag in dict.fromkeys(tag_lines))
    heading = title or repo_id
    return (
        "---\n"
        f"tags:\n{yaml_tags}\n"
        "---\n\n"
        f"# {heading}\n\n"
        "This repository contains artifacts produced by interp-lab: reports, activation records, "
        "intervention records, trained SAE metadata, or related interpretability outputs.\n\n"
        f"Repository type: `{repo_type}`.\n"
    )


def build_hf_publish_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Publish interp-lab artifacts to Hugging Face Hub.")
    parser.add_argument("--repo-id", required=True, help="Hugging Face repo id, e.g. user/interp-lab-demo.")
    parser.add_argument(
        "--path",
        action="append",
        required=True,
        help="File or directory to upload. Repeat for multiple artifacts.",
    )
    parser.add_argument("--repo-type", choices=["model", "dataset", "space"], default="dataset")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--path-in-repo", help="Destination path when uploading one file or folder.")
    parser.add_argument("--revision", help="Optional branch or revision.")
    parser.add_argument("--commit-message", default="Upload interp-lab artifact")
    parser.add_argument("--card-title", help="README title for the HF repo.")
    parser.add_argument("--tag", action="append", default=[], help="Additional Hugging Face tag.")
    parser.add_argument("--dry-run", action="store_true", help="Validate paths and print planned uploads.")
    return parser


def run_hf_publish_from_args(args: argparse.Namespace) -> PublishResult:
    return publish_hf_artifact(
        repo_id=args.repo_id,
        paths=args.path,
        repo_type=args.repo_type,
        private=args.private,
        path_in_repo=args.path_in_repo,
        revision=args.revision,
        commit_message=args.commit_message,
        card_title=args.card_title,
        tags=args.tag,
        dry_run=args.dry_run,
    )


def _optional_import(name: str, message: str):
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise RuntimeError(message) from exc


def _path_in_repo_for(path: Path, *, path_in_repo: str | None, multiple: bool) -> str:
    if path_in_repo:
        return path_in_repo.strip("/")
    if multiple:
        return path.name
    return path.name if path.is_file() else "."
