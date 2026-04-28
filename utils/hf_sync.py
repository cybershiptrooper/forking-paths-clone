"""Sync results folder to/from a HuggingFace Hub dataset repo.

Push is additive (no remote deletion). Pull is content-aware: it compares
remote and local hashes, downloads missing files, and on hash mismatch
asks the user (or a programmatic callback) what to do.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from huggingface_hub import HfApi, hf_hub_download


CONFLICT_OVERWRITE = "o"
CONFLICT_SKIP = "s"
CONFLICT_ALL_OVERWRITE = "a"
CONFLICT_ALL_SKIP = "A"

ConflictPrompt = Callable[[str], str]

DEFAULT_REPO_ID = "cybershiptrooper/cot_interp"
DEFAULT_LOCAL_ROOT = "results"  # path under which to mirror layout in HF
NO_PUSH_ENV_VAR = "IS_EXPTS_NO_HF_PUSH"  # set to non-empty to disable auto-push


@dataclass
class SyncStats:
    downloaded: int = 0
    skipped_match: int = 0
    skipped_conflict: int = 0
    overwritten: int = 0


def _git_blob_sha1(path: Path) -> str:
    """Git's blob sha1 — matches HF's `blob_id` for non-LFS files."""
    size = path.stat().st_size
    h = hashlib.sha1()
    h.update(f"blob {size}\0".encode())
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _local_matches_remote(local_path: Path, info) -> bool:
    """Hash local file using whichever scheme HF used for the remote file.

    HF stores LFS files with `info.lfs.sha256` (true sha256 of content) and
    non-LFS with `info.blob_id` (git's hash-object sha1).
    """
    if not local_path.exists() or not local_path.is_file():
        return False
    if info.lfs is not None:
        return _file_sha256(local_path) == info.lfs.sha256
    return _git_blob_sha1(local_path) == info.blob_id


def _default_prompt(rel_path: str) -> str:
    valid = {CONFLICT_OVERWRITE, CONFLICT_SKIP, CONFLICT_ALL_OVERWRITE, CONFLICT_ALL_SKIP}
    while True:
        ans = input(
            f"Conflict at {rel_path}: local differs from remote. "
            f"[o]verwrite / [s]kip / [a]ll-overwrite / [A]ll-skip: "
        ).strip()
        if ans in valid:
            return ans
        print(f"  invalid response {ans!r}; please pick one of {sorted(valid)}")


# Patterns excluded from every push. Mirrors the .gitignore at the root of
# the HF repo — `upload_folder` does not honour repo-side .gitignore by
# default, so we enforce the same patterns here.
DEFAULT_IGNORE_PATTERNS: tuple[str, ...] = ("*test*",)


def push(
    local_dir: str | Path,
    repo_id: str,
    *,
    remote_prefix: str = "",
    repo_type: str = "dataset",
    private: bool = True,
    commit_message: str = "Sync from local",
    ignore_patterns: Optional[list[str]] = None,
    api: Optional[HfApi] = None,
) -> None:
    """Upload `local_dir` to `<repo_id>:<remote_prefix>/`. Additive — no remote deletes."""
    api = api or HfApi()
    local_dir = Path(local_dir)
    if not local_dir.is_dir():
        raise FileNotFoundError(f"local_dir not found: {local_dir}")
    api.create_repo(
        repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True,
    )
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type=repo_type,
        path_in_repo=remote_prefix or None,
        commit_message=commit_message,
        ignore_patterns=list(
            ignore_patterns if ignore_patterns is not None else DEFAULT_IGNORE_PATTERNS
        ),
    )


def push_file(
    local_path: str | Path,
    repo_id: str,
    *,
    path_in_repo: str,
    repo_type: str = "dataset",
    private: bool = True,
    commit_message: Optional[str] = None,
    api: Optional[HfApi] = None,
) -> None:
    """Upload a single file. Used as an end-of-job hook from training/eval scripts.

    Skips silently and prints a warning if HF auth is missing — this hook
    must never break a training run.
    """
    try:
        api = api or HfApi()
        local_path = Path(local_path)
        if not local_path.is_file():
            raise FileNotFoundError(f"local_path not a file: {local_path}")
        api.create_repo(
            repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True,
        )
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=commit_message or f"Upload {path_in_repo}",
        )
        print(f"[hf] pushed {local_path} -> {repo_id}:{path_in_repo}")
    except Exception as exc:
        print(f"[hf] push_file skipped ({exc.__class__.__name__}): {exc}")


def push_auto(
    local_path: str | Path,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    local_root: str = DEFAULT_LOCAL_ROOT,
    repo_type: str = "dataset",
) -> None:
    """End-of-run hook: push a single file to HF, mirroring its path under `local_root/`.

    Example: `results/circuit_discovery/foo.json` is uploaded to
    `<repo_id>:circuit_discovery/foo.json`.

    Skipped silently when:
      - env var IS_EXPTS_NO_HF_PUSH is set
      - HF auth is missing (push_file handles this internally)
    """
    if os.getenv(NO_PUSH_ENV_VAR):
        print(f"[hf] {NO_PUSH_ENV_VAR} set; skipping auto-push of {local_path}")
        return
    p = Path(local_path).resolve()
    parts = p.parts
    if local_root in parts:
        idx = len(parts) - 1 - list(reversed(parts)).index(local_root)
        path_in_repo = "/".join(parts[idx + 1:])
    else:
        path_in_repo = p.name
    # Honour the repo .gitignore patterns even on single-file pushes.
    if any(_glob_matches(path_in_repo, pat) for pat in DEFAULT_IGNORE_PATTERNS):
        print(f"[hf] auto-push skipped (matches ignore pattern): {path_in_repo}")
        return
    push_file(
        local_path=p,
        repo_id=repo_id,
        path_in_repo=path_in_repo,
        repo_type=repo_type,
    )


def _glob_matches(path: str, pattern: str) -> bool:
    """Match a path against a glob pattern, considering each path component."""
    import fnmatch
    if fnmatch.fnmatch(path, pattern):
        return True
    return any(fnmatch.fnmatch(part, pattern) for part in path.split("/"))


def pull(
    local_dir: str | Path,
    repo_id: str,
    *,
    remote_prefix: str = "",
    repo_type: str = "dataset",
    on_conflict: Optional[ConflictPrompt] = None,
    default_on_conflict: str = CONFLICT_SKIP,
    api: Optional[HfApi] = None,
) -> SyncStats:
    """Download files in `<repo_id>:<remote_prefix>/` to `local_dir`.

    Conflicts (local exists with different content) trigger `on_conflict(rel_path)`,
    which must return one of CONFLICT_OVERWRITE / SKIP / ALL_OVERWRITE / ALL_SKIP.
    If `on_conflict` is None and no TTY is attached, falls back to `default_on_conflict`.
    """
    api = api or HfApi()
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    # Resolve which prompt to use up-front so behaviour is predictable.
    if on_conflict is None:
        if os.isatty(0):
            on_conflict = _default_prompt
        else:
            on_conflict = lambda _p: default_on_conflict  # noqa: E731

    all_files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
    prefix = remote_prefix.rstrip("/")
    remote_paths = [
        p for p in all_files
        if (not prefix) or p == prefix or p.startswith(prefix + "/")
    ]
    if not remote_paths:
        print(f"No files found at {repo_id}:{remote_prefix or '/'}")
        return SyncStats()

    # Pull file metadata in bulk so we can compare hashes without downloading.
    infos = api.get_paths_info(
        repo_id=repo_id, paths=remote_paths, repo_type=repo_type, expand=True,
    )
    info_by_path = {i.path: i for i in infos}

    stats = SyncStats()
    sticky: Optional[str] = None  # set when user picks ALL_OVERWRITE / ALL_SKIP

    for remote_path in sorted(info_by_path.keys()):
        info = info_by_path[remote_path]
        # Strip the remote prefix to get the path under local_dir.
        if prefix and remote_path.startswith(prefix + "/"):
            rel = remote_path[len(prefix) + 1:]
        elif remote_path == prefix:
            rel = Path(remote_path).name
        else:
            rel = remote_path
        local_path = local_dir / rel

        if local_path.exists():
            if _local_matches_remote(local_path, info):
                stats.skipped_match += 1
                continue
            # Hash mismatch.
            if sticky == CONFLICT_ALL_OVERWRITE:
                action = CONFLICT_OVERWRITE
            elif sticky == CONFLICT_ALL_SKIP:
                action = CONFLICT_SKIP
            else:
                action = on_conflict(rel)
                if action == CONFLICT_ALL_OVERWRITE:
                    sticky = CONFLICT_ALL_OVERWRITE
                    action = CONFLICT_OVERWRITE
                elif action == CONFLICT_ALL_SKIP:
                    sticky = CONFLICT_ALL_SKIP
                    action = CONFLICT_SKIP
            if action == CONFLICT_SKIP:
                stats.skipped_conflict += 1
                print(f"  skip (conflict): {rel}")
                continue
            stats.overwritten += 1
            print(f"  overwrite: {rel}")
        else:
            stats.downloaded += 1
            print(f"  download: {rel}")

        # Download via HF's cache, then atomically move to local_path.
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            cached = hf_hub_download(
                repo_id=repo_id, filename=remote_path, repo_type=repo_type,
                local_dir=tmp,
            )
            shutil.move(cached, local_path)

    return stats
