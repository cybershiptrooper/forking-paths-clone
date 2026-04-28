"""End-to-end integration test for utils.hf_sync against a separate test repo.

Critically, this script never touches the production HF repo
(``cybershiptrooper/cot_interp``). All push/pull traffic goes to a
dedicated test repo (default ``cybershiptrooper/cot_interp_test``) which
is wiped clean on completion.

Steps:
  1. Copy ``results/circuit_discovery/`` to a temp directory.
  2. Push the copy to the test repo.
  3. Delete a subdir locally, pull, assert the subdir is restored byte-for-byte
     and that all other files were skipped (hash-match) without touching local.
  4. Modify a file locally, pull with ``on_conflict=skip``, assert the local
     edit is preserved.
  5. Pull with ``on_conflict=overwrite``, assert the local file is restored
     to remote bytes.
  6. Cleanup: delete the local copy and wipe the test repo (or leave it
     in place with --keep for inspection).

Usage:
  uv run python -m scripts.test_sync_e2e                  # default: full round-trip
  uv run python -m scripts.test_sync_e2e --keep           # don't wipe the test repo
  uv run python -m scripts.test_sync_e2e --src DIR        # use a different source dir
  uv run python -m scripts.test_sync_e2e --test-repo X/Y  # use a different test repo
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

from utils.hf_sync import (
    CONFLICT_OVERWRITE,
    CONFLICT_SKIP,
    DEFAULT_REPO_ID,
    pull,
    push,
)


DEFAULT_TEST_REPO = "cybershiptrooper/cot_interp_test"
DEFAULT_SRC = "results/circuit_discovery"


def _file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_tree(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): _file_sha256(p)
        for p in root.rglob("*")
        if p.is_file()
    }


def _pick_subdir_to_delete(local_dir: Path) -> Path:
    """Pick the smallest top-level subdir for the round-trip restore test."""
    subdirs = [p for p in local_dir.iterdir() if p.is_dir()]
    if not subdirs:
        raise RuntimeError(f"No subdirs found in {local_dir} to use for the test.")
    subdirs.sort(key=lambda p: sum(f.stat().st_size for f in p.rglob("*") if f.is_file()))
    return subdirs[0]


def _pick_file_for_conflict(local_dir: Path) -> Path:
    """Pick a small JSON file for the conflict test."""
    candidates = sorted(
        (p for p in local_dir.rglob("*.json") if p.is_file()),
        key=lambda p: p.stat().st_size,
    )
    if not candidates:
        raise RuntimeError(f"No JSON files found in {local_dir}.")
    return candidates[0]


def _wipe_test_repo(api: HfApi, repo_id: str) -> None:
    """Remove every non-meta file from the test repo, leaving the empty repo."""
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    targets = [p for p in files if not p.startswith(".git")]
    if not targets:
        print(f"[cleanup] {repo_id}: nothing to delete.")
        return
    print(f"[cleanup] deleting {len(targets)} files from {repo_id} ...")
    api.delete_files(
        delete_patterns=targets,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="e2e test cleanup",
    )


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        print(f"FAIL: {msg}", file=sys.stderr)
        sys.exit(1)
    print(f"  ok: {msg}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", default=DEFAULT_SRC, help="Source dir to copy.")
    ap.add_argument(
        "--test-repo", default=DEFAULT_TEST_REPO,
        help="HF dataset repo for testing (must NOT be the production repo).",
    )
    ap.add_argument(
        "--remote-prefix", default="circuit_discovery",
        help="Prefix inside the test repo.",
    )
    ap.add_argument(
        "--keep", action="store_true",
        help="Don't wipe the test repo or the local temp copy at the end.",
    )
    args = ap.parse_args()

    if args.test_repo == DEFAULT_REPO_ID:
        print(
            f"refusing to run e2e test against production repo {DEFAULT_REPO_ID}.",
            file=sys.stderr,
        )
        return 2

    src = Path(args.src)
    if not src.is_dir():
        print(f"src not found: {src}", file=sys.stderr)
        return 1

    load_dotenv()
    api = HfApi(token=os.getenv("HF_TOKEN") or None)

    # 1. Copy source to a temp dir.
    tmp_root = Path(tempfile.mkdtemp(prefix="cot_interp_e2e_"))
    local_dir = tmp_root / "circuit_discovery"
    print(f"[1] copying {src} -> {local_dir} ...")
    shutil.copytree(src, local_dir)
    initial_hashes = _hash_tree(local_dir)
    print(f"    {len(initial_hashes)} files copied")

    try:
        # 2. Push the copy to the test repo. Pass empty ignore_patterns so test
        #    fixtures named *test*.json (if any) round-trip too.
        print(f"\n[2] pushing -> {args.test_repo}:{args.remote_prefix}/ ...")
        push(
            local_dir=local_dir,
            repo_id=args.test_repo,
            remote_prefix=args.remote_prefix,
            commit_message="e2e test: initial push",
            ignore_patterns=[],
            api=api,
        )

        # 3. Delete a subdir locally, pull, verify restore + nothing else changes.
        victim = _pick_subdir_to_delete(local_dir)
        rel = victim.relative_to(local_dir)
        before_hashes = {k: v for k, v in initial_hashes.items() if not k.startswith(str(rel) + os.sep) and k != str(rel)}
        deleted_keys = sorted(set(initial_hashes) - set(before_hashes))
        print(f"\n[3] deleting subdir {rel} ({len(deleted_keys)} files), then pulling ...")
        shutil.rmtree(victim)
        stats = pull(
            local_dir=local_dir,
            repo_id=args.test_repo,
            remote_prefix=args.remote_prefix,
            on_conflict=lambda _p: CONFLICT_SKIP,
            api=api,
        )
        _assert(stats.downloaded == len(deleted_keys), f"downloaded == {len(deleted_keys)} (got {stats.downloaded})")
        _assert(stats.skipped_conflict == 0, "no conflicts during restore pull")
        _assert(stats.overwritten == 0, "no overwrites during restore pull")
        after_hashes = _hash_tree(local_dir)
        _assert(after_hashes == initial_hashes, "byte-identical restore (all hashes match initial)")

        # 4. Conflict + skip: local edit is preserved.
        conflict_file = _pick_file_for_conflict(local_dir)
        rel_cf = conflict_file.relative_to(local_dir)
        original_bytes = conflict_file.read_bytes()
        edited = b'{"e2e_test_local_edit": true}\n'
        conflict_file.write_bytes(edited)
        print(f"\n[4] conflict test (skip) on {rel_cf} ...")
        stats = pull(
            local_dir=local_dir,
            repo_id=args.test_repo,
            remote_prefix=args.remote_prefix,
            on_conflict=lambda _p: CONFLICT_SKIP,
            api=api,
        )
        _assert(stats.skipped_conflict == 1, "exactly 1 skipped conflict")
        _assert(conflict_file.read_bytes() == edited, "local edit preserved (skip)")

        # 5. Conflict + overwrite: local restored to remote bytes.
        print(f"\n[5] conflict test (overwrite) on {rel_cf} ...")
        stats = pull(
            local_dir=local_dir,
            repo_id=args.test_repo,
            remote_prefix=args.remote_prefix,
            on_conflict=lambda _p: CONFLICT_OVERWRITE,
            api=api,
        )
        _assert(stats.overwritten == 1, "exactly 1 overwrite")
        _assert(conflict_file.read_bytes() == original_bytes, "local restored to remote (overwrite)")

        print("\nAll e2e checks passed.")
        return 0

    finally:
        if args.keep:
            print(f"\n[--keep] left local copy at {local_dir}; left test repo populated at {args.test_repo}")
        else:
            print(f"\n[cleanup] removing local copy {tmp_root} ...")
            shutil.rmtree(tmp_root, ignore_errors=True)
            try:
                _wipe_test_repo(api, args.test_repo)
            except Exception as exc:
                print(f"[cleanup] test repo wipe failed: {exc}")


if __name__ == "__main__":
    sys.exit(main())
