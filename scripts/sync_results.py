"""CLI for syncing results/circuit_discovery/ to/from HuggingFace Hub.

Examples:
    # First-time push from a VM:
    uv run python -m scripts.sync_results push

    # Pull on a fresh VM (prompts on conflicts):
    uv run python -m scripts.sync_results pull

    # Non-interactive pull (default: skip on conflict):
    uv run python -m scripts.sync_results pull --on-conflict skip
"""

import argparse
import os
import sys

from dotenv import load_dotenv
from huggingface_hub import HfApi

from utils.hf_sync import (
    CONFLICT_ALL_OVERWRITE,
    CONFLICT_OVERWRITE,
    CONFLICT_SKIP,
    pull,
    push,
)


DEFAULT_LOCAL_DIR = "results/circuit_discovery"
DEFAULT_REPO = "cybershiptrooper/cot_interp"
DEFAULT_REMOTE_PREFIX = "circuit_discovery"


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("action", choices=["push", "pull"])
    p.add_argument("--local-dir", default=DEFAULT_LOCAL_DIR)
    p.add_argument("--repo-id", default=DEFAULT_REPO)
    p.add_argument(
        "--remote-prefix", default=DEFAULT_REMOTE_PREFIX,
        help="Prefix inside the HF repo. Default: 'circuit_discovery'.",
    )
    p.add_argument(
        "--repo-type", default="dataset", choices=["dataset", "model", "space"],
    )
    p.add_argument(
        "--on-conflict", choices=["prompt", "skip", "overwrite"], default="prompt",
        help="What to do when a local file differs from remote (pull only).",
    )
    p.add_argument(
        "--commit-message", default="Sync from local",
        help="Commit message for push.",
    )
    return p


def main() -> int:
    load_dotenv()
    args = _make_parser().parse_args()

    # HF token: prefer the cache login, fall back to env.
    api = HfApi(token=os.getenv("HF_TOKEN") or None)

    if args.action == "push":
        push(
            local_dir=args.local_dir,
            repo_id=args.repo_id,
            remote_prefix=args.remote_prefix,
            repo_type=args.repo_type,
            commit_message=args.commit_message,
            api=api,
        )
        print(f"Pushed {args.local_dir} -> {args.repo_id}:{args.remote_prefix}/")
        return 0

    # pull
    if args.on_conflict == "prompt":
        on_conflict = None  # let pull() pick TTY prompt or default
        default = CONFLICT_SKIP
    elif args.on_conflict == "skip":
        on_conflict = lambda _p: CONFLICT_SKIP  # noqa: E731
        default = CONFLICT_SKIP
    else:  # overwrite
        on_conflict = lambda _p: CONFLICT_ALL_OVERWRITE  # noqa: E731
        default = CONFLICT_OVERWRITE

    stats = pull(
        local_dir=args.local_dir,
        repo_id=args.repo_id,
        remote_prefix=args.remote_prefix,
        repo_type=args.repo_type,
        on_conflict=on_conflict,
        default_on_conflict=default,
        api=api,
    )
    print(
        f"\nDone. downloaded={stats.downloaded} "
        f"matched={stats.skipped_match} "
        f"overwritten={stats.overwritten} "
        f"skipped(conflict)={stats.skipped_conflict}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
