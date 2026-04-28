"""Thin wrapper around wandb so callers can opt out cleanly.

Reads WANDB_API_KEY from .env (or environment). If the key is absent, all
helpers become no-ops — training keeps running with no metrics logged.
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Optional

from dotenv import load_dotenv

_LOADED_DOTENV = False


def _ensure_dotenv_loaded() -> None:
    global _LOADED_DOTENV
    if not _LOADED_DOTENV:
        load_dotenv()
        _LOADED_DOTENV = True


def init_wandb_run(
    *,
    project: Optional[str] = "cot_interp",
    run_name: Optional[str] = None,
    config: Optional[Mapping[str, Any]] = None,
    mode: Optional[str] = None,
):
    """Start a wandb run and return it, or None if wandb is disabled / unavailable.

    Disabled when:
      - project is None / empty
      - WANDB_API_KEY is not set (and `mode` is not 'offline'/'disabled')
      - the wandb import fails
    """
    if not project:
        return None
    _ensure_dotenv_loaded()
    api_key = os.getenv("WANDB_API_KEY")
    if not api_key and mode not in ("offline", "disabled"):
        print("[wandb] WANDB_API_KEY not set; skipping wandb logging.")
        return None
    try:
        import wandb
    except ImportError:
        print("[wandb] wandb not installed; skipping wandb logging.")
        return None

    if api_key:
        try:
            wandb.login(key=api_key, verify=False)
        except Exception as exc:
            print(f"[wandb] login failed ({exc}); skipping wandb logging.")
            return None

    run = wandb.init(
        project=project,
        name=run_name,
        config=dict(config) if config else None,
        mode=mode,
        reinit=True,
    )
    print(f"[wandb] run: {run.name} ({run.url})")
    return run


def log_step(run, *, step: int, metrics: Mapping[str, Any]) -> None:
    """Log a step's metrics. No-op when run is None."""
    if run is None:
        return
    run.log(dict(metrics), step=step)


def finish_wandb_run(run) -> None:
    if run is None:
        return
    run.finish()
