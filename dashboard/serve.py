#!/usr/bin/env python3
"""Serve the circuit tracer dashboard with optional NodeMask JSON API."""

from __future__ import annotations

import argparse
import json
import os
import sys
import re
from collections import defaultdict
from glob import glob
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_DIR = Path(__file__).resolve().parent
RESAMPLE_DIR = ROOT / "results/circuit_discovery/tempered_snis"
RESAMPLE_BOOTSTRAP_B = 10_000
RESAMPLE_BOOTSTRAP_SEED = 0

# ── Mode-dependent state (set by main() before server starts) ────────────
ACTIVE_INDEX_FILE: str = "index.html"
ACTIVE_MASK_GLOBS: list[str] = ["results/circuit_discovery/**/*.json"]
ACTIVE_MASK_EXCLUDES: list[str] = []  # substrings to exclude from mask list
ACTIVE_MODE_LABEL: str = "global"


class DashboardHandler(SimpleHTTPRequestHandler):
    """Serves static files from dashboard/ and provides a mask API."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DASHBOARD_DIR), **kwargs)

    def translate_path(self, path: str) -> str:
        """Rewrite '/' to the active index file."""
        if path == "/" or path == "":
            return str(DASHBOARD_DIR / ACTIVE_INDEX_FILE)
        return super().translate_path(path)

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/masks":
            self._serve_mask_list()
        elif parsed.path == "/api/mask_groups":
            self._serve_mask_groups()
        elif parsed.path == "/api/mask":
            qs = parse_qs(parsed.query)
            path = qs.get("path", [None])[0]
            if path:
                self._serve_mask(path)
            else:
                self._json_response({"error": "missing ?path="}, 400)
        elif parsed.path == "/api/resample":
            qs = parse_qs(parsed.query)
            path = qs.get("path", [None])[0]
            if path:
                self._serve_resample(path)
            else:
                self._json_response({"error": "missing ?path="}, 400)
        elif parsed.path == "/api/eval_sidecar":
            qs = parse_qs(parsed.query)
            path = qs.get("path", [None])[0]
            if path:
                self._serve_eval_sidecar(path)
            else:
                self._json_response({"error": "missing ?path="}, 400)
        elif parsed.path == "/api/group_evals":
            qs = parse_qs(parsed.query)
            paths = qs.get("paths", [])
            if paths:
                self._serve_group_evals(paths)
            else:
                self._json_response({"error": "missing ?paths="}, 400)
        elif parsed.path == "/api/resample_sweep":
            self._serve_resample_sweep()
        elif parsed.path == "/api/resample_csv":
            self._serve_resample_csv()
        else:
            super().do_GET()

    def _serve_mask_list(self):
        paths = set()
        for pattern in ACTIVE_MASK_GLOBS:
            paths.update(glob(str(ROOT / pattern), recursive=True))
        # Filter out eval sidecar files and mode-specific excludes
        paths = [
            p for p in sorted(paths)
            if not p.endswith(".eval.json")
            and not any(exc in p for exc in ACTIVE_MASK_EXCLUDES)
        ]
        # Return relative paths from project root
        rel = [str(Path(p).relative_to(ROOT)) for p in paths]
        self._json_response(rel)

    def _serve_mask_groups(self):
        """Group SNP sweep masks by prompt/experiment, collecting sparsity variants.

        Returns a JSON object:
        {
          "groups": [
            {
              "label": "pair_kl_g01_k05_p08",
              "sparsities": [
                {"tsp": 1, "path": "results/snp_sweep/.../pair_kl_g01_k05_p08_tsp01.json"},
                {"tsp": 80, "path": "results/snp_sweep/.../pair_kl_g01_k05_p08_tsp80.json"},
                ...
              ]
            },
            ...
          ],
          "ungrouped": ["path/to/mask_without_tsp.json", ...]
        }
        """
        paths = set()
        for pattern in ACTIVE_MASK_GLOBS:
            paths.update(glob(str(ROOT / pattern), recursive=True))
        paths = [
            p for p in sorted(paths)
            if not p.endswith(".eval.json")
            and not any(exc in p for exc in ACTIVE_MASK_EXCLUDES)
        ]
        rel_paths = [str(Path(p).relative_to(ROOT)) for p in paths]

        tsp_re = re.compile(r'^(.+?)_tsp(\d+)\.json$')
        groups: dict[str, list] = defaultdict(list)
        ungrouped: list[str] = []

        for rp in rel_paths:
            fname = Path(rp).name
            m = tsp_re.match(fname)
            if m:
                parent = str(Path(rp).parent)
                group_key = parent + "/" + m.group(1)
                tsp = int(m.group(2))
                groups[group_key].append({"tsp": tsp, "path": rp})
            else:
                ungrouped.append(rp)

        result_groups = []
        for key in sorted(groups):
            entries = sorted(groups[key], key=lambda e: e["tsp"])
            # Use just the stem (without parent) as the label
            label = key.split("/")[-1]
            result_groups.append({"label": label, "dir": key, "sparsities": entries})

        self._json_response({"groups": result_groups, "ungrouped": ungrouped})

    def _serve_mask(self, rel_path: str):
        full = ROOT / rel_path
        if not full.exists():
            self._json_response({"error": f"not found: {rel_path}"}, 404)
            return
        # Ensure it's under the project root (security)
        try:
            full.resolve().relative_to(ROOT.resolve())
        except ValueError:
            self._json_response({"error": "path outside project"}, 403)
            return

        with open(full) as f:
            data = json.load(f)
        self._json_response(data)

    def _serve_resample(self, rel_path: str):
        """Serve the resample sidecar for a mask, if it exists."""
        mask_full = ROOT / rel_path
        stem = mask_full.stem
        sidecar = mask_full.with_name(f"{stem}_resample.json")
        if not sidecar.exists():
            self._json_response(None)
            return
        try:
            sidecar.resolve().relative_to(ROOT.resolve())
        except ValueError:
            self._json_response({"error": "path outside project"}, 403)
            return
        with open(sidecar) as f:
            data = json.load(f)
        self._json_response(data)

    @staticmethod
    def _find_rollout_sidecar(mask_full: Path) -> "Path | None":
        """Locate the rollout sidecar for a mask.

        Convention: ``<parent>_rollouts/<stem>_rollout.json``
        """
        stem = mask_full.stem
        parent = mask_full.parent
        candidate = parent.parent / f"{parent.name}_rollouts" / f"{stem}_rollout.json"
        return candidate if candidate.exists() else None

    @staticmethod
    def _summarize_rollout(rollout_path: Path) -> "dict | None":
        """Read a rollout file and return a summary row compatible with eval rows."""
        try:
            with open(rollout_path) as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
        masked = d.get("masked_rollouts", [])
        pcs = [r["p_correct"] for r in masked if isinstance(r, dict) and "p_correct" in r]
        kls = [r["kl_vs_clean"] for r in masked if isinstance(r, dict) and "kl_vs_clean" in r]
        if not pcs:
            return None
        return {
            "sparsity": d.get("target_sparsity", 0),
            "p_target": sum(pcs) / len(pcs),
            "kl": sum(kls) / len(kls) if kls else None,
            "n_rollouts": len(pcs),
            "row": "rollout",
            "mode": "top_k",
        }

    @staticmethod
    def _find_eval_sidecar(mask_full: Path) -> "Path | None":
        """Locate the eval sidecar for a mask, trying several conventions.

        1. ``*/masks/*.json``  → ``<parent>/../eval/<stem>.eval.json``
        2. Same directory      → ``<stem>_eval.json``
        3. Sibling ``_eval/``  → ``<parent>_eval/<stem>.eval.json``
        """
        stem = mask_full.stem
        parent = mask_full.parent

        candidates = []
        if parent.name == "masks":
            candidates.append(parent.parent / "eval" / f"{stem}.eval.json")
        candidates.append(parent / f"{stem}_eval.json")
        candidates.append(
            parent.parent / f"{parent.name}_eval" / f"{stem}.eval.json"
        )

        for c in candidates:
            if c.exists():
                return c
        return None

    def _serve_eval_sidecar(self, rel_path: str):
        """Serve the eval sidecar JSON for a mask, if it exists."""
        mask_full = (ROOT / rel_path).resolve()
        try:
            mask_full.relative_to(ROOT.resolve())
        except ValueError:
            self._json_response({"error": "path outside project"}, 403)
            return

        sidecar = self._find_eval_sidecar(mask_full)
        if sidecar is None:
            self._json_response(None)
            return

        try:
            sidecar.resolve().relative_to(ROOT.resolve())
        except ValueError:
            self._json_response({"error": "path outside project"}, 403)
            return

        with open(sidecar) as f:
            data = json.load(f)
        self._json_response(data)

    def _serve_group_evals(self, mask_paths: list[str]):
        """Load eval sidecars for a list of mask paths and return them all.

        Also attempts to find a TA baseline eval for the same prompt.
        Returns::

            {
                "evals": [
                    {"mask_path": "...", "tsp": 80, "eval": <eval JSON or null>},
                    ...
                ],
                "ta_eval": <eval JSON or null>,
                "clean_answer_probs": [...],
                "kl_max": float,
                "target_letter": "A",
                "answer_letters": ["A","B","C","D"],
                "prompt_index": int
            }
        """
        evals = []
        prompt_index = None
        kl_max = None
        clean_probs = None
        target_letter = None
        answer_letters = None

        for mp in mask_paths:
            mask_full = (ROOT / mp).resolve()
            try:
                mask_full.relative_to(ROOT.resolve())
            except ValueError:
                evals.append({"mask_path": mp, "eval": None})
                continue

            sidecar = self._find_eval_sidecar(mask_full)
            if sidecar is None:
                evals.append({"mask_path": mp, "eval": None})
                continue

            try:
                with open(sidecar) as f:
                    data = json.load(f)
                evals.append({"mask_path": mp, "eval": data})
                if prompt_index is None:
                    prompt_index = data.get("prompt_index")
                if kl_max is None:
                    kl_max = data.get("kl_max")
                if clean_probs is None:
                    clean_probs = data.get("clean_answer_probs")
                if target_letter is None:
                    target_letter = data.get("target_letter")
                if answer_letters is None:
                    answer_letters = data.get("answer_letters")
            except (OSError, json.JSONDecodeError):
                evals.append({"mask_path": mp, "eval": None})

        # Try to find a TA baseline for this prompt.
        # When the SNP group freezes the prompt, use the frozen-prompt TA
        # eval so both methods are compared on equal footing (prompt edges
        # always on).
        ta_eval = None
        if prompt_index is not None:
            frozen = any("frozen_prompt" in mp for mp in mask_paths)
            mp0 = mask_paths[0] if mask_paths else ""
            # The TA overlay must come from the same dataset as the mask
            # group; matching by prompt index alone silently overlays a
            # different dataset's curve.
            if "aqua_reward_gap/masks_train_split" in mp0:
                ta_patterns = [
                    f"results/aqua_reward_gap/ta_eval_train_split_frozen/aqua_train_ta_g01_k05_p{prompt_index:02d}_thought_anchors.eval.json",
                    f"results/aqua_reward_gap/ta_eval_train_split/aqua_train_ta_g01_k05_p{prompt_index:02d}_thought_anchors.eval.json",
                ]
            elif "aqua_reward_gap" in mp0:
                # AQuA reward-gap masks are frozen-prompt; prefer the
                # frozen-prompt TA evaluation.
                ta_patterns = [
                    f"results/aqua_reward_gap/ta_eval_frozen/aqua_ta_g01_k05_p{prompt_index:02d}_thought_anchors.eval.json",
                    f"results/aqua_reward_gap/ta_eval/aqua_ta_g01_k05_p{prompt_index:02d}_thought_anchors.eval.json",
                ]
            elif "math_reward_gap" in mp0:
                late = any("late_p" in mp for mp in mask_paths)
                ta_dirs = (
                    ["ta_late_eval", "ta_eval"] if late else ["ta_eval"]
                )
                ta_patterns = [
                    f"results/math_reward_gap/{d}/math_ta_{'late_' if 'late' in d else 'g01_k05_'}p{prompt_index:02d}_thought_anchors.eval.json"
                    for d in ta_dirs
                ]
            else:
                ta_eval_dirs = (
                    ["ta_g01_k05_eval_frozen_prompt", "ta_g01_k05_eval"]
                    if frozen
                    else ["ta_g01_k05_eval"]
                )
                ta_patterns = [
                    f"results/snp_sweep/{d}/ta_g01_k05_p{prompt_index:02d}_thought_anchors.eval.json"
                    for d in ta_eval_dirs
                ] + [
                    f"results/snp_sweep/{d}/ta_g01_k05_p{prompt_index}_thought_anchors.eval.json"
                    for d in ta_eval_dirs
                ]
            for tp in ta_patterns:
                ta_path = ROOT / tp
                if ta_path.exists():
                    try:
                        with open(ta_path) as f:
                            ta_eval = json.load(f)
                    except (OSError, json.JSONDecodeError):
                        pass
                    break

        # Load rollout sidecars for SNP masks
        snp_rollouts = []
        for mp in mask_paths:
            mask_full = (ROOT / mp).resolve()
            try:
                mask_full.relative_to(ROOT.resolve())
            except ValueError:
                continue
            rollout_path = self._find_rollout_sidecar(mask_full)
            if rollout_path is not None:
                summary = self._summarize_rollout(rollout_path)
                if summary is not None:
                    snp_rollouts.append(summary)
        snp_rollouts.sort(key=lambda r: r["sparsity"])

        # Load TA rollout sidecars (GPQA snp_sweep layout only — other
        # datasets' TA rollouts get their own paths when they exist;
        # never overlay another dataset's rollouts by prompt index).
        ta_rollouts = []
        _mp0 = mask_paths[0] if mask_paths else ""
        if prompt_index is not None and "snp_sweep" in _mp0:
            frozen = any("frozen_prompt" in mp for mp in mask_paths)
            ta_rollout_dirs = (
                ["ta_g01_k05_rollouts_frozen_prompt", "ta_g01_k05_rollouts"]
                if frozen
                else ["ta_g01_k05_rollouts"]
            )
            sparsities = [1, 5, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
            for d in ta_rollout_dirs:
                found_any = False
                for tsp in sparsities:
                    tp = ROOT / f"results/snp_sweep/{d}/ta_g01_k05_p{prompt_index:02d}_tsp{tsp:02d}_rollout.json"
                    if not tp.exists():
                        continue
                    found_any = True
                    summary = self._summarize_rollout(tp)
                    if summary is not None:
                        ta_rollouts.append(summary)
                if found_any:
                    break
            ta_rollouts.sort(key=lambda r: r["sparsity"])

        self._json_response({
            "evals": evals,
            "ta_eval": ta_eval,
            "snp_rollouts": snp_rollouts if snp_rollouts else None,
            "ta_rollouts": ta_rollouts if ta_rollouts else None,
            "clean_answer_probs": clean_probs,
            "kl_max": kl_max,
            "target_letter": target_letter,
            "answer_letters": answer_letters,
            "prompt_index": prompt_index,
        })

    def _serve_resample_csv(self):
        """Stream the matplotlib-side CSV alongside the resample plot."""
        path = ROOT / "notes/images/resample_evals/resample_fraction_correct_vs_sparsity.csv"
        if not path.exists():
            self._json_response({"error": f"not found: {path.relative_to(ROOT)}"}, 404)
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/csv")
        self.send_header(
            "Content-Disposition",
            'attachment; filename="resample_fraction_correct_vs_sparsity.csv"',
        )
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _serve_resample_sweep(self):
        """Aggregate every resample sidecar + single-sparsity result file under
        ``RESAMPLE_DIR`` into one payload for the dashboard plot. Bootstrap CIs
        are computed server-side so the JS just consumes pre-baked numbers.
        """
        try:
            import numpy as np
        except ImportError:
            self._json_response(
                {"error": "numpy required for /api/resample_sweep"}, 500
            )
            return

        rng = np.random.default_rng(RESAMPLE_BOOTSTRAP_SEED)

        def _bootstrap_ci(answers, target):
            n = len(answers)
            if target is None or n < 2:
                return None, None
            correct = np.asarray([1.0 if a == target else 0.0 for a in answers])
            idx = rng.integers(0, n, size=(RESAMPLE_BOOTSTRAP_B, n))
            samples = correct[idx].mean(axis=1)
            return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))

        if not RESAMPLE_DIR.exists():
            self._json_response({"series": [], "warning": f"{RESAMPLE_DIR} missing"})
            return

        series = []

        # Multi-threshold sidecars: one curve per file.
        for path in sorted(RESAMPLE_DIR.glob("*_resample.json")):
            try:
                with open(path) as f:
                    d = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            target = d.get("correct_answer_normalized")
            baseline = d.get("original", {}).get("fraction_correct")
            name = path.stem.removesuffix("_resample")
            points = []
            for t in d.get("thresholds", []):
                sp = t.get("sparsity")
                fc = t.get("resample_fraction_correct")
                if sp is None or fc is None:
                    continue
                answers = [b.get("answer") for b in t.get("resample_branches", [])]
                lo, hi = _bootstrap_ci(answers, target)
                points.append({
                    "sparsity": sp,
                    "fraction_correct": fc,
                    "ci_lo": lo if lo is not None else fc,
                    "ci_hi": hi if hi is not None else fc,
                    "n_resample_branches": len(answers),
                })
            if points:
                points.sort(key=lambda p: p["sparsity"])
                series.append({
                    "kind": "line",
                    "task_name": name,
                    "source_file": str(path.relative_to(ROOT)),
                    "mask_path": d.get("mask_path"),
                    "baseline_fraction_correct": baseline,
                    "correct_answer": d.get("correct_answer"),
                    "points": points,
                })

        # Single-sparsity result files: one point per file, grouped by parent
        # directory (which is the task name).
        for sub in sorted(p for p in RESAMPLE_DIR.iterdir() if p.is_dir()):
            for path in sorted(sub.glob("*_results.json")):
                try:
                    with open(path) as f:
                        d = json.load(f)
                except (OSError, json.JSONDecodeError):
                    continue
                target = d.get("correct_answer_normalized")
                baseline = d.get("original", {}).get("fraction_correct")
                entries = d.get("thresholds", [])
                if not entries:
                    continue
                entry = entries[0]
                sp = entry.get("sparsity")
                fc = entry.get("resample_fraction_correct")
                if sp is None or fc is None:
                    continue
                answers = [b.get("answer") for b in entry.get("resample_branches", [])]
                lo, hi = _bootstrap_ci(answers, target)
                series.append({
                    "kind": "standalone",
                    "task_name": sub.name,
                    "source_file": str(path.relative_to(ROOT)),
                    "mask_path": d.get("mask_path"),
                    "baseline_fraction_correct": baseline,
                    "correct_answer": d.get("correct_answer"),
                    "points": [{
                        "sparsity": sp,
                        "fraction_correct": fc,
                        "ci_lo": lo if lo is not None else fc,
                        "ci_hi": hi if hi is not None else fc,
                        "n_resample_branches": len(answers),
                        "elapsed_seconds": entry.get("elapsed_seconds"),
                    }],
                })

        self._json_response({
            "series": series,
            "bootstrap_B": RESAMPLE_BOOTSTRAP_B,
            "ci_low": 2.5,
            "ci_high": 97.5,
        })

    def _json_response(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        # Quieter logging
        if "/api/" in (str(args[0]) if args else ""):
            return
        super().log_message(format, *args)


def main():
    global ACTIVE_INDEX_FILE, ACTIVE_MASK_GLOBS, ACTIVE_MODE_LABEL

    parser = argparse.ArgumentParser(
        description="Serve the circuit tracer dashboard.",
    )
    parser.add_argument(
        "port",
        nargs="?",
        type=int,
        default=None,
        help="Port to listen on (default: $PORT or 8765)",
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--global",
        dest="mode",
        action="store_const",
        const="global",
        help="Default mode: serve index.html with circuit_discovery masks",
    )
    mode_group.add_argument(
        "--direct-answer",
        dest="mode",
        action="store_const",
        const="direct-answer",
        help="Serve direct_answer.html with SNP sweep + direct-answer masks",
    )
    mode_group.add_argument(
        "--legacy",
        dest="mode",
        action="store_const",
        const="legacy",
        help="Serve index_legacy.html (frozen copy, no shared.js dependency)",
    )
    parser.set_defaults(mode="global")

    args = parser.parse_args()

    # ── Resolve port ──────────────────────────────────────────────────
    port = args.port or int(os.environ.get("PORT", "8765"))

    # ── Configure mode ────────────────────────────────────────────────
    if args.mode == "direct-answer":
        ACTIVE_INDEX_FILE = "direct_answer.html"
        ACTIVE_MASK_GLOBS = [
            "results/snp_sweep/**/*.json",
            "results/circuit_discovery/direct_answer_circuit_discovery/**/*.json",
            "results/aqua_reward_gap/**/*.json",
            "results/math_reward_gap/**/*.json",
        ]
        ACTIVE_MODE_LABEL = "direct-answer"
    elif args.mode == "legacy":
        ACTIVE_INDEX_FILE = "index_legacy.html"
        ACTIVE_MASK_GLOBS = ["results/circuit_discovery/**/*.json"]
        ACTIVE_MODE_LABEL = "legacy"
    else:  # global (default)
        ACTIVE_INDEX_FILE = "index.html"
        ACTIVE_MASK_GLOBS = ["results/circuit_discovery/**/*.json"]
        ACTIVE_MASK_EXCLUDES = ["direct_answer_circuit_discovery"]
        ACTIVE_MODE_LABEL = "global"

    # ── Start server ──────────────────────────────────────────────────
    server = HTTPServer(("0.0.0.0", port), DashboardHandler)
    print(f"Circuit Tracer Dashboard: http://localhost:{port}")
    print(f"  Mode : {ACTIVE_MODE_LABEL}")
    print(f"  Index: {ACTIVE_INDEX_FILE}")
    print(f"  Globs: {ACTIVE_MASK_GLOBS}")
    print(f"  Root : {ROOT}")

    # ── ngrok tunnel ──────────────────────────────────────────────────
    tunnel = None
    try:
        from pyngrok import ngrok
        tunnel = ngrok.connect(port, "http")
        print(f"  ngrok: {tunnel.public_url}")
    except Exception as e:
        print(f"  ngrok: unavailable ({e})")

    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        if tunnel:
            from pyngrok import ngrok as _ngrok
            _ngrok.disconnect(tunnel.public_url)
        server.shutdown()


if __name__ == "__main__":
    main()
