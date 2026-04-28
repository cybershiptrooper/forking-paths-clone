#!/usr/bin/env python3
"""Serve the circuit tracer dashboard with optional NodeMask JSON API."""

from __future__ import annotations

import json
import os
import sys
from glob import glob
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_DIR = Path(__file__).resolve().parent
DEFAULT_MASK_GLOB = "results/circuitviz/**/*.json"
DEFAULT_MASK_GLOB = "results/circuit_discovery/**/*.json"
RESAMPLE_DIR = ROOT / "results/circuit_discovery/tempered_snis"
RESAMPLE_BOOTSTRAP_B = 10_000
RESAMPLE_BOOTSTRAP_SEED = 0


class DashboardHandler(SimpleHTTPRequestHandler):
    """Serves static files from dashboard/ and provides a mask API."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DASHBOARD_DIR), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/masks":
            self._serve_mask_list()
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
        elif parsed.path == "/api/resample_sweep":
            self._serve_resample_sweep()
        elif parsed.path == "/api/resample_csv":
            self._serve_resample_csv()
        else:
            super().do_GET()

    def _serve_mask_list(self):
        paths = sorted(glob(str(ROOT / DEFAULT_MASK_GLOB), recursive=True))
        # Return relative paths from project root
        rel = [str(Path(p).relative_to(ROOT)) for p in paths]
        self._json_response(rel)

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
    port = int(os.environ.get("PORT", sys.argv[1] if len(sys.argv) > 1 else "8765"))
    server = HTTPServer(("0.0.0.0", port), DashboardHandler)
    print(f"Circuit Tracer Dashboard: http://localhost:{port}")
    print(f"Serving masks from: {ROOT}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
