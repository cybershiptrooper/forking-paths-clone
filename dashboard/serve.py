#!/usr/bin/env python3
"""Serve the circuit tracer dashboard with optional NodeMask JSON API."""

from __future__ import annotations

import json
import sys
from glob import glob
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_DIR = Path(__file__).resolve().parent
DEFAULT_MASK_GLOB = "results/circuitviz/**/*.json"


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
        if "/api/" in (args[0] if args else ""):
            return
        super().log_message(format, *args)


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
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
