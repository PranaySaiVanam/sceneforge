"""Zero-dependency dev server (Python stdlib only) with the same routes as app.py.

    python devserver.py --port 8000
"""
import argparse
import gzip
import json
import mimetypes
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from sceneforge import api

import threading
import time

RATE_LIMIT = int(os.getenv("SCENEFORGE_RATE_LIMIT", "20"))  # scenes per visitor per minute
_hits: dict[str, list[float]] = {}
_lock = threading.Lock()


def allowed(ip: str) -> bool:
    """Sliding one-minute window per client IP, so a public demo can't run up an LLM bill."""
    now = time.time()
    with _lock:
        recent = [t for t in _hits.get(ip, []) if now - t < 60]
        if len(recent) >= RATE_LIMIT:
            _hits[ip] = recent
            return False
        recent.append(now)
        _hits[ip] = recent
        if len(_hits) > 10_000:  # keep memory bounded
            _hits.clear()
        return True


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: bytes, ctype: str, cache: bool = False):
        if "gzip" in self.headers.get("accept-encoding", "") and len(body) > 1024:
            body = gzip.compress(body, 5)
            gz = True
        else:
            gz = False
        self.send_response(status)
        self.send_header("content-type", ctype)
        self.send_header("content-length", str(len(body)))
        if gz:
            self.send_header("content-encoding", "gzip")
        self.send_header("cache-control", "public, max-age=86400" if cache else "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, obj):
        self._send(status, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/api/health":
            return self._json(200, api.health())
        if path == "/api/providers":
            return self._json(200, api.providers())
        if path == "/api/library":
            return self._json(200, api.library())
        if path.startswith("/motions/") and path.endswith(".json"):
            p = api.motion_path(path[len("/motions/"):-5])
            if not p:
                return self._json(404, {"error": "unknown clip"})
            return self._send(200, p.read_bytes(), "application/json", cache=True)
        rel = "index.html" if path in ("/", "") else path.lstrip("/")
        f = (api.FRONTEND / rel).resolve()
        if api.FRONTEND.resolve() not in f.parents or not f.is_file():
            return self._json(404, {"error": "not found"})
        ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
        if f.suffix == ".js":
            ctype = "text/javascript"
        self._send(200, f.read_bytes(), ctype)

    def do_POST(self):
        if self.path.split("?")[0] != "/api/direct":
            return self._json(404, {"error": "not found"})
        ip = (self.headers.get("x-forwarded-for") or self.client_address[0]).split(",")[0].strip()
        if not allowed(ip):
            return self._json(429, {"error": "Too many scenes in a minute. Wait a moment and try again."})
        n = int(self.headers.get("content-length", 0))
        try:
            body = json.loads(self.rfile.read(min(n, 100_000)) or b"{}")
        except json.JSONDecodeError:
            return self._json(400, {"error": "invalid JSON"})
        status, out = api.direct_scene(body)
        self._json(status, out)

    def log_message(self, fmt, *args):
        print(f"[{self.log_date_time_string()}] {fmt % args}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # Hosting platforms (Render, Railway, Fly) pass the port in $PORT and need 0.0.0.0.
    ap.add_argument("--host", default=os.getenv("HOST", "0.0.0.0" if os.getenv("PORT") else "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    a = ap.parse_args()
    print(f"SceneForge dev server on http://{a.host}:{a.port}")
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()
