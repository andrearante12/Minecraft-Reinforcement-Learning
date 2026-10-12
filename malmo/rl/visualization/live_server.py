"""
visualization/live_server.py
----------------------------
Stdlib-only HTTP + Server-Sent-Events publisher for the live split-screen page
(site/live.html): Minecraft camera on the left, world-model predictions on the
right. Shared by training/train.py (--viz-port) and visualization/live_split.py
(checkpoint playback).

No third-party web deps (train_env has no aiohttp). The server runs in a daemon
thread; publish_*() never blocks the caller. Camera frames are latest-wins per
browser (a slow tab skips frames rather than lagging); schema / status / stats
events are queued, and stats history is replayed to late joiners.

    server = LiveServer(8765)
    server.publish_schema({"type": "schema", "kind": "vector", ...})
    if server.has_clients:
        server.publish_frame(frame_dict)
"""

import io
import os
import json
import base64
import mimetypes
import threading
import collections
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SITE_DIR = os.path.join(REPO_ROOT, "site")
KEEPALIVE_SEC = 15.0
STATS_HISTORY = 5000      # per kind; replayed to a page opened mid-run


class _Client:
    """Per-browser outbox. Frames are latest-wins (a slow browser skips camera
    frames instead of lagging); schema/status/stats are queued and never dropped."""

    def __init__(self):
        self.cond = threading.Condition()
        self.frame = None
        self.events = collections.deque(maxlen=STATS_HISTORY)

    def put(self, msg, is_frame):
        with self.cond:
            if is_frame:
                self.frame = msg
            else:
                self.events.append(msg)
            self.cond.notify()

    def take(self, timeout):
        """Next batch of messages to write (events first), or [] on timeout."""
        with self.cond:
            if self.frame is None and not self.events:
                self.cond.wait(timeout)
            out = list(self.events)
            self.events.clear()
            if self.frame is not None:
                out.append(self.frame)
                self.frame = None
            return out


def encode_jpeg(frame, quality=80, max_width=None):
    """(H,W,C) uint8 or [0,1] float array -> base64 JPEG string (None if no frame)."""
    if frame is None:
        return None
    from PIL import Image
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    img = Image.fromarray(arr)
    if max_width and img.width > max_width:
        img = img.resize((max_width, int(round(img.height * max_width / img.width))), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _sse(event, payload):
    return "event: {0}\ndata: {1}\n\n".format(event, json.dumps(payload, separators=(",", ":"))).encode("utf-8")


class LiveServer:
    def __init__(self, http_port=8765, host="127.0.0.1", site_dir=SITE_DIR):
        self.site_dir = os.path.abspath(site_dir)
        self._lock = threading.Lock()
        self._clients = set()
        self._schema = None
        self._latest_frame = None
        self._latest_status = None
        self._stats = collections.defaultdict(lambda: collections.deque(maxlen=STATS_HISTORY))

        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):  # keep the training console clean
                pass

            def do_GET(self):
                path = self.path.split("?", 1)[0]
                if path == "/events":
                    server._serve_events(self)
                elif path in ("/", "/index.html"):
                    server._serve_file(self, "live.html")
                else:
                    server._serve_file(self, path.lstrip("/"))

        self.httpd = ThreadingHTTPServer((host, http_port), Handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.url = "http://localhost:{0}/".format(self.port)
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self._thread.start()

    # ── publishing (any thread) ──────────────────────────────────────────────
    @property
    def has_clients(self):
        return bool(self._clients)

    def publish_schema(self, schema):
        with self._lock:
            self._schema = schema
        self._fanout(_sse("schema", schema))

    def publish_frame(self, frame):
        with self._lock:
            self._latest_frame = frame
        self._fanout(_sse("frame", frame), is_frame=True)

    def publish_status(self, phase, text=""):
        status = {"phase": phase, "text": text}
        with self._lock:
            self._latest_status = status
        self._fanout(_sse("status", status))

    def publish_stats(self, kind, data):
        """kind: "episode" (one per finished episode) or "update" (one per agent.update()).
        Kept server-side so a page opened mid-run gets the whole curve."""
        item = dict(data, kind=kind)
        with self._lock:
            self._stats[kind].append(item)
        self._fanout(_sse("stats", item))

    def _fanout(self, msg, is_frame=False):
        with self._lock:
            clients = list(self._clients)
        for c in clients:
            c.put(msg, is_frame)

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    # ── request handling (server threads) ────────────────────────────────────
    def _serve_events(self, handler):
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Cache-Control", "no-cache")
        handler.send_header("Connection", "keep-alive")
        handler.end_headers()

        client = _Client()
        with self._lock:
            hello = []
            if self._schema is not None:
                hello.append(_sse("schema", self._schema))
            if self._latest_status is not None:
                hello.append(_sse("status", self._latest_status))
            history = [item for kind in self._stats for item in self._stats[kind]]
            if history:
                hello.append(_sse("stats_history", history))
            if self._latest_frame is not None:
                hello.append(_sse("frame", self._latest_frame))
            self._clients.add(client)
        try:
            for msg in hello:
                handler.wfile.write(msg)
            handler.wfile.flush()
            while True:
                batch = client.take(KEEPALIVE_SEC) or [b": keepalive\n\n"]
                for msg in batch:
                    handler.wfile.write(msg)
                handler.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            with self._lock:
                self._clients.discard(client)

    def _serve_file(self, handler, rel):
        full = os.path.abspath(os.path.join(self.site_dir, rel))
        if not full.startswith(self.site_dir + os.sep) or not os.path.isfile(full):
            handler.send_error(404)
            return
        ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
        with open(full, "rb") as f:
            body = f.read()
        handler.send_response(200)
        handler.send_header("Content-Type", ctype)
        handler.send_header("Content-Length", str(len(body)))
        handler.send_header("Cache-Control", "no-cache")
        handler.end_headers()
        handler.wfile.write(body)
