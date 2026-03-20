#!/usr/bin/env python3
"""PI05 inference server for ARX deployment."""

from __future__ import annotations

import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SERVER_ROOT = Path(__file__).resolve().parent
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

from pi05_protocol import dumps_json, loads_json  # noqa: E402
from server_utils import (  # noqa: E402
    DEFAULT_PI05_MODEL_PATH,
    PI05InferenceService,
    create_pi05_inference_service,
)


class PI05RequestHandler(BaseHTTPRequestHandler):
    server_version = "PI05Server/1.0"

    @property
    def service(self) -> PI05InferenceService:
        return self.server.service  # type: ignore[attr-defined]

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = dumps_json(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path not in {"/health", "/meta"}:
            self._send_json({"error": f"Unknown path: {self.path}"}, status=404)
            return
        self._send_json(self.service.metadata())

    def do_POST(self) -> None:
        if self.path != "/infer":
            self._send_json({"error": f"Unknown path: {self.path}"}, status=404)
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_length)
            request = loads_json(raw)
            response = self.service.infer(request)
            self._send_json(response, status=200)
        except Exception as exc:
            self._send_json(
                {
                    "error": type(exc).__name__,
                    "message": str(exc),
                },
                status=500,
            )

    def log_message(self, format: str, *args) -> None:
        message = "%s - - [%s] %s\n" % (
            self.address_string(),
            self.log_date_time_string(),
            format % args,
        )
        sys.stdout.write(message)
        sys.stdout.flush()


def run_pi05_server(
    host: str = "0.0.0.0",
    port: int = 8005,
    model_path: str = DEFAULT_PI05_MODEL_PATH,
    device: str = "cuda",
) -> None:
    service = create_pi05_inference_service(
        model_path=model_path,
        device=device,
    )

    httpd = ThreadingHTTPServer((host, port), PI05RequestHandler)
    httpd.service = service  # type: ignore[attr-defined]
    meta = service.metadata()
    print(
        f"PI05 server listening on {host}:{port} | "
        f"model={meta['model_path']} | device={meta['device']} | "
        f"rgb={meta['rgb_camera_keys']} | depth={meta['depth_camera_keys']} | "
        f"action_dim={meta['action_dim']} | chunk={meta['server_chunk_length']} | "
        f"default_task={meta['default_task']!r}"
    )
    httpd.serve_forever()


def main() -> None:
    run_pi05_server(
        host="0.0.0.0",
        port=8005,
        model_path=DEFAULT_PI05_MODEL_PATH,
        device="cuda",
    )


if __name__ == "__main__":
    main()
