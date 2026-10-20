"""Small HTTP service fake shared by adapter and application tests."""

import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = "prism-ml/ternary-bonsai-2-2b"
NOTE = "S — Subjetivo: Palpitaciones.\nO — Objetivo: No consta.\nA — Análisis: No consta.\nP — Plan: No consta."


@contextmanager
def note_server(*, models=None, content=NOTE, finish_reason="stop"):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, value):
            payload = json.dumps(value).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            self.respond(
                {
                    "object": "list",
                    "data": [
                        {
                            "id": model,
                            "object": "model",
                            "created": 0,
                            "owned_by": "local",
                        }
                        for model in (models if models is not None else [MODEL])
                    ],
                }
            )

        def do_POST(self):
            requests.append(
                json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            )
            self.respond(
                {
                    "id": "test",
                    "object": "chat.completion",
                    "created": 0,
                    "model": MODEL,
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": finish_reason,
                            "message": {"role": "assistant", "content": content},
                        }
                    ],
                }
            )

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
