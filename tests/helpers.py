"""Small HTTP service fake shared by adapter and application tests."""

import json
import threading
from contextlib import contextmanager
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = "prism-ml/ternary-bonsai-2-2b"
NOTE = "S — Subjetivo: Palpitaciones.\nO — Objetivo: No consta.\nA — Análisis: No consta.\nP — Plan: No consta."


@contextmanager
def serve(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


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

    with serve(Handler) as url:
        yield url, requests


@contextmanager
def transcription_server(data, *, status=200):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            header = f"Content-Type: {self.headers['Content-Type']}\r\n\r\n".encode()
            message = BytesParser(policy=policy.default).parsebytes(header + raw)
            fields = {}
            for part in message.iter_parts():
                name = part.get_param("name", header="content-disposition")
                body = part.get_payload(decode=True)
                fields[name] = len(body) if name == "file" else body.decode()
            requests.append((self.path, fields))
            payload = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    with serve(Handler) as url:
        yield url, requests
