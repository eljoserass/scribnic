"""Check the actual HTTP note contract without loading speech models."""

import json
import io
import sys
import tempfile
import threading
import unittest
import wave
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from scribnic.backends.generation import LMStudioTextGenerator
from scribnic.core import Pipeline
from scribnic.cli import main
from scribnic.models import Audio, Utterance
from scribnic.prompts import SOAP_INSTRUCTION


MODEL = "prism-ml/ternary-bonsai-2-2b"
NOTE = "S — Subjetivo: Palpitaciones.\nO — Objetivo: No consta.\nA — Análisis: No consta.\nP — Plan: No consta."
TURNS = (Utterance(0, 2, "Tengo palpitaciones desde ayer.", "speaker_1", ("speaker_1",)),)


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
            self.respond({"object": "list", "data": [
                {"id": model, "object": "model", "created": 0, "owned_by": "local"}
                for model in (models if models is not None else [MODEL])
            ]})

        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.respond({
                "id": "test", "object": "chat.completion", "created": 0, "model": MODEL,
                "choices": [{"index": 0, "finish_reason": finish_reason,
                             "message": {"role": "assistant", "content": content}}],
            })

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


class GenerationTests(unittest.TestCase):
    def test_spanish_soap_over_http(self):
        with note_server() as (url, requests):
            generator = LMStudioTextGenerator("auto", url)
            self.assertEqual(generator.check_available(), MODEL)
            self.assertEqual(generator.generate(TURNS, SOAP_INSTRUCTION), NOTE)
        request = requests[0]
        self.assertEqual(request["model"], MODEL)
        self.assertEqual(request["max_tokens"], 768)
        self.assertEqual(request["reasoning_effort"], "none")
        self.assertFalse(request["chat_template_kwargs"]["enable_thinking"])
        self.assertNotIn("response_format", request)
        self.assertIn("SOAP breve en español", request["messages"][0]["content"])
        self.assertIn("No consta", request["messages"][0]["content"])
        self.assertIn("[speaker_1] Tengo palpitaciones desde ayer.", request["messages"][1]["content"])
        self.assertNotIn('"candidates"', request["messages"][1]["content"])

    def test_auto_never_falls_back_to_another_model(self):
        with note_server(models=["ternary-bonsai-2-4b", "bonsai-1.7b"]) as (url, requests):
            with self.assertRaisesRegex(ValueError, "Expected one Ternary Bonsai 2 2B"):
                LMStudioTextGenerator("auto", url).check_available()
            self.assertFalse(requests)

    def test_multiple_variants_require_explicit_selection(self):
        with note_server(models=[MODEL, MODEL + "-q2_0"]) as (url, _):
            with self.assertRaisesRegex(ValueError, "exact ID"):
                LMStudioTextGenerator("auto", url).check_available()
            self.assertEqual(LMStudioTextGenerator(MODEL, url).check_available(), MODEL)

    def test_long_transcript_is_rejected_before_request(self):
        with note_server() as (url, requests):
            generator = LMStudioTextGenerator(MODEL, url, max_input_chars=5)
            with self.assertRaisesRegex(ValueError, "Nothing was truncated"):
                generator.generate(TURNS, SOAP_INSTRUCTION)
            self.assertFalse(requests)

    def test_incomplete_note_is_not_success(self):
        with note_server(finish_reason="length") as (url, _):
            with self.assertRaisesRegex(ValueError, "output limit"):
                LMStudioTextGenerator(MODEL, url).generate(TURNS, SOAP_INSTRUCTION)

    def test_empty_or_thinking_response_is_not_a_note(self):
        for content in ("  ", "<think>razonamiento</think>"):
            with self.subTest(content=content), note_server(content=content) as (url, _):
                with self.assertRaises(ValueError):
                    LMStudioTextGenerator(MODEL, url).generate(TURNS, SOAP_INSTRUCTION)

    def test_speech_checkpoint_survives_generation_failure(self):
        class Source:
            def load(self):
                return Audio(Path("unused.wav"), 16000, 2)

        class Recognizer:
            def recognize(self, audio):
                return TURNS

        checkpoints = []
        stages = []
        with note_server(finish_reason="length") as (url, _):
            pipeline = Pipeline(recognizer=Recognizer(), generator=LMStudioTextGenerator(MODEL, url))
            with self.assertRaises(ValueError):
                pipeline.run(Source(), SOAP_INSTRUCTION, on_progress=stages.append,
                             on_transcript=checkpoints.append)
        self.assertEqual(checkpoints[0].utterances, TURNS)
        self.assertIsNone(checkpoints[0].generated_text)
        self.assertEqual(stages[-1], "generation")

    def test_cli_saves_transcript_and_error_report_when_note_fails(self):
        class Recognizer:
            def recognize(self, audio):
                return TURNS

        with tempfile.TemporaryDirectory() as directory, note_server(finish_reason="length") as (url, _):
            root = Path(directory)
            audio = root / "input.wav"
            with wave.open(str(audio), "wb") as wav:
                wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                wav.writeframes(b"\0\0" * 32000)
            output = root / "run"
            pipeline = Pipeline(recognizer=Recognizer(), generator=LMStudioTextGenerator(MODEL, url))
            stdout, stderr = io.StringIO(), io.StringIO()
            argv = ["scribnic", str(audio), "--soap", "--base-url", url,
                    "--format", "json", "--output-dir", str(output)]
            with patch.object(sys, "argv", argv), patch("scribnic.cli.build_pipeline", return_value=pipeline), \
                    redirect_stdout(stdout), redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as exit:
                    main()
            self.assertEqual(exit.exception.code, 1)
            self.assertEqual(stdout.getvalue(), "")
            saved = json.loads((output / "transcript.json").read_text())
            self.assertEqual(saved["utterances"][0]["text"], TURNS[0].text)
            self.assertIsNone(saved["generated_text"])
            self.assertIn("output limit", json.loads((output / "run.json").read_text())["error"])
            self.assertTrue((output / "report.html").is_file())
            self.assertFalse((output / "note.txt").exists())

    def test_cli_checks_note_model_before_loading_speech(self):
        with tempfile.TemporaryDirectory() as directory, note_server(models=[]) as (url, _):
            audio = Path(directory) / "input.wav"
            with wave.open(str(audio), "wb") as wav:
                wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                wav.writeframes(b"\0\0" * 32000)
            argv = ["scribnic", str(audio), "--soap", "--base-url", url,
                    "--output-dir", str(Path(directory) / "run")]
            with patch.object(sys, "argv", argv), patch("scribnic.cli.build_pipeline") as build, \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    main()
                build.assert_not_called()


if __name__ == "__main__":
    unittest.main()
