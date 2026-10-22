"""Native server startup, reuse, provenance and cleanup without real models."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scribnic.config import load_workflow
from scribnic.downloads import PreparedStep
from scribnic.runtimes import LlamaServer, WhisperServer


class FakeProcess:
    def __init__(self, *args, **kwargs):
        self.pid = 12345
        self.returncode = None
        self.terminated = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def wait(self, timeout):
        return self.returncode


def prepared(root: Path) -> PreparedStep:
    executable = root / "server"
    executable.write_bytes(b"fake server")
    return PreparedStep(root, root / "model", executable, root / "vad.bin")


class NativeServerTests(unittest.TestCase):
    def test_whisper_server_reuses_a_ready_model_and_binds_loopback(self):
        runtime = load_workflow("whisper_bonsai").steps[0].runtime
        with tempfile.TemporaryDirectory() as directory:
            server = WhisperServer(runtime, prepared(Path(directory)))
            with patch(
                "scribnic.runtimes.subprocess.Popen", side_effect=FakeProcess
            ) as launch:
                with patch("scribnic.runtimes._health", return_value=True):
                    first = server.start()
                    self.assertEqual(server.start(), first)
                process = server.process
                server.close()
            self.assertEqual(launch.call_count, 1)
            self.assertEqual(
                server.command[server.command.index("--host") + 1], "127.0.0.1"
            )
            self.assertIn("/v1/audio/transcriptions", server.command)
            self.assertIn("--vad-model", server.command)
            self.assertTrue(process.terminated)
            self.assertTrue(server._log is None)

    def test_interrupted_startup_terminates_its_process(self):
        runtime = load_workflow("whisper_bonsai").steps[0].runtime
        with tempfile.TemporaryDirectory() as directory:
            server = WhisperServer(runtime, prepared(Path(directory)))
            with patch("scribnic.runtimes.subprocess.Popen", side_effect=FakeProcess):
                with patch("scribnic.runtimes._health", side_effect=KeyboardInterrupt):
                    with self.assertRaises(KeyboardInterrupt):
                        server.start()
            self.assertTrue(server.process.terminated)
            self.assertIsNone(server._log)

    def test_prism_is_explicit_without_changing_the_llama_http_engine(self):
        runtime = load_workflow("bonsai_note").steps[0].runtime
        self.assertEqual(runtime.name, "llama_cpp_prism")
        self.assertEqual(runtime.engine, "llama_cpp")
        with tempfile.TemporaryDirectory() as directory:
            server = LlamaServer(runtime, prepared(Path(directory)), "bonsai")
            server.command = server._command(1234)
            metadata = server.metadata()
        self.assertEqual(metadata["distribution"], "prism")
        self.assertEqual(len(metadata["binary_sha256"]), 64)
        self.assertIn("--metrics", server.command)


if __name__ == "__main__":
    unittest.main()
