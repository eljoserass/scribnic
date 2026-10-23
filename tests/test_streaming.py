"""NeMo-specific stream finalization and per-recording state with a tiny socket."""

import importlib.util
import json
import queue
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_whisper_cpp import write_audio

from scribnic.backends.nemo_streaming import transcribe_stream


class FakeSocket:
    """Simulate one isolated recognition connection with provisional and final text."""

    def __init__(self) -> None:
        """Allocate a separate event queue and PCM byte count for this recording."""
        self.events = queue.Queue()
        self.events.put({"type": "session.created"})
        self.pcm_bytes = 0
        self.session = None
        self.closed = False

    def __enter__(self) -> "FakeSocket":
        """Provide the connection owned by one transcription request."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Release the fake connection at the end of the request."""
        self.close()

    def send(self, message: bytes | str) -> None:
        """Respond to configuration and commit while retaining bounded PCM chunks."""
        if isinstance(message, bytes):
            if len(message) > 6400:
                raise AssertionError("Audio upload was not chunked.")
            self.pcm_bytes += len(message)
            return
        event = json.loads(message)
        if event["type"] == "session.update":
            self.session = event["session"]
            self.events.put({"type": "session.updated"})
        elif event["type"] == "input_audio_buffer.commit":
            self.events.put(
                {
                    "type": "conversation.item.input_audio_transcription.delta",
                    "delta": "provisional",
                }
            )
            self.events.put(
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "transcript": "Hola",
                    "words": [{"word": "Hola", "start": 0, "end": 0.5, "speaker": 1}],
                }
            )
            self.events.put({"type": "input_audio_buffer.committed"})

    def recv(self, *, timeout: float) -> str:
        """Return ordered protocol events with a finite receive deadline."""
        return json.dumps(self.events.get(timeout=timeout))

    def close(self) -> None:
        """Mark this connection closed without affecting any other recording."""
        self.closed = True


@unittest.skipUnless(
    importlib.util.find_spec("websockets"), "streaming extra not installed"
)
class StreamingTests(unittest.TestCase):
    """Verify actual protocol ordering without an inference service or model."""

    def test_replay_finalizes_each_recording_and_uses_only_final_text(self) -> None:
        """Reuse of weights does not imply reuse of sockets or hypotheses."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_audio(root)
            sockets = [FakeSocket(), FakeSocket()]
            with patch(
                "websockets.sync.client.connect", side_effect=sockets
            ) as connect:
                for index in range(2):
                    result = transcribe_stream(
                        source.path,
                        base_url="http://127.0.0.1:1234/v1/",
                        language="es-ES",
                        diarized=True,
                        timeout_seconds=2,
                        event_path=root / f"events-{index}.json",
                        api_key="local",
                    )
                    self.assertEqual(result["text"], "Hola")
                    self.assertEqual(result["words"][0]["speaker"], 1)
                    self.assertTrue((root / f"events-{index}.json").is_file())
            self.assertEqual(connect.call_count, 2)
            self.assertEqual(
                connect.call_args.args[0],
                "ws://127.0.0.1:1234/v1/audio/transcriptions/realtime",
            )
            for socket in sockets:
                self.assertEqual(socket.pcm_bytes, 64000)
                self.assertTrue(socket.session["speaker_diarization"])
                self.assertTrue(socket.closed)
