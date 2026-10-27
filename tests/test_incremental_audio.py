"""Incremental input boundaries and window reconciliation without model downloads."""

import io
import struct
import tempfile
import threading
import unittest
import wave
from collections.abc import Iterator
from pathlib import Path

from scribnic import Note, Transcript, open_workflow, wav_chunks
from scribnic.artifacts import Artifact, StepOutput, TextSegment
from scribnic.audio_stream import BYTES_PER_SECOND, PCMRecording
from scribnic.backends.whisper_windowed import (
    WindowSettings,
    _merge,
    transcribe_windows,
)
from scribnic.config import StepConfig, load_workflow


class WindowTests(unittest.TestCase):
    """Check processing before EOF, bounded requests and overlap/last-window behavior."""

    def test_boundary_jitter_preserves_word_order_with_explicit_time_ownership(
        self,
    ) -> None:
        """Neighboring words from different windows can straddle the same midpoint."""
        result = _merge(
            [TextSegment("valores", 8.83, 9.12)],
            [TextSegment("alterados", 8.79, 9.27)],
            start=8,
            previous_end=10,
        )
        self.assertEqual([segment.text for segment in result], ["valores", "alterados"])
        self.assertEqual(result[0].end, 9)
        self.assertEqual(result[1].start, 9)
        self.assertLessEqual(result[0].start, result[1].start)

    def test_processes_before_eof_and_preserves_overlap_and_final_words(self) -> None:
        """Every source word appears once, including the incomplete final window."""
        consumed = 0
        requests = []
        updates = []

        def chunks() -> Iterator[bytes]:
            """Encode eight seconds with each second carrying a distinct value."""
            nonlocal consumed
            for second in range(8):
                for _ in range(5):
                    consumed += 0.2
                    yield struct.pack("<h", second) * 3200

        def recognize(payload: bytes) -> dict:
            """Recover deterministic word times from the bounded window's PCM."""
            with wave.open(io.BytesIO(payload), "rb") as audio:
                frames = audio.readframes(audio.getnframes())
            seconds = len(frames) // BYTES_PER_SECOND
            start = struct.unpack("<h", frames[:2])[0]
            requests.append((start, seconds, consumed))
            return {
                "segments": [
                    {"text": f"w{start + index}", "start": index, "end": index + 0.5}
                    for index in range(seconds)
                ]
            }

        with tempfile.TemporaryDirectory() as directory:
            result = transcribe_windows(
                chunks(),
                request=recognize,
                settings=WindowSettings(window_seconds=3, overlap_seconds=1),
                event_path=Path(directory) / "windows.jsonl",
                on_update=updates.append,
            )
        self.assertLess(requests[0][2], 8)
        self.assertTrue(all(seconds <= 3 for _, seconds, _ in requests))
        self.assertEqual(requests[-1][:2], (6, 2))
        self.assertEqual(result["text"], " ".join(f"w{index}" for index in range(8)))
        self.assertEqual(result["segments"][-1]["start"], 7)
        self.assertEqual(updates[-1], result["text"])

    def test_exact_window_end_does_not_reprocess_overlap_only(self) -> None:
        """Finishing exactly at a request boundary does not create another request."""
        requests = []

        def recognize(payload: bytes) -> dict:
            """Record requests and return a valid silent window."""
            requests.append(payload)
            return {"segments": []}

        with tempfile.TemporaryDirectory() as directory:
            transcribe_windows(
                [b"\0\0" * 48000],
                request=recognize,
                settings=WindowSettings(window_seconds=3, overlap_seconds=1),
                event_path=Path(directory) / "events.jsonl",
            )
        self.assertEqual(len(requests), 1)

    def test_invalid_geometry_fails_without_consuming_audio(self) -> None:
        """Overlaps must permit forward progress and useful recognition windows."""
        for window, overlap in ((0, 0), (2, 2), (2, -1), (2, float("nan"))):
            with self.subTest(window=window, overlap=overlap):
                with self.assertRaises(ValueError):
                    WindowSettings(window_seconds=window, overlap_seconds=overlap)


class StreamWorkflowTests(unittest.TestCase):
    """Use the public workflow API with a source that is not a WAV file."""

    def test_stream_finalizes_recording_before_note_and_isolates_requests(self) -> None:
        """Both requests reuse backends while retaining independent durable audio."""
        constructed = []

        class Backend:
            """A tiny speech/note adapter recording its actual lifecycle."""

            def __init__(self, step: StepConfig) -> None:
                """Keep the configured operation without creating model resources."""
                self.name = step.name

            def start(self) -> float:
                """Represent an already available fake model."""
                return 0.0

            def run_stream(self, source: PCMRecording) -> StepOutput:
                """Consume PCM and return a transcript based on the received duration."""
                for _ in source:
                    pass
                source.notify("partial")
                source.notify(f"{source.duration}", final=True)
                return StepOutput(Transcript.from_text(f"{source.duration}"))

            def run(self, source: Artifact) -> StepOutput:
                """Generate a note only from the finalized transcript artifact."""
                return StepOutput(Note(source.text))

            def process_ids(self) -> tuple[int, ...]:
                """The fake owns no external process."""
                return ()

            def close(self) -> None:
                """The fake owns no resources to release."""

        def factory(step: StepConfig) -> Backend:
            """Record construction so model reuse can be asserted."""
            constructed.append(step.name)
            return Backend(step)

        with tempfile.TemporaryDirectory() as directory:
            updates = []
            config = load_workflow("whisper_bonsai_streaming")
            with open_workflow(config, backend_factory=factory) as session:
                first = session.run_stream(
                    [b"\0\0" * 16000],
                    output_dir=Path(directory) / "one",
                    on_update=updates.append,
                )
                second = session.run_stream(
                    [b"\0\0" * 32000],
                    output_dir=Path(directory) / "two",
                )
            self.assertEqual(first.note.text, "1.0")
            self.assertEqual(second.note.text, "2.0")
            self.assertNotEqual(first.source.path, second.source.path)
            self.assertEqual(sum(map(len, wav_chunks(first.source.path))), 32000)
            self.assertTrue(first.source.sha256)
            self.assertTrue((Path(directory) / "one" / "report.html").is_file())
            self.assertTrue(updates[-1].final)
        self.assertEqual(constructed, ["speech", "note"])

    def test_non_streaming_workflow_rejects_before_consuming_input(self) -> None:
        """Unsupported incremental calls are configuration errors, not model runs."""

        def chunks() -> Iterator[bytes]:
            """Fail if a rejected request touches the producer."""
            raise AssertionError("Consumed unsupported input")
            yield b""

        with open_workflow("whisper_bonsai", prepare=False) as session:
            with self.assertRaisesRegex(ValueError, "streaming transcription"):
                session.run_stream(chunks())

    def test_cancelled_recording_closes_partial_wav(self) -> None:
        """Cancellation preserves received PCM and rejects the incomplete operation."""
        cancelled = threading.Event()

        def chunks() -> Iterator[bytes]:
            """Cancel after the first complete input block."""
            yield b"\0\0" * 3200
            cancelled.set()
            yield b"\0\0" * 3200

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partial.wav"
            with PCMRecording(chunks(), path, cancel=cancelled) as source:
                with self.assertRaises(InterruptedError):
                    list(source)
            self.assertAlmostEqual(source.duration, 0.2)
            self.assertEqual(sum(map(len, wav_chunks(path))), 6400)


if __name__ == "__main__":
    unittest.main()
