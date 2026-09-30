"""Opt-in integration check against the local multi-speaker sample."""

import json
import os
import tempfile
import unittest
import wave
from pathlib import Path

from scribnic import FileSource, build_pipeline


SAMPLE_DIR = Path(__file__).resolve().parents[1] / "data" / "examples"


@unittest.skipUnless(os.getenv("SCRIBNIC_LIVE_TESTS") == "1", "set SCRIBNIC_LIVE_TESTS=1")
class LiveMossTests(unittest.TestCase):
    def test_two_speaker_excerpt(self):
        audio_path = SAMPLE_DIR / "dialogue8.wav"
        reference_path = SAMPLE_DIR / "original.json"
        if not audio_path.exists() or not reference_path.exists():
            self.skipTest("local audio and reference transcript are required")
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(reference["turns"]), 2)
        self.assertNotEqual(reference["turns"][0]["speaker"], reference["turns"][1]["speaker"])

        with tempfile.TemporaryDirectory() as directory:
            excerpt = Path(directory) / "excerpt.wav"
            with wave.open(str(audio_path), "rb") as source, wave.open(str(excerpt), "wb") as target:
                target.setparams(source.getparams())
                target.writeframes(source.readframes(35 * source.getframerate()))
            result = build_pipeline(model="moss", skip_generation=True).run(FileSource(excerpt))

        self.assertGreater(len(result.transcription), 0)
        self.assertEqual(len(result.transcription), len(result.utterances))
        self.assertGreaterEqual(len({turn.speaker for turn in result.utterances}), 2)
        self.assertTrue(all(turn.text and turn.start < turn.end for turn in result.utterances))
        self.assertIn("Buenos días", " ".join(turn.text for turn in result.utterances))
