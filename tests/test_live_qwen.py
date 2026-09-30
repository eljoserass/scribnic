"""Opt-in Qwen ASR integration check against the local reference sample."""

import json
import os
import tempfile
import unittest
import wave
from pathlib import Path

from scribnic.backends.asr import QwenTranscriber
from scribnic.models import Audio


SAMPLE_DIR = Path(__file__).resolve().parents[1] / "data" / "examples"


@unittest.skipUnless(os.getenv("SCRIBNIC_LIVE_TESTS") == "1", "set SCRIBNIC_LIVE_TESTS=1")
class LiveQwenTests(unittest.TestCase):
    def test_first_reference_turn(self):
        import torch

        audio_path = SAMPLE_DIR / "dialogue8.wav"
        reference_path = SAMPLE_DIR / "original.json"
        if not audio_path.exists() or not reference_path.exists():
            self.skipTest("local audio and reference transcript are required")
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        self.assertIn("Buenos días", reference["turns"][0]["text"])

        with tempfile.TemporaryDirectory() as directory:
            excerpt = Path(directory) / "excerpt.wav"
            with wave.open(str(audio_path), "rb") as source, wave.open(str(excerpt), "wb") as target:
                target.setparams(source.getparams())
                target.writeframes(source.readframes(12 * source.getframerate()))
            audio = Audio(excerpt, 16000, 12)
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            transcription = QwenTranscriber(device, "es-ES").transcribe(audio, ())

        self.assertEqual(len(transcription), 1)
        self.assertIn("Buenos días", transcription[0].text)
