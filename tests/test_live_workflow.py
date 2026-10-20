"""Opt-in real MOSS -> Bonsai check; ordinary tests need no weights or Torch."""

import os
import tempfile
import unittest
import wave
from pathlib import Path

from scribnic import AudioFile, open_workflow


@unittest.skipUnless(
    os.getenv("SCRIBNIC_LIVE_TESTS") == "1", "set SCRIBNIC_LIVE_TESTS=1"
)
class LiveWorkflowTests(unittest.TestCase):
    def test_moss_bonsai_and_speaker_removal(self):
        sample = os.getenv("SCRIBNIC_TEST_AUDIO")
        if not sample:
            self.skipTest("set SCRIBNIC_TEST_AUDIO to a local mono 16 kHz WAV")
        with tempfile.TemporaryDirectory() as directory:
            excerpt = Path(directory) / "excerpt.wav"
            with wave.open(sample, "rb") as source:
                with wave.open(str(excerpt), "wb") as target:
                    target.setparams(source.getparams())
                    target.writeframes(source.readframes(35 * source.getframerate()))
            with open_workflow("moss_bonsai", download=False) as session:
                result = session.run(AudioFile(excerpt))
                self.assertTrue(result.note.text.strip())
                self.assertTrue(result.transcript.segments)
                modified = session.run(result.transcript.without_speakers())
                self.assertEqual([record.name for record in modified.steps], ["note"])
                self.assertTrue(modified.note.text.strip())
