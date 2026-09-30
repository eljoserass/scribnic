"""Fast checks for stage composition and model output conversion."""

import unittest
from pathlib import Path

from scribnic.core import Pipeline
from scribnic.models import Audio, SpeakerSpan, TextSpan, Utterance
from scribnic.backends.joint import parse_moss_transcript


class Source:
    def load(self):
        return Audio(Path("unused.wav"), 16000, 5.0)


class SeparateDiarizer:
    def diarize(self, audio):
        return (SpeakerSpan(0, 3, "speaker_0"), SpeakerSpan(2, 5, "speaker_1"))


class SeparateTranscriber:
    def transcribe(self, audio, speakers):
        return (TextSpan(0, 2, "hello"), TextSpan(2, 3, "overlap"),
                TextSpan(3, 5, "bye"))


class JointRecognizer:
    def recognize(self, audio):
        return (Utterance(0, 2, "hello", "speaker_0", ("speaker_0",)),
                Utterance(2, 5, "bye", "speaker_1", ("speaker_1",)))


class PipelineTests(unittest.TestCase):
    def test_separate_stages_and_overlap(self):
        result = Pipeline(SeparateTranscriber(), SeparateDiarizer()).run(Source())
        self.assertEqual([turn.speaker for turn in result.utterances],
                         ["speaker_0", None, "speaker_1"])
        self.assertEqual(result.utterances[1].candidates, ("speaker_0", "speaker_1"))

    def test_joint_preserves_model_speaker(self):
        result = Pipeline(recognizer=JointRecognizer()).run(Source())
        self.assertEqual(result.transcription[0].text, "hello")
        self.assertEqual(result.diarization[1], SpeakerSpan(2, 5, "speaker_1"))
        self.assertEqual(result.utterances[1].speaker, "speaker_1")

    def test_joint_rejects_out_of_bounds(self):
        class Invalid:
            def recognize(self, audio):
                return (Utterance(4, 6, "bad", "speaker_0", ("speaker_0",)),)

        with self.assertRaises(ValueError):
            Pipeline(recognizer=Invalid()).run(Source())

    def test_moss_parser(self):
        output = "[0.48][S01]Hola[1.66][2.26][S02]Buenos días[3.81]"
        turns = parse_moss_transcript(output)
        self.assertEqual(turns[0], Utterance(0.48, 1.66, "Hola", "speaker_0", ("speaker_0",)))
        self.assertEqual(turns[1].speaker, "speaker_1")
        with self.assertRaises(ValueError):
            parse_moss_transcript("not a transcript")


if __name__ == "__main__":
    unittest.main()
