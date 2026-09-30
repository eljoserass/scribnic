"""Fast checks for stage composition and model output conversion."""

import unittest
import tempfile
import wave
from pathlib import Path

from scribnic.backends.asr import audio_chunks
from scribnic.core import Pipeline
from scribnic.models import Audio, SpeakerSpan, TextSpan, Utterance
from scribnic.backends.joint import VibeVoiceRecognizer, parse_moss_transcript, parse_vibevoice_segments


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
    def test_audio_chunks_follow_speaker_regions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audio.wav"
            with wave.open(str(path), "wb") as wav:
                wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                wav.writeframes(b"\0\0" * 5 * 16000)
            audio = Audio(path, 16000, 5)
            speakers = (SpeakerSpan(1, 2, "speaker_0"), SpeakerSpan(3, 5, "speaker_1"))
            chunks = list(audio_chunks(audio, speakers, max_seconds=1))

        self.assertEqual([(start, end) for start, end, _ in chunks],
                         [(1, 2), (3, 4), (4, 5)])
        self.assertTrue(all(len(samples) == 16000 for _, _, samples in chunks))

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

    def test_vibevoice_parser(self):
        segments = [
            {"Start": 0.0, "End": 0.5, "Content": "[Silence]"},
            {"Start": 0.5, "End": 1.5, "Speaker": 0, "Content": "Hola"},
            {"Start": 1.5, "End": 2.5, "Speaker": 1, "Content": "Buenos días"},
        ]
        turns = parse_vibevoice_segments(segments)
        self.assertEqual([turn.speaker for turn in turns], ["speaker_0", "speaker_1"])
        self.assertEqual(turns[0].start, 0.5)
        with self.assertRaises(ValueError):
            parse_vibevoice_segments("bad")

    def test_vibevoice_adapter_uses_parsed_segments(self):
        import torch

        class Inputs(dict):
            def to(self, device, dtype):
                return self

        class Processor:
            def apply_transcription_request(self, audio):
                self.audio = audio
                return Inputs(input_ids=torch.tensor([[1, 2]]))

            def decode(self, generated, return_format):
                self.generated = generated
                return [[{"Start": 0, "End": 1, "Speaker": 0, "Content": "Hola"}]]

        class Model:
            device = "cpu"
            dtype = torch.float32

            def generate(self, **inputs):
                return torch.tensor([[1, 2, 3]])

        recognizer = VibeVoiceRecognizer.__new__(VibeVoiceRecognizer)
        recognizer.torch = torch
        recognizer.processor = Processor()
        recognizer.model = Model()
        turns = recognizer.recognize(Audio(Path("sample.wav"), 16000, 2))
        self.assertEqual(turns[0].speaker, "speaker_0")
        self.assertEqual(recognizer.processor.audio, "sample.wav")


if __name__ == "__main__":
    unittest.main()
