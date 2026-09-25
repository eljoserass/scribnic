"""Run NVIDIA's streaming multitalker ASR example on a local WAV file."""

import argparse
import json
import os
import sys
import tarfile
import tempfile
import wave
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--output", type=Path, default=Path("transcript.json"))
    args = parser.parse_args()

    audio = args.audio.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    with wave.open(str(audio), "rb") as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 16000):
            parser.error("Audio must be a mono, 16-bit PCM WAV at 16 kHz")
        if wav.getnframes() == 0:
            parser.error("Audio file is empty")

    # NeMo extracts several GB of model files. WSL's /tmp is usually a small tmpfs.
    if "TMPDIR" not in os.environ:
        cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        model_tmpdir = cache_root / "scribnic" / "tmp"
        model_tmpdir.mkdir(parents=True, exist_ok=True)
        os.environ["TMPDIR"] = str(model_tmpdir)
        tempfile.tempdir = str(model_tmpdir)

    import torch

    if torch.version.hip is None or not torch.cuda.is_available():
        parser.error("ROCm PyTorch cannot see the AMD GPU; run 'uv sync' and retry with 'uv run'")
    # MIOpen convolution fails on this gfx1151 setup; PyTorch's GEMM fallback
    # still runs the convolutions on the GPU.
    torch.backends.cudnn.enabled = False

    from nemo.collections.asr.models import ASRModel, SortformerEncLabelModel
    from nemo.collections.asr.parts.utils.multispk_transcribe_utils import SpeakerTaggedASR, write_seglst_file
    from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
    from omegaconf import OmegaConf

    from multitalker_transcript_config import MultitalkerTranscriptionConfig

    device = torch.device("cuda:0")  # ROCm uses PyTorch's cuda device name.
    print(f"Using {torch.cuda.get_device_name(device)} ({torch.__version__})", flush=True)
    diar_model = SortformerEncLabelModel.from_pretrained(
        "nvidia/diar_streaming_sortformer_4spk-v2.1", map_location=device
    ).eval()
    # NeMo enables NVIDIA CUDA graphs while constructing this RNNT decoder.
    # Disable them in the checkpoint config before construction on ROCm.
    asr_archive = ASRModel.from_pretrained(
        "nvidia/multitalker-parakeet-streaming-0.6b-v1", return_model_file=True
    )
    with tarfile.open(asr_archive, "r:*") as archive:
        config_file = archive.extractfile("model_config.yaml")
        if config_file is None:
            raise ValueError("The multitalker model archive has no model_config.yaml")
        asr_config = OmegaConf.load(config_file)
    asr_config.decoding.greedy.use_cuda_graph_decoder = False
    asr_model = ASRModel.restore_from(
        str(asr_archive), override_config_path=asr_config, map_location=device
    ).eval()
    # NeMo's fused Triton subsampling autotunes many kernels on ROCm before
    # processing the first chunk. The regular PyTorch GPU path avoids that cost.
    diar_model.encoder.pre_encode.conv.fuse_triton = False
    asr_model.encoder.pre_encode.conv.fuse_triton = False

    cfg = OmegaConf.structured(MultitalkerTranscriptionConfig())
    cfg.audio_file = str(audio)
    cfg.output_path = str(output)
    diar_model = MultitalkerTranscriptionConfig.init_diar_model(cfg, diar_model)

    samples = [{"audio_filepath": cfg.audio_file}]
    streaming_buffer = CacheAwareStreamingAudioBuffer(
        model=asr_model,
        online_normalization=cfg.online_normalization,
        pad_and_drop_preencoded=cfg.pad_and_drop_preencoded,
    )
    streaming_buffer.append_audio_file(audio_filepath=cfg.audio_file, stream_id=-1)
    streamer = SpeakerTaggedASR(cfg, asr_model, diar_model)
    print("Models loaded; streaming audio...", flush=True)

    for step_num, (chunk_audio, chunk_lengths) in enumerate(streaming_buffer):
        drop_extra_pre_encoded = (
            0 if step_num == 0 and not cfg.pad_and_drop_preencoded
            else asr_model.encoder.streaming_cfg.drop_extra_pre_encoded
        )
        with torch.inference_mode(), torch.amp.autocast(device.type, enabled=cfg.use_amp):
            streamer.perform_parallel_streaming_stt_spk(
                step_num=step_num,
                chunk_audio=chunk_audio,
                chunk_lengths=chunk_lengths,
                is_buffer_empty=streaming_buffer.is_buffer_empty(),
                drop_extra_pre_encoded=drop_extra_pre_encoded,
            )
        if (step_num + 1) % 5 == 0:
            print(f"Processed {step_num + 1} chunks", flush=True)

    segments = streamer.generate_seglst_dicts_from_parallel_streaming(samples=samples)
    output.parent.mkdir(parents=True, exist_ok=True)
    if segments:
        write_seglst_file(segments, str(output))
    else:
        output.write_text(json.dumps(segments) + "\n", encoding="utf-8")
        print("No speech segments were transcribed from this audio.", file=sys.stderr)
    print(f"Wrote {len(segments)} segments to {output}")


if __name__ == "__main__":
    main()
