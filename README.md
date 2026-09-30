# scribnic

Pipeline local para transcribir un audio, separar hablantes y, opcionalmente, generar texto con LM Studio. Los modelos por defecto son [Nemotron 3 Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization) y [Nemotron 3.5 ASR Streaming](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b). El idioma por defecto es `es-ES`.

## Uso

Requiere Python 3.12, Linux x86-64 y el entorno ROCm indicado en `pyproject.toml`. Desde la raíz:

```bash
uv sync
uv run scribnic entrada.wav
uv run scribnic entrada.wav --llm-model ID_EN_LM_STUDIO
uv run scribnic entrada.wav --device cpu
uv run scribnic entrada.wav --device cuda:0
uv run scribnic entrada.wav --model moss
```

El comando imprime JSON con `transcription`, `diarization`, `utterances` y `generated_text`. Sin `--llm-model`, `generated_text` es `null`. Para probar el flujo sin descargar pesos ni llamar a LM Studio:

```bash
uv run scribnic entrada.wav --demo
```

El audio de entrada debe ser WAV PCM16, mono, 16 kHz. Por ejemplo:

```bash
ffmpeg -i entrada.mp3 -ac 1 -ar 16000 -c:a pcm_s16le entrada.wav
uv run scribnic entrada.wav
```

Para usar los modelos anteriores:

```bash
uv run scribnic entrada.wav --asr canary --diarizer sortformer
```

`--model moss` usa MOSS-Transcribe-Diarize para transcripción y diarización en una sola pasada; no ejecuta `--asr` ni `--diarizer`. Mantiene las marcas temporales y etiquetas de hablante que devuelve el modelo. MOSS carga código remoto de su repositorio de Hugging Face (`trust_remote_code=True`); úsalo solo si confías en ese checkpoint.

Qwen3-ASR-0.6B-hf transcribe, pero no devuelve etiquetas de hablante ni tiempos por sí solo. Se puede combinar con un diarizador:

```bash
uv run scribnic entrada.wav --asr qwen --diarizer nemotron
```

El adaptador de Qwen procesa regiones de hasta 30 segundos y les asigna esos tiempos aproximados. La calidad de la atribución de hablantes depende del diarizador elegido.

`--device auto` (por defecto) usa la GPU si PyTorch la detecta y CPU si no. `--device cpu` y `--device cuda:0` eligen el dispositivo para **ambos modelos**; PyTorch llama `cuda:0` a la GPU AMD con ROCm. La inferencia en CPU puede ser lenta. Los pesos se descargan en la primera ejecución real. El modelo nuevo de diarización necesita la versión de Transformers desde Git fijada en `pyproject.toml`, porque la versión publicada anteriormente no reconoce su arquitectura. NeMo, usado por los modelos antiguos, puede necesitar espacio en disco al descomprimirlos; el programa usa `~/.cache/scribnic/tmp` salvo que `TMPDIR` esté definido.

## Estructura

| Ruta | Contenido |
| --- | --- |
| `src/scribnic/models.py` | Tipos de datos compartidos, sin dependencias de ML. |
| `src/scribnic/core.py` | Contratos, atribución de hablantes y pipeline. |
| `src/scribnic/backends/` | Implementaciones de ASR, diarización, generación y demo. |
| `src/scribnic/factory.py` | Selección de modelos por defecto u opciones del CLI. |
| `src/scribnic/cli.py` | Entrada por línea de comandos. |
| `scripts/` | Experimento anterior de transcripción multitalker. |
| `data/examples/` | Grabaciones y salidas locales, excluidas de Git. |
| `pyproject.toml`, `uv.lock` | Dependencias y versiones bloqueadas. |

El mismo `build_pipeline()` sirve para un servicio o para evaluar etapas por separado:

```python
from pathlib import Path
from scribnic import FileSource, build_pipeline

pipeline = build_pipeline(device="auto")  # también "cpu" o "cuda:0"
result = pipeline.run(FileSource(Path("entrada.wav")))
```

Los tiempos son segundos desde el inicio del archivo. Nemotron 3.5 ASR devuelve texto sin marcas de palabra en la API documentada, así que el adaptador transcribe regiones de diarización y les asigna tiempos aproximados de región. Divide turnos largos cada 30 segundos. Una región con voces simultáneas conserva varios `candidates` y deja `speaker` en `null`. Si el diarizador no detecta hablantes, se transcribe el archivo completo sin atribución. Canary conserva las marcas de palabra del flujo anterior.

Para comprobar los contratos del pipeline sin descargar modelos:

```bash
uv run python -m unittest discover -s tests -v
```

## Experimento multitalker anterior

```bash
uv run python scripts/build_conversation.py entrada.wav --output salida.json
```

Este experimento usa los modelos Sortformer v2.1 y Multitalker Parakeet y conserva sus ajustes ROCm. La transcripción de ejemplo en español que se generó localmente no era fiable; el modelo Multitalker Parakeet se evaluó principalmente en inglés.
