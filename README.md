# scribnic

Pipeline local para transcribir un audio, separar hablantes y, opcionalmente, generar texto con LM Studio. Los modelos por defecto son [Nemotron 3 Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization) y [Nemotron 3.5 ASR Streaming](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b). El idioma por defecto es `es-ES`.

## Prueba MOSS → Bonsai → SOAP en español

En LM Studio, carga **Ternary Bonsai 2 2B** (`prism-ml/Ternary-Bonsai-2-2B-gguf`, Q2_0), configura al menos 8192 tokens de contexto, desactiva *Thinking* y arranca el servidor. El adaptador pide `reasoning_effort=none` y `enable_thinking=false`; desactívalo en la aplicación si tu versión del servidor no respeta esos parámetros. [La ficha del modelo](https://huggingface.co/prism-ml/Ternary-Bonsai-2-2B-gguf) documenta formatos y ejecución.

Desde WSL, comprueba primero la conexión. Si LM Studio está en Windows y `localhost` no funciona, utiliza la IP de Windows; la puerta de enlace de `ip route show default` puede servir en una instalación WSL con NAT:

```bash
export LM_STUDIO_BASE_URL=http://IP_DE_WINDOWS:1234/v1
curl "$LM_STUDIO_BASE_URL/models"
```

Si la conexión requiere acceso desde WSL, activa **Serve on Local Network** en LM Studio. Si el servidor exige autenticación, exporta `LM_STUDIO_API_KEY`. `--base-url` permite sustituir la variable de entorno en cada ejecución.

Con el servidor accesible:

```bash
uv run scribnic data/examples/dialogue8.wav --soap
```

En la configuración local comprobada, el servidor de Windows responde desde WSL en `http://localhost:1235/v1`:

```bash
uv run scribnic data/examples/dialogue8.wav --soap --base-url http://localhost:1235/v1
```

`--soap` selecciona MOSS para transcripción y diarización, busca **solo** Ternary Bonsai 2 2B entre los modelos del servidor y solicita una nota breve con los apartados Subjetivo, Objetivo, Análisis y Plan, en español y texto libre. Las etiquetas de hablante siguen siendo anónimas. Si hay varias variantes de Bonsai 2 2B, selecciona el identificador exacto devuelto por `/models`:

```bash
uv run scribnic data/examples/dialogue8.wav --soap \
  --llm-model prism-ml/ternary-bonsai-2-2b
```

La terminal muestra el inicio y tiempo de cada etapa, la transcripción con hablantes y la nota. Al terminar, abre el `report.html` de la ruta impresa: incluye nota, transcripción, reproductor del audio original, tiempos e instrucción del generador. No necesita un servidor web ni recursos externos. Los archivos quedan en un directorio nuevo de `runs/`:

| Archivo | Contenido |
| --- | --- |
| `transcript.json` | Resultado de voz, guardado **antes** de llamar al generador. |
| `note.txt` | Texto libre de la nota, si termina la generación. |
| `result.json` | Resultado completo, si termina el pipeline. |
| `run.json` | Modelos, dispositivo, límites, instrucción, tiempos y estado/error. |
| `report.html` | Informe visual local; enlaza al audio original. |

Puedes elegir un directorio vacío con `--output-dir runs/mi-prueba`, obtener JSON en stdout con `--format json`, o cambiar la instrucción con `--instruction '...'`. `--soap --skip-generation` prueba solo MOSS. `--soap --demo` muestra el recorrido y el informe con **salidas inventadas**, sin ejecutar ningún modelo.

Esta primera prueba envía la conversación completa en una sola petición: límite de 12000 caracteres de diálogo y 768 tokens de salida. Los caracteres son un límite preventivo, no una medición del contexto en tokens. Una entrada demasiado larga se rechaza sin truncarla; ajusta `--llm-max-input-chars` y `--llm-max-tokens` según el contexto cargado en LM Studio. Si el servidor falla o corta la nota al alcanzar su límite, el comando falla y conserva la transcripción y el informe. Todavía no hay resumen por fragmentos para conversaciones largas.

## Estado de la arquitectura

Es un prototipo local ejecutable: usa modelos reales, valida el audio y los intervalos, y tiene pruebas de contratos y pruebas reales opcionales. No es todavía un servicio de producción ni garantiza la exactitud de una nota. `ARCHITECTURE_EVALUATION.md` contiene una propuesta de evolución, no una migración implementada.

```text
WAV → MOSS (transcripción + diarización) → turnos con hablante → LM Studio → nota
    o diarizador → ASR → atribución de hablantes ────────────┘
```

`Pipeline` acepta un reconocedor conjunto **o** un par diarizador/transcriptor, y un generador opcional. En modelos conjuntos, los campos `transcription` y `diarization` se derivan de sus turnos. El código se reutiliza desde Python; actualmente no hay API HTTP, cola de trabajos, entrada de audio en streaming ni aplicación con interfaz gráfica. Las dependencias están fijadas para Linux x86-64, Python 3.12 y esta configuración ROCm.

Validación local del 19 de octubre de 2026: la consulta de `data/examples/dialogue8.wav` (297,9 s) completó MOSS → Ternary Bonsai 2 2B Q2_0 en unos 122 s, con 81 segmentos y dos etiquetas de hablante; la nota tardó 2,4 s. Se observaron errores de palabras y cambios incorrectos de hablante en MOSS. Bonsai produjo los cuatro apartados, pero omitió detalles del plan y añadió un «No consta» sobrante. Esta prueba confirma la ejecución, no la fiabilidad clínica. El resultado local está en `runs/soap-bonsai-moss/` (excluido de Git).

## Uso

Requiere Python 3.12, Linux x86-64 y el entorno ROCm indicado en `pyproject.toml`. Desde la raíz:

```bash
uv sync
uv run scribnic entrada.wav
uv run scribnic entrada.wav --llm-model ID_EN_LM_STUDIO
uv run scribnic entrada.wav --device cpu
uv run scribnic entrada.wav --device cuda:0
uv run scribnic entrada.wav --model moss
uv run scribnic entrada.wav --model vibevoice
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

`--model vibevoice` utiliza `microsoft/VibeVoice-ASR-HF`, la versión **no streaming** compatible con la versión de Transformers fijada en este proyecto. A diferencia del checkpoint `VibeVoice-ASR-Streaming-7B` del experimento local, devuelve transcripción, hablante y tiempos de cada segmento. Es un modelo grande y requiere mucha memoria de GPU o RAM; su primera ejecución descarga los pesos. El checkpoint streaming requiere el paquete externo `vibevoice` con Transformers 4.x, incompatible con el Transformers 5.x que usan los otros modelos aquí.

Para limitar la concurrencia de las descargas, descarga primero el checkpoint con un solo archivo y una sola solicitud Xet a la vez; después ejecuta sin acceso al Hub:

```bash
HF_XET_NUM_CONCURRENT_RANGE_GETS=1 uv run hf download microsoft/VibeVoice-ASR-HF --max-workers 1
HF_HUB_OFFLINE=1 uv run scribnic entrada.wav --model vibevoice
```

Esto reduce la concurrencia, pero no garantiza un límite exacto de MB/s. En una GPU de 20 GB y 11 GB de RAM, VibeVoice necesitó mantener parte del modelo en CPU y la prueba de dos hablantes resultó demasiado lenta para completarla; el adaptador y la carga del modelo sí se verificaron, pero no su salida real en esa máquina.

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

Para ejecutar las pruebas reales con el audio local de `data/examples/`, define `SCRIBNIC_LIVE_TESTS=1`. VibeVoice se prueba por separado con `SCRIBNIC_LIVE_VIBEVOICE=1` porque descarga un checkpoint grande.

## Experimento multitalker anterior

```bash
uv run python scripts/build_conversation.py entrada.wav --output salida.json
```

Este experimento usa los modelos Sortformer v2.1 y Multitalker Parakeet y conserva sus ajustes ROCm. La transcripción de ejemplo en español que se generó localmente no era fiable; el modelo Multitalker Parakeet se evaluó principalmente en inglés.
