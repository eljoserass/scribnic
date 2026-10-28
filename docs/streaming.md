# Audio incremental

El paso de voz configura `mode = "streaming"`. El backend selecciona la estrategia
según el protocolo declarado; el runtime continúa siendo el mismo servidor residente.
`WorkflowSession` orquesta y guarda artefactos; no implementa ventanas ni WebSocket.

| Workflow | Procesamiento | Entrada del servicio |
| --- | --- | --- |
| `nvidia_cpp_streaming` | ASR cache-aware + diarización | PCM por WebSocket nativo |
| `nvidia_cpp_asr_streaming` | El mismo ASR sin diarizador | PCM por WebSocket nativo |
| `whisper_bonsai_streaming` | Ventanas con solape, encoder recalculado | WAV pequeños por HTTP |
| `lfm3_audio_note` | Encoder causal y prefill incremental; nota directa | PCM por WebSocket de llama-server |

Los workflows Whisper y NVIDIA producen un transcript final y después una sola nota. `--output transcript`
sirve con Whisper y NVIDIA ASR; con NVIDIA diarizado se usa `--output diarized_transcript`.
LFM3 produce directamente una nota, sin transcript ni actualizaciones de transcripción.
Su configuración y límites están en [LFM3](lfm3.md).

```bash
# Consumir el archivo tan rápido como permita la inferencia.
uv run scribnic run consulta.wav --workflow whisper_bonsai_streaming
uv run --extra streaming scribnic run consulta.wav --workflow nvidia_cpp_streaming

# Simular audio que llega durante una consulta, con el mismo pipeline.
uv run scribnic run consulta.wav --workflow whisper_bonsai_streaming --realtime
uv run --extra streaming scribnic run consulta.wav --workflow nvidia_cpp_streaming --realtime
```

## Fuentes y API Python

`run_stream` recibe un iterable de `bytes`: PCM16 little-endian, mono, 16 kHz,
con frames completos. La fuente puede reproducir un WAV, leer una cola de captura
o recibir audio desde otra aplicación. No se instala una biblioteca de micrófono.
El llamante posee el dispositivo y debe poder detener una fuente bloqueante.

```python
from pathlib import Path
from scribnic import open_workflow, wav_chunks

with open_workflow("whisper_bonsai_streaming") as session:
    result = session.run_stream(
        wav_chunks(Path("consulta.wav"), realtime=True),
        output_dir="runs/consulta-incremental",
    )
    print(result.note.text)
```

Para separar carga de captura, puede llamarse primero a
`session.start(output="transcript")` con Whisper, o
`session.start(output="diarized_transcript")` con NVIDIA diarizado. Así se precarga
solo el paso de voz antes de abrir el dispositivo; la nota puede cargar después.

`on_update` es un callback opcional con `TranscriptUpdate(text, audio_seconds, final)`.
Cada actualización sustituye el texto anterior; no se concatena. Whisper puede
revisar el solape. NVIDIA publica bloques finalizados y conserva los deltas nativos
en el log, porque su contrato no identifica inequívocamente todas las revisiones.
`final=True` indica que terminó la entrada, no que se está imprimiendo un token.

`cancel` acepta un `threading.Event`, comprobado entre bloques de entrada. No fuerza
el desbloqueo de un dispositivo externo: eso pertenece a la implementación de captura.
La lectura tiene backpressure del consumidor y no descarta audio silenciosamente.
Una captura con callback de hardware debe gestionar su propia cola o grabación a
disco; no puede asumir que Python siempre consumirá al ritmo del dispositivo.

La ejecución guarda un WAV en `runtime/recording-*.wav`, lo incorpora como `AudioFile`
con duración/hash y reutiliza el almacenamiento, reportes, CPU/RSS y spans habituales.
También conserva el audio ya recibido si se interrumpe la operación. Sin `output_dir`
la grabación queda en la caché de diagnósticos. Los artefactos de consultas distintas
permanecen aislados aunque se reutilicen modelos.

## Ventanas de Whisper

El preset configura `window_seconds = 10`, `overlap_seconds = 2`: el avance es 8 s.
Se pueden sobrescribir en `[steps.speech.model_settings]` de un workflow TOML.
El cliente solicita tiempos por palabra, ajusta los tiempos al audio original y
reconcilia el solape por texto; si no encuentra un anclaje compatible usa su punto
medio temporal y recorta los intervalos que lo cruzan, conservando el orden de las
palabras. La política queda registrada en el resultado y los tiempos nativos
originales permanecen en JSONL. Siempre procesa la última ventana con audio nuevo.

Esta estrategia necesita evaluar errores en bordes y repeticiones. No es una
garantía de texto idéntico a la inferencia offline ni de menor tiempo de cómputo.
El modelo se mantiene cargado, pero el encoder vuelve a procesar cada ventana.
El ejemplo oficial `whisper-stream` también usa reconocimiento por ventanas; su
captura SDL no es necesaria para esta fuente PCM ni convierte al servidor HTTP en
OpenAI Realtime. No se añade otro ejecutable ni un fork del servidor.

## NVIDIA y contratos de red

NeMo recibe fragmentos de 200 ms y conserva la caché acústica y el diarizador durante
la conexión. El tamaño de esos fragmentos no es el contexto interno del encoder.
El cliente envía un único `commit` al finalizar: en v0.2.0 ese evento termina y
reinicia el estado del reconocimiento y sus hablantes. El endpointing interno puede
producir finales intermedios sin abrir otra consulta.

El protocolo nativo difiere de OpenAI Realtime en sesión, eventos, IDs y diarización.
Compartimos la API Python y la fuente PCM, conservando adaptadores de protocolo.
No se implementa un gateway ni un servidor OpenAI Realtime.

## Memoria y medición

La validación WAV usa bloques y los eventos nativos se escriben a JSONL, sin guardar
su historial completo en una lista. Whisper retiene una ventana de PCM; la salida
textual y el estado interno del diarizador sí pueden crecer con la consulta.
Los pesos y buffers de cómputo permanecen residentes. No se promete RAM total constante.
El contexto del generador de notas tampoco aumenta por transcribir en streaming:
para grabaciones largas puede pedirse solo el transcript y evaluar la nota aparte.

El tiempo de request incluye la espera de la fuente. Un replay `--realtime` y uno
sin pausas miden cosas distintas; deben etiquetarse por separado al comparar RTF.
La CLI conserva esos flags en `telemetry/invocation.json`; una aplicación Python
debe registrar cómo temporiza su fuente.
`--profile` conserva CPU/RSS y spans para C++; declara que no hay traza Torch de
sus operadores. Los JSONL de ventanas/eventos y las respuestas finales quedan en
`runtime/`. Los benchmarks de latencia deben hacerse sin captura detallada.

Comprobado en WSL/Linux CPU con el mismo WAV de 24 s para NVIDIA diarizado y Whisper,
ambos seguidos de Bonsai. Esto prueba ejecución y generación de artefactos; calidad,
captura física y estabilidad durante horas necesitan validación propia.
