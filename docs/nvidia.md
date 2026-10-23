# NVIDIA: servidor residente y contratos

Scribnic usa NeMo-Speech.cpp oficial v0.2.0 con dos archivos fijados por revisión
Hugging Face y SHA256: `nemotron-3.5-asr-streaming-0.6b.q8_0.gguf` y
`Nemotron-3-Diarization.q8_0.gguf`. El ASR es multilingüe y selecciona `es-ES`.
El diarizador es Sortformer V3, de ocho hablantes. Q8_0 es el formato de pesos;
no significa que todas las operaciones o todos los tensores sean enteros de 8 bits.

## Preparación y ciclo de vida

```bash
uv run scribnic setup --workflow nvidia_cpp_offline
uv run scribnic run consulta.wav --workflow nvidia_cpp_offline
uv run --extra streaming scribnic run consulta.wav --workflow nvidia_cpp_streaming
```

`setup` descarga solo los assets configurados y verifica hashes. No requiere el
extra antiguo `[nvidia]`, NeMo/PyTorch ni una ruta manual al ejecutable. El runtime
arranca `nemo-speech serve` en loopback, carga ASR y diarizador una vez y espera
`/ready`. El gestor reside en `runtimes/nemo_cpp.py`; el cliente y las conversiones
están en `backends/openai_transcription.py` y `backends/transcription_formats.py`.

Una sesión mantiene los modelos entre requests. Cada archivo tiene su propio
estado de reconocimiento y etiquetas. Los procesos se cierran al salir, incluso
si se interrumpe el arranque. Para evaluar: llamar a `session.start()` antes de
los `session.run(...)`, guardar sus medidas de carga aparte y omitir `--profile`
en las mediciones de latencia.

## Dos modos de entrada

- `nvidia_cpp_offline` sube el WAV a `/v1/audio/transcriptions`, pide `verbose_json`
  y activa `diarization=true`. Conserva las palabras con tiempos y hablantes.
- `nvidia_cpp_streaming` abre una conexión nueva al protocolo específico
  `/v1/audio/transcriptions/realtime`, envía PCM en chunks de 200 ms y finaliza con
  `input_audio_buffer.commit`. Recibe a la vez para evitar bloqueo por respuestas
  parciales. No introduce esperas de reproducción. No es OpenAI Realtime ni una
  API pública de captura de micrófono: por ahora la entrada sigue siendo un WAV.

La salida pública es `DiarizedTranscript`. El texto puede proyectarse mediante
`without_speakers()` para ablations; la nota acepta ambos tipos. El cliente nunca
inventa roles clínicos a partir del número de hablante.

## Variantes sin diarización

`nvidia_cpp_asr_offline` y `nvidia_cpp_asr_streaming` usan el mismo ASR multilingüe,
runtime y generador de notas, sin declarar ni preparar el diarizador. El servidor
arranca solo con ASR; la petición HTTP usa `diarization=false` y la sesión WebSocket
usa `speaker_diarization=false`. Producen `Transcript` con tiempos y sin hablantes;
la nota se genera a partir de ese texto. Los recursos ya presentes se reutilizan.

```bash
uv run scribnic run consulta.wav --workflow nvidia_cpp_asr_offline
uv run --extra streaming scribnic run consulta.wav --workflow nvidia_cpp_asr_streaming

# Obtener solo la transcripción, sin generar nota.
uv run scribnic run consulta.wav --workflow nvidia_cpp_asr_offline --output transcript
```

Desactivar diarización evita ejecutar ese modelo. Para aislar solo el efecto de las
etiquetas sobre una nota, utilizar en cambio la misma transcripción diarizada y su
proyección `without_speakers()` con el mismo generador.

## Estrategia de diarización

El runtime C++ reconoce palabras y les atribuye un hablante mediante la actividad
temporal del diarizador. Se registra `speaker_attribution_strategy=word_time_alignment`.
La antigua referencia NeMo/Python streaming usaba `masked_asr=true`: características
acústicas enmascaradas por actividad y un estado ASR por hablante. No es una
separación de fuentes; durante solapes puede conservar la misma mezcla de voces.
La referencia Python se retiró del núcleo y sus dependencias, no se presenta como
una implementación equivalente al C++. Su código permanece en la historia de Git.

## Profiling

Se guardan carga, request, RTF, configuración, hashes, PID, respuesta nativa y
logs. Python y el servidor C++ se muestrean por separado. En streaming se conserva
también el flujo de eventos recibido. `--profile` declara explícitamente que no
hay captura de operadores C++ mediante Torch; los recursos y spans siguen activos.
Los servidores externos necesitan su propia instrumentación para medir RAM/CPU.

La memoria en la caché de modelos, el proceso Python y el proceso nativo son
conceptos distintos. Mantener el servidor cargado ahorra recargas, pero mantiene
sus pesos residentes. No sumar picos RSS individuales como si fueran simultáneos.

## Plataformas y fuentes

Validado en WSL/Linux x86_64 CPU. Los presets fijan también los archivos oficiales
para Linux arm64, Windows x86_64 CPU y macOS arm64 Metal; esas plataformas no se
han validado localmente. La calidad de transcripción/diarización queda pendiente.

- [Servidor oficial v0.2.0](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/v0.2.0/docs/server.md)
- [API y protocolo WebSocket](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/v0.2.0/docs/api.md)
- [Modelos y cuantización](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/v0.2.0/docs/asr/models.md)
- [Referencia de ASR por hablante](https://huggingface.co/nvidia/Nemotron-3-Diarization/blob/main/ASR_INTEGRATION_GUIDE.md)
