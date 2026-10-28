# LFM3: audio directo a nota

`lfm3_audio_note` genera una nota SOAP en español directamente del audio, sin
crear un transcript intermedio. Usa `LiquidAI/LFM3-Audio-2B-GGUF` en `Q4_K_M`
(1,10 GB) y su companion `mmproj` en `Q8_0` (338 MB). Ambos tienen revisión y
SHA256 fijados. No requiere Torch ni Transformers.

```bash
uv run --extra streaming scribnic setup --workflow lfm3_audio_note
uv run --extra streaming scribnic run consulta.wav --workflow lfm3_audio_note

# Reproducir el WAV al ritmo de la grabación.
uv run --extra streaming scribnic run consulta.wav --workflow lfm3_audio_note --realtime
```

La CLI imprime la ruta de `report.html`. En el directorio de resultados también
quedan la nota, la configuración efectiva, la procedencia de ambos checkpoints,
el log del servidor, los eventos WebSocket en JSONL y la telemetría habitual.
Los eventos guardados no incluyen los bloques de audio enviados.

## Recursos y configuración

El preset `llama_cpp` selecciona upstream `b11946`, que incluye soporte de LFM3
([#31257](https://github.com/ggml-org/llama.cpp/pull/31257)) y conversación realtime
([#31269](https://github.com/ggml-org/llama.cpp/pull/31269)). Conserva el gestor
`LlamaServer` existente. El preset Prism de Bonsai sigue seleccionando su propia
distribución. No se instalan bindings de llama.cpp en Python.

El workflow vincula el companion mediante `[steps.note.companions]`; el runtime
recibe su ruta verificada con `--mmproj`. Los pesos permanecen cargados durante
la sesión Python y cada ejecución abre un WebSocket nuevo para aislar consultas.

Se pueden pasar opciones adicionales del ejecutable en TOML:

```toml
[steps.note.runtime_settings.server_options]
jinja = true
context-shift = false
temp = 0.0
top-k = 50
top-p = 1.0
min-p = 0.0
```

Las claves son nombres largos de llama-server sin `--`. Un booleano produce
`--nombre` o `--no-nombre`; un número o string produce una opción con un valor.
Se pasan como argumentos literales, sin shell. Las tablas de overrides sustituyen
la tabla del preset completa: incluir las opciones necesarias, especialmente
`context-shift = false`. Rutas de modelos, host, puerto, alias, contexto y ajustes
que Scribnic ya gestiona usan sus campos propios y no se pueden sobrescribir aquí.

El muestreo de realtime se configura en el servidor: esta versión no acepta
temperatura por `session.update`. `max_tokens` pertenece al modelo y se envía como
`max_output_tokens`. El preset limita cada consulta a 1.200 segundos, con contexto
32.768. Cambiar estos valores requiere evaluar memoria y calidad de audio largo.

## Prefill incremental y límites

El backend comparte una implementación para `run(AudioFile)` y `run_stream`.
Envía bloques de audio PCM16 mono a 16 kHz y el servidor procesa encoder y prefill
mientras llegan. Con un WAV sin `--realtime` los envía tan rápido como permite
la conexión; con captura en vivo consume el iterable del llamante.

Solo el fin explícito de la entrada hace `commit`, añade la instrucción de la
plantilla al mismo turno de usuario y solicita una nota. Se usa el mensaje de
sistema de audio understanding publicado por LiquidAI, fijado en
`model_settings.system_prompt`. La plantilla SOAP es la petición del usuario;
usar la plantilla como mensaje de sistema produjo transcripción y repeticiones en
la prueba real, tanto con HTTP como con WebSocket.
Una pausa no cierra la consulta. `output_modalities = ["text"]` desactiva la salida de voz.
No se publican `TranscriptUpdate`: no hay reconocimiento de texto intermedio.

```python
from scribnic import open_workflow

with open_workflow("lfm3_audio_note") as session:
    session.start()  # Cargar antes de empezar la captura.
    result = session.run_stream(audio_chunks, output_dir="runs/lfm3-consulta")
    print(result.note.text)
```

El llamante posee la captura y debe poder detener una fuente bloqueante.
`cancel` acepta un `threading.Event`, comprobado entre bloques y mientras se espera
la respuesta. La ejecución cierra el socket al cancelar o fallar y conserva el WAV
ya recibido. El backend no crea hilos emisores ni colas de audio en Python; los
buffers de transporte aplican backpressure. La cola interna del encoder y el
contexto del servidor pueden crecer hasta el límite configurado.

El prefill incremental reduce el trabajo pendiente al terminar de hablar; no
elimina el límite del contexto ni garantiza RAM constante. Se estima el presupuesto
con las 12,5 posiciones de audio por segundo del frontend y una reserva conservadora
para el prompt y la nota. El servidor usa `--no-context-shift`. Se rechazan entradas
que exceden los límites y respuestas vacías, canceladas o que alcanzan el máximo de
tokens. `b11946` informa `completed` incluso al alcanzar ese máximo, por lo que el
cliente comprueba también el uso de tokens.

Los tiempos de request incluyen espera de la fuente y procesamiento. La métrica
`first_text_after_response_request_seconds` mide desde la solicitud final hasta
el primer delta de texto; no equivale al tiempo total de procesamiento.

## Comprobación local

Comprobado el 28 de octubre de 2026 en WSL/Linux x86_64, CPU y seis hilos,
con el checkpoint Q4_K_M y el companion Q8_0 fijados por el preset:

- `consulta.wav`, de 297,891 segundos, produjo una nota en 60,93 segundos totales;
  el request duró 57,76 segundos y la carga del servicio, 1,92 segundos.
- Pico de la suma RSS muestreada: 2,96 GB entre cliente y servidor. Esta suma
  puede contar páginas compartidas varias veces y no mide RAM física exclusiva.
- Primer texto 0,94 segundos después de solicitar la respuesta. Se enviaron los
  fragmentos sin temporización de captura; no es una medición de micrófono real.
- Un fragmento de 24 segundos con `--realtime` produjo nota en 30,40 segundos
  totales, con primer texto 0,80 segundos después de solicitar la respuesta.
  Su informe queda en `runs/lfm3-q4-realtime/report.html`.

El informe local queda en `runs/lfm3-q4-soap/report.html`. La nota se genera, pero
mezcla tratamiento y recomendaciones en Objetivo y deja un Plan poco informativo;
esta ejecución no demuestra calidad clínica ni cobertura completa del audio.
No se han validado diez minutos continuos, GPU, macOS o Windows.

Fuentes: [modelo](https://huggingface.co/LiquidAI/LFM3-Audio-2B),
[GGUF](https://huggingface.co/LiquidAI/LFM3-Audio-2B-GGUF),
[protocolo fijado](https://github.com/ggml-org/llama.cpp/blob/b11946/tools/server/README.md#realtime-conversation).
