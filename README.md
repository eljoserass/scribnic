# Scribnic

Núcleo local de escriba clínico con la misma API para CLI, aplicación y evaluación.
Un workflow declara cómo producir una nota desde audio o una transcripción. Incluye
Whisper.cpp, Nemotron multilingüe + diarizador con NeMo-Speech.cpp, MOSS opcional
con Transformers o un servicio HTTP existente, y Bonsai mediante llama-server Prism.
MOSS-Audio y LFM3-Audio permiten generar la nota directamente desde audio, sin
transcripción. LFM3 usa llama-server y admite prefill incremental.

## Arquitectura del refactor

El workflow selecciona los pasos y sus recursos. Un **modelo** identifica los pesos
y su precisión; un **backend** adapta una API de inferencia a los artefactos de
Scribnic; un **gestor de runtime** instala o localiza el motor, arranca el proceso,
espera a que esté listo y lo cierra. El motor real puede ser un ejecutable C++ o un
servicio existente. Compartir HTTP no implica compartir formatos ni capacidades.

Las rutas HTTP comparten clientes; los gestores conservan los modelos cargados:

```mermaid
flowchart TB
    subgraph callers["Entradas: mismo núcleo"]
        CLI["CLI · cli.py"]
        APP["Aplicación / plugin"]
        EVAL["scribe_eval · repositorio externo"]
    end
    CLI --> CORE
    APP --> CORE
    EVAL --> CORE

    CORE["WorkflowSession · workflow.py<br/>Resuelve productores y aísla cada ejecución"]
    CONFIG["Workflow TOML<br/>Modelo + variante de pesos + runtime<br/>Backend + parámetros + template"] --> CORE

    subgraph execution["Ejecución y artefactos"]
        AUDIO["Audio WAV / PCM incremental"] --> SPEECH["backends/openai_transcription.py<br/>HTTP, NeMo WebSocket o ventanas Whisper"]
        SPEECH --> TRANSCRIPT["Transcript o DiarizedTranscript<br/>artifacts.py · datos guardables y reutilizables"]
        TRANSCRIPT --> NOTE["backends/openai_chat.py<br/>Template SOAP + conversación"]
        NOTE --> OUTPUT["Note"]
        AUDIO --> AUDIO_NOTE["backends/moss_audio.py<br/>MOSS-Audio · Audio → Note offline"]
        AUDIO_NOTE --> OUTPUT
        AUDIO --> LFM_NOTE["backends/llama_audio_note.py<br/>LFM3 Q4_K_M · prefill incremental → Note"]
        LFM_NOTE --> OUTPUT
    end
    CORE --> SPEECH
    CORE --> NOTE
    CORE --> AUDIO_NOTE
    CORE --> LFM_NOTE
    SUPPLIED["Transcripción suministrada<br/>También permite perturbaciones del evaluador"] --> TRANSCRIPT

    subgraph services["Motores de inferencia seleccionados"]
        WS["whisper-server<br/>Whisper multilingüe Q8_0"]
        NS["nemo-speech serve<br/>Nemotron ASR + diarizador Q8_0"]
        MS["Servicio MOSS existente<br/>SGLang Omni · plataforma compatible"]
        LS["llama-server · distribución Prism<br/>Bonsai PQ2_0"]
        LFS["llama-server · upstream b11946<br/>LFM3-Audio Q4_K_M + mmproj Q8_0"]
    end
    SPEECH <-->|"API de transcripción"| WS
    SPEECH <-->|"HTTP / protocolo streaming específico"| NS
    SPEECH <-->|"API de transcripción"| MS
    NOTE <-->|"Chat Completions"| LS
    LFM_NOTE <-->|"WebSocket /v1/realtime"| LFS

    MANAGERS["runtimes/<br/>Ciclo de vida común + comandos por motor<br/>Solo administra procesos propios"]
    CORE --> MANAGERS
    MANAGERS -. "arranque / disponibilidad / cierre" .-> WS
    MANAGERS -. "arranque / disponibilidad / cierre" .-> NS
    MANAGERS -. "arranque / disponibilidad / cierre" .-> LS
    MANAGERS -. "arranque / disponibilidad / cierre" .-> LFS

    CORE -. "spans y métricas" .-> OBS["telemetry/<br/>OpenTelemetry · CPU / RSS por proceso"]
    MANAGERS -. "PID, versión y logs" .-> OBS
    CORE --> STORE["storage.py / reporting.py<br/>Predicciones, procedencia e informe local"]
```

El despliegue del médico incluye las dependencias, pesos y motores necesarios para
su workflow. El código puede conocer otros proveedores sin importar ni instalar
sus bibliotecas. El entorno de evaluación puede preparar varios workflows; los
datasets y métricas permanecen fuera de Scribnic. Un servicio externo ya instalado
no se descarga ni se termina desde el cliente.

El ciclo de vida es `setup → start → run → close`: preparar recursos en
disco, cargar los pasos seleccionados, reutilizarlos entre consultas y liberar los
procesos propios. La carga y el warmup se miden por separado de la inferencia. Los
artefactos y caches de una grabación nunca se reutilizan como estado de otra.
SGLang Omni para MOSS requiere validar la combinación concreta de modelo y
plataforma; no se promete soporte local de MOSS en AMD o Mac por usar HTTP.

## Instalar y ejecutar

Python 3.12–3.14 y [uv](https://docs.astral.sh/uv/). Desde el repositorio:

```bash
uv sync --locked
uv run scribnic setup --workflow nvidia_cpp_offline
uv run scribnic run consulta.wav --workflow nvidia_cpp_offline
```

`setup` descarga/verifica solo los recursos del workflow, sin cargar modelos.
`prepare` sigue disponible como alias. Los workflows C++ no necesitan Torch,
Transformers ni bindings de llama.cpp. Para instalar un cliente desde un wheel,
usa `pip install scribnic-0.1.0-py3-none-any.whl`; el grupo `dev` no se instala.
En un checkout, `uv sync --no-dev` excluye las herramientas de desarrollo.

El runtime oficial NeMo-Speech.cpp y Bonsai/Prism se descargan de releases fijadas
con SHA256. Whisper se compila automáticamente desde el source fijado de v1.9.5:
requiere un compilador C/C++ y CMake o `uvx`; este último obtiene CMake 4.1.0 en un
entorno de herramientas aislado. No hace falta configurar una ruta al ejecutable.

Abre el `report.html` de la ruta impresa. Conserva nota, transcripción, tiempos,
configuración, procedencia y recursos por proceso. El audio permanece en su ruta
original. El WAV debe ser PCM16, mono, 16 kHz:

```bash
ffmpeg -i entrada.mp3 -ac 1 -ar 16000 -c:a pcm_s16le consulta.wav
```

## Workflows disponibles

| Workflow | Voz | Nota / instalación opcional |
| --- | --- | --- |
| `nvidia_cpp_offline` | Nemotron 3.5 ASR Q8_0 + Nemotron 3 Diarization Q8_0; archivo por HTTP | Bonsai PQ2_0 / Prism; instalación base |
| `nvidia_cpp_streaming` | Los mismos modelos; WAV por WebSocket, sin esperas de tiempo real | Bonsai / Prism; extra `[streaming]` |
| `nvidia_cpp_asr_offline` | Solo Nemotron 3.5 ASR Q8_0; transcripción sin hablantes por HTTP | Bonsai / Prism; instalación base |
| `nvidia_cpp_asr_streaming` | Solo Nemotron 3.5 ASR Q8_0; WAV por WebSocket, sin esperas de tiempo real | Bonsai / Prism; extra `[streaming]` |
| `whisper_bonsai` / `whisper_bonsai_vad` | Whisper small multilingüe Q8_0, con VAD opcional | Bonsai / Prism; instalación base |
| [Presets Whisper por tamaño](docs/asr_presets.md) | tiny, base, small, medium y turbo Q8_0; large-v3 Q5_0; con VAD | Bonsai / Prism; instalación base |
| `parakeet_v3_q8_0_bonsai` / `parakeet_v3_q8_0_diarized_bonsai` | Parakeet v3 Q8_0 offline, con diarizador Nemotron Q8 opcional | Bonsai / Prism; instalación base |
| `whisper_bonsai_streaming` | El mismo whisper-server residente; ventanas HTTP de 10 s con 2 s de solape | Bonsai / Prism; instalación base |
| `moss_bonsai` | MOSS local / Transformers | Bonsai / Prism; extra `[moss]` |
| `moss_bonsai_http` | MOSS local / Transformers | Bonsai en endpoint existente; extra `[moss]` |
| `moss_bonsai_sglang` | MOSS en SGLang Omni existente | Bonsai / Prism; instalación base del cliente |
| `bonsai_note` | Entrada de texto | Bonsai / Prism; instalación base |
| `moss_audio_note` | Audio completo → nota con MOSS-Audio-4B-Thinking BF16 | Extra `[moss-audio]`, sin Bonsai ni transcripción |
| `lfm3_audio_note` | Audio incremental → nota con LFM3-Audio-2B Q4_K_M + mmproj Q8_0 | llama-server upstream; extra `[streaming]`, sin Torch ni transcripción |

También hay workflows de texto → nota para Qwen3.5 de 0.8B, 2B y 4B en Q4_K_M
y Q8_0, Bonsai 2 4B PQ2_0 y LFM3-1.7B de texto en Q4_K_M, Q8_0 y QAD-Q4_0.
Reutilizan `openai_chat` con `llama_cpp` o `llama_cpp_prism`, sin otras dependencias.
[Catálogo, tamaños y comandos para compararlos](docs/note_models.md).
[Procedencia y política de pesos cuantizados](docs/model_sources.md).

La investigación de próximos modelos, licencias y benchmarks está en
[candidatos ASR y audio a nota](docs/asr_models.md). Los
[presets ASR nativos](docs/asr_presets.md) incluyen tamaños y comandos para Whisper
y Parakeet, usando los runtimes existentes. La revisión de
[Unsloth y Whisper](docs/model_publishers.md) distingue formatos y publicadores.
El [mapa de selección de hardware](docs/hardware_selection.md) es una propuesta
sin implementar; incluye la evidencia de la prueba MOSS BF16 frente a FP32.

Los workflows streaming procesan mientras consumen audio: NVIDIA conserva su estado
ASR/diarizador por WebSocket, Whisper reconoce ventanas por HTTP y LFM3 realiza
prefill directo del audio. La nota se genera
al terminar. `--realtime` reproduce el WAV al ritmo de llegada; sin él se procesa
tan rápido como permita el consumidor. La API Python acepta PCM de otra fuente;
la captura física del micrófono pertenece al llamante. [Contrato y ejemplos](docs/streaming.md).

```bash
# Streaming NVIDIA: solo se añade el pequeño cliente WebSocket.
uv run --extra streaming scribnic run consulta.wav --workflow nvidia_cpp_streaming

uv run scribnic run consulta.wav --workflow whisper_bonsai_streaming --realtime

# Nota directa desde audio con LFM3, procesado por fragmentos.
uv run --extra streaming scribnic run consulta.wav --workflow lfm3_audio_note

# MOSS local: sus dependencias quedan limitadas a esta instalación opcional.
uv run --extra moss scribnic run consulta.wav --workflow moss_bonsai

# MOSS servido por una instalación compatible de SGLang Omni, fuera del cliente.
SCRIBNIC_MOSS_BASE_URL=http://127.0.0.1:8000/v1 \
  uv run scribnic run consulta.wav --workflow moss_bonsai_sglang
```

Cada workflow declara los extras que necesita en esta tabla. `setup` prepara pesos
y motores, no instala paquetes Python dentro del entorno que lo invoca. Una
instalación de evaluación usa entornos separados para `[moss]` y `[moss-audio]`,
que requieren versiones incompatibles de Transformers. `--all-extras` no es válido.
El cliente médico instala solo los extras seleccionados. El código de proveedores que no se
usan no carga sus bibliotecas. `uv run --no-sync` conserva un entorno ya preparado.

NeMo-Speech.cpp conecta ASR y diarización por atribución temporal de palabras.
No implementamos aquí el ASR enmascarado por hablante de NeMo/Python; la referencia
Python anterior se retiró del núcleo, junto al extra `[nvidia]`. Esa diferencia de
estrategia y la cuantización impiden tratar sus resultados como equivalentes en
calidad. [NVIDIA: servidor, modos y profiling](docs/nvidia.md).

MOSS/SGLang se integra como servicio externo: Scribnic no instala SGLang ni reclama
medir recursos de otro host. Las rutas de MOSS documentadas por sus autores cubren
CUDA y GPU Intel; no hemos validado MOSS/SGLang en AMD o Mac. La opción Transformers
local sigue disponible. [MOSS: configuración y límites](docs/moss.md).

MOSS-Audio usa código upstream fijado, Torch 2.9.1 y Transformers 4.57.1 en su extra:

```bash
UV_PROJECT_ENVIRONMENT=.venv-moss-audio uv run --extra moss-audio \
  scribnic run consulta.wav --workflow moss_audio_note --profile
```

La integración MOSS-Audio se comprobó con una instancia reducida aleatoria para
validar inferencia y profiling; el checkpoint completo y Apple MPS requieren una
prueba en una máquina con memoria suficiente. Esa comprobación no mide calidad.

LFM3 utiliza un único backend para WAV y audio en vivo, con una sesión WebSocket
nueva por consulta. El preset `llama_cpp` fija una versión upstream; Bonsai conserva
Prism. Las opciones adicionales del ejecutable se configuran mediante
`runtime_settings.server_options` en TOML. [LFM3: configuración y límites](docs/lfm3.md).

## API Python: carga separada de inferencia

```python
from scribnic import AudioFile, Transcript, open_workflow

with open_workflow("nvidia_cpp_offline") as scribe:
    startup = scribe.start()  # Opcional: precarga antes de medir consultas.
    result = scribe.run(AudioFile("consulta.wav"), output_dir="runs/consulta")
    print(result.note.text)

    speech = scribe.run(AudioFile("otra.wav"), output="diarized_transcript")
    plain = speech.transcript.without_speakers()
    ablation = scribe.run(plain)  # Solo nota sobre el texto suministrado.
    manual = scribe.run(Transcript.from_text("Paciente: tengo tos desde ayer."))
```

`start(source_kind="transcript")` carga solo lo necesario para esa entrada. Devuelve
los tiempos y recursos del arranque para guardarlos por separado en evaluación.
`run` también arranca servicios bajo demanda si no se llamó a `start`. La CLI cierra
la sesión tras una ejecución; la app o el evaluador mantienen una sesión para varios
audios. `close` cierra solo procesos propios, nunca servicios externos.

`start()` es precarga explícita; Scribnic no envía una consulta de prueba para
garantizar warmup de todas las operaciones. Después, `run()` reutiliza esos modelos
y sigue validando entradas, convirtiendo salidas y guardando resultados si se pide.
La opción propuesta `keep_loaded` aún no existe: conservar un modelo después de
usarlo y cargarlo por adelantado son decisiones distintas. Queda pendiente la
retención por servicio, incluida la granularidad ASR/diarización de NVIDIA:
[notas de diseño](docs/nvidia.md#pendiente-residencia-por-servicio).

Los modelos permanecen cargados; cada grabación recibe estado de reconocimiento y
artefactos nuevos. `without_speakers()` conserva texto y tiempos, elimina identidades
y roles mediante una copia explícita. Speaker IDs y roles clínicos son conceptos
distintos. El modelo de nota recibe exactamente la transcripción suministrada.

`open_workflow(..., prepare=False)` evita preparar recursos de pasos que no lleguen
a ejecutarse. `download=False` exige caché. Las llamadas de una sesión se serializan;
el harness controla la concurrencia según la memoria disponible. Datasets, jueces,
perturbaciones y métricas de calidad pertenecen a `scribe_eval`, fuera del núcleo.

Para evaluación paralela, cada worker puede mantener su propio proceso Python y
una `WorkflowSession` residente. Las sesiones comparten la caché en disco, pero
cada una arranca sus servidores propios bajo demanda, con PIDs y puertos distintos
si no se fija un puerto. Sus estados de inferencia y artefactos permanecen separados;
la RAM necesaria aumenta con las réplicas. Cada muestra necesita su directorio de
resultados. Una sola sesión serializa sus llamadas, incluso desde varios hilos.

Si varios workers configuran la misma URL externa, comparten el servidor: crear
clientes no crea réplicas del modelo. El harness deberá limitar peticiones en vuelo
y distinguir latencia por consulta de throughput total. Actualmente Bonsai se
lanza con un slot y NVIDIA con batching desactivado; los gestores propios no
configuran un servicio compartido para alta concurrencia.

Preparar primero los workflows con `setup`, y después abrir workers con
`download=False`, evita instalaciones concurrentes: la extracción y compilación de
runtimes todavía no tienen bloqueo entre procesos. No hay un pool ni reparto
automático de CPU/RAM en Scribnic; el harness debe fijar workers e hilos por runtime.
Para esos experimentos conviene registrar también recursos de la máquina completa:
los informes individuales no representan por sí solos el consumo total concurrente.

## Configuración y CLI

La CLI usa Typer y Rich, declarados en la instalación base; no necesita instalar
MOSS para disponer de colores y progreso. Importar la API de Python no importa
Typer, Rich ni Torch.

En una terminal muestra las fases que realmente ejecuta el workflow, spinners y
tiempo transcurrido, carga de servicios, verificación de caché y guardado del
informe. Las descargas nativas muestran bytes y velocidad; aparece una barra solo
cuando el servidor informa su tamaño. La inferencia por HTTP no ofrece un
porcentaje fiable y se presenta con un spinner. No interpretamos el presupuesto
de tokens como un total de trabajo. La exportación de trazas Torch tiene su fase.
Las fases de transcripción y generación empiezan después de cargar su servicio;
el resumen conserva el tiempo total de cada etapa, incluida su carga inicial.
Esto cubre Whisper, NVIDIA offline y WebSocket, MOSS por HTTP y Transformers,
y las notas por Chat Completions. La precarga desde Python también identifica
el modelo; reutilizarlo no muestra otra carga ficticia.

El progreso y los avisos van a `stderr`; `--format json` mantiene un único documento
JSON en `stdout`. Al redirigir la salida de progreso se imprimen líneas estables,
sin animaciones ni ANSI. `NO_COLOR` desactiva los colores. `--verbose` / `-v` muestra
los diagnósticos de Scribnic; los logs de servidores y la telemetría siguen en el
informe/directorio del run. Los avisos accionables siguen visibles sin verbose.
`Ctrl+C` cierra la sesión y los servidores propios, y conserva el código de salida
130 y los artefactos ya guardados.

```bash
uv run scribnic run consulta.wav --workflow whisper_bonsai
uv run scribnic run consulta.wav --workflow whisper_bonsai --format json > result.json
uv run scribnic setup --workflow nvidia_cpp_offline --verbose
# Instalar el autocompletado para el shell actual.
uv run scribnic --install-completion
```

La presentación vive en `terminal.py`. Las señales de `telemetry/progress.py`
quedan limitadas al contexto del llamante; el core no importa librerías de UI.
Los spans de OpenTelemetry existentes proporcionan fases de ejecución y carga,
sin leer ni sondear archivos de telemetría para actualizar la terminal.

Un paso selecciona su **backend**, modelo y runtime de forma explícita:

```toml
name = "consulta"
default_output = "note"

[steps.speech]
backend = "openai_transcription"
model = "nemotron-3.5-asr-streaming-0.6b.q8_0"
runtime = "nemo_cpp"
output = "diarized_transcript"

[steps.speech.companions]
diarization = "Nemotron-3-Diarization.q8_0"

[steps.note]
backend = "openai_chat"
model = "Ternary-Bonsai-2-2B-PQ2_0"
runtime = "llama_cpp_prism"
template = "soap_es"
source = "speech"

[steps.note.model_settings]
temperature = 0.0
```

El preset del modelo declara entradas, salidas y formato de respuesta. El workflow
puede sobrescribir ajustes y solicitar una salida compatible. El resolver detecta
productores ausentes, ambigüedades y ciclos antes de preparar modelos. Un archivo,
`default.toml` o el único workflow de una carpeta determinan la selección.

Migración: `backend` pasa del modelo al paso; los IDs de variantes de Nemotron y
Bonsai incluyen cuantización; `http` se llama `external_endpoint`. El motor
`llama_cpp` sigue siendo una familia de servidores: el preset `llama_cpp_prism`
selecciona explícitamente el fork y la versión compatibles con Bonsai PQ2_0.

```bash
# Detenerse en el artefacto de voz.
uv run scribnic run consulta.wav --workflow nvidia_cpp_offline \
  --output diarized_transcript --output-dir runs/voz

# Generar nota desde un artefacto guardado, sin reconocimiento de audio.
uv run scribnic run runs/voz/diarized_transcript.json --workflow bonsai_note

# JSON en stdout; progreso en stderr.
uv run scribnic run conversacion.txt --workflow bonsai_note --format json
```

`--offline` exige recursos locales, pero permite llamar al endpoint seleccionado;
no es un bloqueo de red. `--profile` añade profiling detallado cuando el motor lo
permite. Sampling, templates, dispositivos e hilos se configuran en TOML.
El preset `external_endpoint` admite `SCRIBNIC_BASE_URL` y `SCRIBNIC_API_KEY`.

Bonsai usa el [perfil publicado para modelos pequeños](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/SMALL-MODELS.md),
con thinking desactivado. Los workflows de voz C++ usan `temperature=0` como baseline.
Las entradas que exceden el límite configurado se rechazan sin truncar; una nota
incompleta o con thinking tampoco se acepta. La calidad clínica queda por evaluar.

## Observabilidad y resultados

Se guardan los artefactos realmente producidos, `result.json`, `run.json`,
`report.html`, diagnósticos `runtime/*` y telemetría OTLP en `telemetry/*.jsonl`.
Cada etapa se persiste antes de comenzar la siguiente. Un fallo expone el resultado
parcial mediante `WorkflowRunError.result`. Los resultados existentes no se pisan.

Se muestrean CPU y RSS de Python **y de cada proceso propio**: `nemo-speech`,
`whisper-server` y `llama-server`. El informe identifica los ejecutables; el JSON
incluye el pico de la suma RSS de cada muestreo. Esa suma puede contar páginas
compartidas varias veces y no equivale a RAM física ni VRAM. No se suman máximos
individuales tomados en momentos distintos.

Los servidores nativos conservan logs y procedencia; `--profile` declara que no
captura operadores C++ con Torch. MOSS/Transformers permite trazas de operadores
comprimidas para Perfetto; instrumentar y exportar añade tiempo y memoria. Los
benchmarks de latencia se ejecutan sin esa captura y separan carga de inferencia.

La biblioteca usa APIs OpenTelemetry; la app o el harness configuran SDK/exporters.
`scribnic.telemetry.Telemetry` proporciona exportación local opcional sin
reemplazar el proveedor global del proceso anfitrión.

La instrumentación se agrupa en `telemetry/`: `otel.py` configura spans, métricas
y exportadores; `process_metrics.py` muestrea CPU/RSS; `profiling.py` captura
perfiles detallados opcionales; `progress.py` emite avisos independientes de la UI.

## Desarrollo

```bash
uv sync --locked --extra moss --extra streaming
uv run ruff check src tests
uv run ruff format --check src tests
uv run ruff check src/scribnic/telemetry --extend-select D,ANN
uv run python -m unittest discover -s tests -v
```

Las pruebas ordinarias usan fakes, no descargan modelos ni contactan servicios de
inferencia. El test real de MOSS se activa mediante `SCRIBNIC_LIVE_TESTS=1` y
`SCRIBNIC_TEST_AUDIO`. Las reglas de responsabilidades, documentación y estilo están
en [AGENTS.md](AGENTS.md). Validación de modelos y binarios: WSL/Linux x86_64 CPU;
los assets de macOS/Windows están configurados, pero requieren pruebas en sus hosts.

Ruff comprueba estilo, imports y formato de docstrings Google. El chequeo adicional
de `telemetry/` exige docstrings y firmas anotadas en ese paquete ya migrado.
Las responsabilidades y decisiones de diseño se revisan siguiendo las referencias
de Google enlazadas en `AGENTS.md`; los linters no sustituyen esa revisión.
