# MOSS: modelo local o servicio de transcripción

`moss_bonsai` conserva la integración local de MOSS-Transcribe-Diarize con
Transformers. Solo esta ruta requiere `[moss]`; Torch y Transformers se importan
dentro del backend. Los snapshots y el código del modelo tienen revisión fijada.
El modelo genera transcripción diarizada conjuntamente, sin un diarizador externo.

`moss_bonsai_sglang` utiliza un servicio existente de SGLang Omni mediante el
cliente común `openai_transcription` y `response_schema=moss_segments`. La base
URL se configura con `SCRIBNIC_MOSS_BASE_URL`; la clave, si hace falta, mediante
`SCRIBNIC_MOSS_API_KEY`. `setup` no descarga el modelo ni instala SGLang en esta
ruta externa. El servicio es una instalación y un recurso independiente.

El cliente solicita `verbose_json`, conserva los tiempos y extrae etiquetas como
`[S01]` del prefijo de cada segmento. Las identidades ausentes quedan desconocidas.
Los parámetros `prompt`, `language`, `temperature` y `max_new_tokens` pertenecen a
la configuración de la petición; no se envían parámetros de Whisper o NVIDIA.

La ruta oficial de OpenMOSS recomienda CUDA 13 para SGLang Omni. El cookbook de
SGLang documenta además MOSS con GPU Intel. Su instalación Apple Silicon cubre
otros modelos; no implica soporte de MOSS. No hemos probado el servicio MOSS real
en esta máquina AMD ni en Mac. Las pruebas ordinarias verifican el contrato HTTP
con respuestas pequeñas, sin ejecutar el modelo.

Mantener el servicio cargado no convierte este modelo en ASR nativo de entrada
streaming. La emisión progresiva de texto y la recepción incremental de audio
son capacidades diferentes. Cortar MOSS externamente exige resolver la consistencia
de hablantes y no se activa automáticamente con VAD.

El backend Transformers permite profiling opcional de operadores Torch con
`--profile`; el cliente HTTP no puede obtener los operadores del proceso remoto.
Para ello el servicio debe instrumentarse y proporcionar sus propios artefactos.

## Pesos BF16, cómputo en CPU y cuantización

Revisión del 28 de octubre de 2026: OpenMOSS publica MOSS-Transcribe-Diarize
como modelo de **0.9B**, distinto de MOSS-Audio-4B-Thinking. El snapshot seleccionado
`704aa4a9c304e8520be88901e0d1960158ef5b15` contiene pesos BF16 y no declara
`quantization_config`. No se encontró un Q4/Q8 oficial de ese modelo en la
organización OpenMOSS. Las conversiones de cuentas personales quedan excluidas
por la [política de procedencia](model_sources.md); no se ha añadido ninguna.
[Modelo oficial](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize),
[arquitectura y uso del autor](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize).

El archivo ya descargado ocupa 1,817,113,576 bytes y su cabecera safetensors
identifica 683 tensores BF16. El preset registra ese formato como `BF16`, separado
del dtype efectivo que el backend escribe en el resultado.

Actualmente `dtype = "auto"` en este backend elige BF16 en CUDA y FP32 en CPU.
Por eso nuestra ejecución anterior en CPU amplió los pesos al cargarlos:
no habíamos descargado otra versión FP32. Sobre 297.891 segundos de `dialogue8`,
el paso de voz tardó 755.066 segundos con seis hilos y FP32. Ese dato pertenece
a esa configuración; no mide el rendimiento de BF16 ni de un quant.

El AMD Ryzen AI Max PRO 390 de esta máquina expone `avx512_bf16` a WSL. Torch
2.14.1+cpu confirma esa capacidad mediante `torch.cpu.get_capabilities()` y tiene
oneDNN disponible. Se hizo una comparación corta con el mismo checkpoint,
los primeros 15 segundos de `dialogue8`, seis hilos y SDPA. Cada dtype se ejecutó
en un proceso independiente, sin warmup y con una sola pasada:

| Medida | BF16 | FP32 |
| --- | ---: | ---: |
| Generación, incluido prefill | 13.775 s | 30.521 s |
| Tiempo hasta primer token dentro de generación | 2.147 s | 4.193 s |
| Ejecución completa, incluida carga | 21.382 s | 35.874 s |
| Pico RSS muestreado del proceso Python | 2.412 GB | 5.675 GB |
| Tokens generados | 180 | 180 |

Los artefactos resultaron idénticos: texto, siete segmentos, hablantes y tiempos.
Esto comprueba esa ruta de inferencia BF16, pero una muestra de 15 segundos no
establece equivalencia de calidad ni rendimiento para consultas completas.
La memoria es RSS muestreado cada 250 ms; no es una medida de pesos solos.
Los resultados generados, fuera del código versionado, están en
`runs/moss-cpu-dtype-20261028/{bfloat16,float32}/report.html` y sus `run.json`.
La [propuesta de selección de cómputo](hardware_selection.md) sigue sin implementar;
esta prueba no modifica los valores por defecto.

Para solicitar BF16 con la configuración existente, añadir al workflow elegido:

```toml
[steps.speech.runtime_settings]
device = "cpu"
dtype = "bfloat16"
```

### Qué supondría crear nuestro quant

Partir del checkpoint oficial permite un experimento reproducible sin depender
de conversiones ajenas. La cuantización posterior al entrenamiento no exige
reentrenar el modelo por definición; algunos métodos sí necesitan calibración.
Todavía falta implementar y validar esa ruta en Scribnic.

1. Establecer primero una referencia BF16 con consultas completas. En esta CPU
   ya funciona y mejora la prueba corta sin convertir los pesos a INT8/INT4.
2. Elegir un método y kernels compatibles con el dispositivo y las versiones
   instaladas. Transformers documenta `quantization_config`, por ejemplo mediante
   bitsandbytes; eso no acredita compatibilidad con este modelo personalizado.
   Cambiar `dtype` a `int8` no sustituye esas capas ni activa esos kernels.
   [Guía de Transformers](https://huggingface.co/docs/transformers/quantization/bitsandbytes).
3. Probar primero ocho bits en el decoder de texto y conservar inicialmente
   encoder/adaptador en coma flotante. Revisar los módulos realmente convertidos,
   pesos compartidos, movimiento al dispositivo y dtype de los inputs. El cargador
   actual y su `.to(device)` deben revisarse según el método seleccionado.
4. Registrar versión de herramientas, revisión original, módulos, formato de pesos
   y dtype efectivo. Las dependencias específicas pertenecerían al extra/backend,
   sin introducir Torch en la instalación base.
5. Comparar con la referencia sobre el mismo audio español: memoria, latencia,
   términos médicos, dosis, negaciones, hablantes y timestamps, y después la nota
   resultante. Probar cuatro bits solo si ocho bits aporta una mejora útil.

Menos bits reducen el almacenamiento de los tensores convertidos. No reducen en
esa misma proporción activaciones, KV cache ni todo el proceso; tampoco garantizan
aceleración si faltan kernels eficaces. Exportarlo a GGUF sería otra ruta de
integración: el llama.cpp fijado no implementa esta arquitectura MOSS. El formato
del archivo por sí solo no añade su encoder, preprocesado ni decoder al runtime.

## Hotwords y profiling del servicio

Los hotwords de MOSS son pistas de vocabulario dentro del prompt: pueden incluir
nombres de medicamentos o términos esperados. No constituyen un corrector posterior
ni garantizan su reconocimiento. El campo `prompt` sustituye las instrucciones
predeterminadas; por ello debe conservar la petición de timestamps y etiquetas de
hablante además de las pistas. Conviene comparar aciertos y términos añadidos que
no aparezcan en el audio. [Recetas de OpenMOSS](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize/blob/main/examples/prompts.md).

El [profiler de SGLang Omni](https://github.com/sgl-project/sglang-omni/blob/f073b1c6e3e6a5143f16ced1b9bd5d0c24ef9d76/docs/developer_reference/profiler.md)
ofrece dos capturas del servidor: eventos por petición y etapa en JSONL, y trazas
Torch de operadores CPU/dispositivo en formato Chrome trace comprimido, compatible
con Perfetto. Los eventos no sustituyen a las trazas de operadores. Las capturas
pueden compartir un `run_id`; sus archivos pertenecen al host del servidor.

El cliente de Scribnic todavía no controla esas capturas. Una futura integración
puede asociar los archivos al run, con inicio y cierre explícitos. Tampoco se asume
propagación automática de OpenTelemetry: en la revisión enlazada, la ruta de
transcripción genera su propio ID y no propaga `traceparent`. La respuesta SSE
expone `X-Request-Id`; la respuesta buffered no lo expone explícitamente.

El [cookbook](https://github.com/sgl-project/sglang-omni/blob/f073b1c6e3e6a5143f16ced1b9bd5d0c24ef9d76/docs/cookbook/moss_transcribe_diarize.md)
separa las ejecuciones medidas de la pasada adicional con `--profile-events`.
Conviene mantener esa separación, excluir warmup del tiempo de inferencia y
registrar o desactivar la caché del encoder al repetir audios. Sus resultados H100
BF16 no predicen el rendimiento de esta CPU AMD ni de una Mac.

## MOSS-Audio: audio a nota offline

`moss_audio_note` integra MOSS-Audio-4B-Thinking con un único paso `Audio → Note`.
Es otro modelo, con encoder propio y Qwen3: no usa MOSS-Transcribe-Diarize ni Bonsai.
La plantilla SOAP en español, sampling y presupuesto de salida pertenecen al workflow.
La calidad clínica y en español sigue siendo una hipótesis a evaluar.
[Ficha del modelo](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-4B-Thinking).

```bash
UV_PROJECT_ENVIRONMENT=.venv-moss-audio uv run --extra moss-audio \
  scribnic setup --workflow moss_audio_note
UV_PROJECT_ENVIRONMENT=.venv-moss-audio uv run --extra moss-audio \
  scribnic run consulta.wav --workflow moss_audio_note --profile
```

El extra fija Torch/torchaudio 2.9.1 y Transformers 4.57.1. No se puede combinar con
`[moss]`, que usa Transformers 5; se declaran extras incompatibles y se recomienda
un entorno separado. No se instalan Gradio/Streamlit ni SGLang para esta ruta.
El gestor de recursos descarga el source oficial por commit y SHA256, y el snapshot
de pesos por revisión. Las bibliotecas del modelo se importan al cargar este backend.

El checkpoint Thinking declara unos 9.73 GiB solo en pesos, antes de activaciones
y KV cache; no asumir que será ligero por llamarse 4B. Su contexto publicado es
40,960 posiciones y su encoder emite 12.5 representaciones por segundo: una hora
ya excedería ese contexto antes de prompt y respuesta.
[Índice de pesos](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-4B-Thinking/blob/5163125c56c9d66eb3887221ef28e4ee506d59ef/model.safetensors.index.json),
[configuración](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-4B-Thinking/blob/5163125c56c9d66eb3887221ef28e4ee506d59ef/config.json).

El backend selecciona CUDA, MPS o CPU si están disponibles; el workflow puede fijar
`device` y `dtype` en `[steps.note.runtime_settings]`. El cómputo automático usa
BF16 en CUDA, FP16 en MPS y FP32 en CPU, registrado aparte del formato de pesos.
Antes de cargar en CPU/MPS se comprueba que los pesos solos quepan en la memoria
disponible; eso es un límite inferior y no garantiza que quepan las activaciones.
El perfil conserva frontend, encoder, adapters, prefill y decode, con ventanas
Torch opcionales, CPU/RSS y procedencia en el mismo informe.
`[steps.note.runtime_settings] profile_gpu = true` solicita además actividades
CUDA o signposts Metal para Instruments cuando el dispositivo lo permite.

El ejemplo local contempla CUDA, MPS y CPU. Su servicio alternativo utiliza otro fork,
`OpenMOSS/sglang`, rama `moss-audio`, con chat multimodal; no es el servicio
SGLang Omni de transcripción anterior. La salida SSE documentada no demuestra
entrada incremental de audio ni function calling fiable sobre ese audio.
[Inferencia local](https://github.com/OpenMOSS/MOSS-Audio/blob/66326e6e0db34f036c86a76ba005efa4830c69dd/infer.py),
[guía del servidor](https://github.com/OpenMOSS/MOSS-Audio/blob/66326e6e0db34f036c86a76ba005efa4830c69dd/moss_audio_usage_guide.md).

El planificador ejecuta el paso directo y no inventa un transcript intermedio.
El backend separa el pensamiento del texto final y rechaza salida incompleta o
presupuestos de contexto que exigirían truncar audio. No implementa streaming.
La memoria por ventanas sería una estrategia adicional: cada ventana produciría una
actualización validada o `sin_cambios`, conservando evidencia temporal, negaciones
y correcciones antes de generar la nota final. Primero conviene probar audio
completo corto. Para aislar el efecto de diarizar, usar el mismo texto con y sin
etiquetas y el mismo generador; comparar dos sistemas distintos mide otra cosa.

Validación actual: tests con fakes de la frontera pública, e inferencia con una
instancia reducida aleatoria del código upstream para verificar formatos, generación
y exportación de trazas. Esa instancia no valida el checkpoint ni la calidad.
También se compararon los nombres y el número de parámetros de la arquitectura
completa en dispositivo meta con el índice fijado: coinciden; el buffer calculado
de posiciones lo contempla el cargador estándar de Transformers.
El checkpoint completo no se ejecutó en esta WSL de 11 GiB; falta la prueba en una
máquina con memoria suficiente. MPS/CUDA tampoco se han validado localmente.

- [Servidor recomendado por MOSS](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize#serve-with-sglang-omni)
- [Contrato y plataformas de SGLang Omni](https://github.com/sgl-project/sglang-omni/blob/main/docs/cookbook/moss_transcribe_diarize.md)
