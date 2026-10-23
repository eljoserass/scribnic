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

## MOSS-Audio: posible experimento de audio a nota

Esta integración todavía no está implementada. MOSS-Audio es otro modelo: permite
comprensión, preguntas y resúmenes de audio por instrucciones. Una SOAP directa es
una hipótesis a evaluar; la ficha no acredita calidad clínica ni español. Probaría
Instruct primero y Thinking como comparación. [Ficha del modelo](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-4B-Thinking).

El checkpoint Thinking declara unos 9.73 GiB solo en pesos, antes de activaciones
y KV cache; no asumir que será ligero por llamarse 4B. Su contexto publicado es
40,960 posiciones y su encoder emite 12.5 representaciones por segundo: una hora
ya excedería ese contexto antes de prompt y respuesta.
[Índice de pesos](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-4B-Thinking/blob/5163125c56c9d66eb3887221ef28e4ee506d59ef/model.safetensors.index.json),
[configuración](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-4B-Thinking/blob/5163125c56c9d66eb3887221ef28e4ee506d59ef/config.json).

El ejemplo local contempla CUDA, MPS y CPU. Su servicio utiliza otro fork,
`OpenMOSS/sglang`, rama `moss-audio`, con chat multimodal; no es el servicio
SGLang Omni de transcripción anterior. La salida SSE documentada no demuestra
entrada incremental de audio ni function calling fiable sobre ese audio.
[Inferencia local](https://github.com/OpenMOSS/MOSS-Audio/blob/66326e6e0db34f036c86a76ba005efa4830c69dd/infer.py),
[guía del servidor](https://github.com/OpenMOSS/MOSS-Audio/blob/66326e6e0db34f036c86a76ba005efa4830c69dd/moss_audio_usage_guide.md).

El contrato futuro sería un backend `Audio → Note`, con template y sampling en el
workflow. El planificador no necesita un transcript intermedio. La memoria por
ventanas sería una estrategia adicional nuestra: cada ventana produciría una
actualización validada o `sin_cambios`, conservando evidencia temporal, negaciones
y correcciones antes de generar la nota final. Primero conviene probar audio
completo corto. Para aislar el efecto de diarizar, usar el mismo texto con y sin
etiquetas y el mismo generador; comparar dos sistemas distintos mide otra cosa.

- [Servidor recomendado por MOSS](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize#serve-with-sglang-omni)
- [Contrato y plataformas de SGLang Omni](https://github.com/sgl-project/sglang-omni/blob/main/docs/cookbook/moss_transcribe_diarize.md)
