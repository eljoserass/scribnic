# Candidatos de voz para notas clínicas

Investigación del 28 de octubre de 2026. El criterio del proyecto es conservar la
información clínica en español y su atribución, con un coste local razonable.
Un resultado de ASR no demuestra por sí solo que la nota final sea buena.
Las incorporaciones de esta página son propuestas: no añaden backends ni pesos.
Se aplica la [política de procedencia](model_sources.md).

## Lo que ya tenemos

| Modelo | Uso actual | Licencia de los pesos |
| --- | --- | --- |
| [Whisper small](https://github.com/openai/whisper#available-models-and-languages), 244M | GGML Q8_0 con whisper.cpp; español, sin diarización propia | MIT; distribución oficial de whisper.cpp aceptada |
| [Nemotron 3.5 ASR Streaming 0.6B](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/tree/1c8deaecc64b91f034d73e08dd8b64625eb3395d) | GGUF Q8_0, español, offline o streaming con estado | OpenMDW-1.1 |
| [Nemotron 3 Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization/tree/f667ed73aee57d40cc39428eb768b4fd87a0a29e), unos 99M | GGUF Q8_0; atribución de hasta ocho hablantes junto al ASR | OpenMDW-1.1 |
| [MOSS-Transcribe-Diarize](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize), 0.9B | Checkpoint BF16, Transformers; texto, tiempos y hablantes conjuntamente | Apache-2.0 |

El NVIDIA actual **no es Parakeet v2 ni v3**. Usa un transductor RNNT multilingüe
con estado de streaming. El reconocimiento y el diarizador son modelos separados.
[Contrato e implementación actual](nvidia.md).

MOSS-Transcribe-Diarize tampoco es MOSS-Audio-4B-Thinking. El primero tiene unos
0.9B parámetros; el segundo es un modelo de audio a texto general con otra
arquitectura. El checkpoint de transcripción descargado es BF16; el backend lo
cargaba en FP32 por defecto al seleccionar CPU. La prueba corta BF16 frente a FP32
y los límites de esa evidencia están en [MOSS](moss.md).

OpenMDW-1.1 es una licencia distinta de CC BY 4.0: sus términos están en la
[licencia del proyecto](https://openmdw.ai/license/1-1/). No atribuir a todos los
modelos NVIDIA la misma licencia. Las fichas y revisiones enlazadas identifican
los pesos seleccionados; la licencia del motor es una pieza independiente.

## Orden propuesto y trabajo necesario

Para obtener una comparación nueva con poco desarrollo, probar primero Parakeet
v3. Qwen3-ASR-0.6B tiene más interés como siguiente candidato médico multilingüe.
Ultra ofrece resultados públicos prometedores de calidad; Redux interesa por
memoria y CPU, a cambio de una degradación más clara con ruido. Canary queda detrás
por el trabajo de integración y por la evidencia médica disponible.

| Candidato | Español y hablantes | Qué habría que cambiar en Scribnic |
| --- | --- | --- |
| [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | 25 idiomas, incluido español; ASR sin etiquetas de hablante propias | Modelo y workflow TOML sobre NeMo-Speech.cpp v0.2.0, con el GGUF Q8 oficial de NVIDIA; validar contrato offline, idioma y tiempos |
| [Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) | 30 idiomas y 22 dialectos chinos; sin diarización propia | Reutilizar gestor llama-server; adaptar la petición de audio y la conversión de respuesta ASR, añadir presets y contrato de texto sin tiempos inventados |
| [Parakeet Ultra](https://huggingface.co/moondream/parakeet-ultra) | 25 idiomas; sin diarización propia | Integrar Photon o convertir y validar sus safetensors, incluido el nuevo cabezal VAD; el conversor `.nemo` actual no lo resuelve directamente |
| [Parakeet Redux](https://huggingface.co/moondream/parakeet-redux) | Los mismos 25 idiomas; encoder ternario | Integrar el runtime y formato empaquetado del autor; no es sustituir un archivo GGUF en nuestro NeMo actual |
| [Canary 1B v2](https://huggingface.co/nvidia/canary-1b-v2) | 25 idiomas; transcripción y traducción; sin diarización propia | Backend para su ruta NeMo/Transformers o soporte nativo adicional; el runtime C++ fijado no incluye su decodificador AED |
| [Parakeet TDT 0.6B v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) | Solo inglés | No priorizar para consultas españolas, aunque figure bien en una métrica médica inglesa |

Parakeet usa un transductor TDT, que predice tokens y duración. Canary usa un
encoder y un decoder de atención autoregresivo, y añade traducción. El nombre
NVIDIA no hace intercambiables sus arquitecturas ni sus protocolos.

NeMo-Speech.cpp v0.2.0 documenta Parakeet v3 como **offline**: rechaza peticiones
de streaming con estado. El GGUF de NVIDIA ya publicado evita crear nuestro quant.
Una variante del motor que soporte RNNT no implica que soporte cualquier AED.
[Modelos y formatos del runtime fijado](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/v0.2.0/docs/asr/models.md).

llama.cpp b11946 ya enumera Qwen3-ASR y
[ggml-org publica su GGUF Q8 y companion](https://huggingface.co/ggml-org/Qwen3-ASR-0.6B-GGUF).
Eso permite reutilizar el gestor; nuestro backend de transcripción actual aún no
convierte su respuesta de idioma y texto. El modelo oficial ofrece streaming por
vLLM; no demuestra que nuestro cliente llama-server tenga ese protocolo. Los tiempos
de palabra requieren una ruta de alineación adicional, como ForcedAligner.
[Capacidades del motor fijado](https://github.com/ggml-org/llama.cpp/blob/b11946/docs/multimodal.md),
[contratos del autor](https://huggingface.co/Qwen/Qwen3-ASR-0.6B#quickstart).

Ultra y Redux dividen grabaciones largas en pausas, con segmentos de hasta 30 s y
un cabezal VAD incluido. Pueden procesar una conversación de diez minutos por
fragmentos; esa estrategia no equivale a ASR nativo con caché de streaming.
Whisper, Qwen, Parakeet y Canary necesitan además una estrategia explícita para
atribuir hablantes; un transcript de una conversación no es un transcript diarizado.

Qwen3-ASR declara Apache-2.0. Parakeet v2/v3, Ultra, Redux y Canary v2 declaran
CC BY 4.0 en sus fichas oficiales: permiten uso comercial con las condiciones de
atribución correspondientes. Estas son licencias de modelos concretos, no una
garantía sobre cualquier conversión de terceros que use sus nombres.

## Quants publicados y coste de los archivos

Los tamaños siguientes son decimales y corresponden a archivos, no a RAM de
inferencia. Se distinguen publicaciones del autor y conversiones de organizaciones.
No se han añadido ni ejecutado estos nuevos candidatos en Scribnic.

| Modelo | Fuente y formato comprobados | Tamaño de pesos |
| --- | --- | ---: |
| Parakeet TDT 0.6B v3 | NVIDIA publica GGUF Q8_0 junto al checkpoint original | 714 MB |
| Parakeet Ultra | Moondream publica safetensors F16, con ciertos tensores F32; sin Q4/Q8 del autor en el repositorio revisado | 1.255 GB |
| Parakeet Redux | Moondream publica encoder ternario empaquetado, con otros pesos F16; la compresión forma parte de esta variante entrenada | 178 MB |
| Qwen3-ASR-0.6B | ggml-org publica GGUF Q8_0 de texto y companion de audio Q8_0 | 805 + 214 MB, unos 1.019 GB en total |
| Canary 1B v2 | NVIDIA publica safetensors F32 y checkpoint NeMo; no se encontró Q4/Q8 del autor | 3.916 GB para safetensors |

Fuentes: [archivos de NVIDIA Parakeet](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/tree/main),
[Ultra](https://huggingface.co/moondream/parakeet-ultra/tree/main),
[Redux](https://huggingface.co/moondream/parakeet-redux/tree/main),
[Qwen GGUF](https://huggingface.co/ggml-org/Qwen3-ASR-0.6B-GGUF/tree/main) y
[Canary](https://huggingface.co/nvidia/canary-1b-v2/tree/main).
En Qwen, contar solamente el archivo de texto omite el encoder de audio. En
Canary, el tamaño del paquete NeMo completo tampoco equivale al de sus tensores.
También hay [Qwen3-ASR-0.6B de cuatro bits en MLX Community](https://huggingface.co/mlx-community/Qwen3-ASR-0.6B-4bit),
unos 708 MB, con Apache-2.0. Es otra organización y otro formato: requiere MLX en
Apple Silicon; no lo carga nuestro llama-server ni esta CPU AMD mediante MLX.

### Por qué Ultra requiere trabajo adicional

Ultra mantiene el backbone TDT de Parakeet v3. El obstáculo actual es la ruta de
carga y ejecución: NeMo-Speech.cpp v0.2.0 espera un GGUF con su esquema y su
conversor ASR busca `.nemo`, configuración YAML, pesos `.ckpt` y tokenizer NeMo.
Ultra publica `config.json`, safetensors y tokenizer JSON. No basta cambiar el
repositorio en un TOML. Además, hay que conservar el cabezal VAD y la segmentación
que utiliza para audio largo.
[Conversor fijado de NVIDIA](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/v0.2.0/conversion/source.py).

Eso no obliga a crear un motor completo ni demuestra que el núcleo ASR no pueda
adaptarse. Las rutas posibles son añadir un conversor compatible y validar las
diferencias de ejecución, o integrar Photon del autor. Ultra y Redux podrían
compartir una integración Photon; Redux necesita además sus kernels para pesos
ternarios empaquetados. Su [formato publicado](https://huggingface.co/moondream/parakeet-redux/blob/main/ternary.json)
es `thrush-ternary-v2`, con cinco valores ternarios por byte y escalas por grupos;
no es el PQ2_0 de Bonsai ni un GGUF Q2 genérico. Las capacidades concretas de cada
modelo seguirían siendo explícitas.

### Conversiones de otra organización: Handy

La organización [Handy](https://huggingface.co/handy-computer) enlaza su aplicación
y el proyecto [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp).
Publica conversiones identificadas, distintas de los pesos de Moondream/NVIDIA:

| Conversión de Handy | Q8_0 | Q4_K_M | Motor documentado |
| --- | ---: | ---: | --- |
| [Ultra](https://huggingface.co/handy-computer/parakeet-ultra-gguf) | 740 MB | 486 MB | transcribe.cpp |
| [Canary 1B v2](https://huggingface.co/handy-computer/canary-1b-v2-gguf) | 1.144 GB | 735 MB | transcribe.cpp |

Estos repositorios incluyen licencia CC BY 4.0 y avisos de procedencia. Son
conversiones de una organización del runtime, no quants oficiales de los autores.
Su existencia no aprueba automáticamente su catálogo ni acredita calidad clínica.
No se han incorporado como fuentes de presets. Requerirían integrar y fijar
transcribe.cpp; sus GGUF no son intercambiables con los del NeMo-Speech.cpp actual.
Una integración compartida de ese motor podría cubrir Ultra y Canary, con sus
contratos de tiempos, audio largo y salida revisados por separado.

## Benchmarks relevantes y sus límites

### Español, reuniones, ruido y grabaciones largas

Moondream publica esta comparación con el mismo conjunto de audio y normalizador;
son mediciones del autor, no una evaluación clínica de Scribnic. Ultra y Redux usan
Photon; el original usa NeMo BF16. Valores WER en %, menor es mejor.

| Prueba | Parakeet v3 | Ultra | Redux |
| --- | ---: | ---: | ---: |
| FLEURS español, frases leídas | 3.12 | 2.72 | 3.71 |
| FLEURS español + ruido MUSAN, SNR 0 dB | 5.44 | 4.64 | 7.22 |
| AMI, reuniones en inglés | 10.86 | 9.77 | 10.80 |
| TED-LIUM 3, once charlas completas de 10–20 min, inglés | 2.71 | 1.94 | 2.51 |

Fuentes: [mediciones de Ultra](https://huggingface.co/moondream/parakeet-ultra#benchmarks)
y [mediciones de Redux](https://huggingface.co/moondream/parakeet-redux#benchmarks).
Redux publica unos 178 MB de pesos empaquetados: es una variante entrenada para
encoder ternario, no el resultado esperado de convertir cualquier Parakeet a dos
bits. Su menor tamaño no garantiza menor error, especialmente con ruido.

FLEURS usa frases de Wikipedia leídas; AMI evalúa reconocimiento en reuniones,
pero su WER no mide atribución de hablantes. TED-LIUM demuestra otra clase de
audio largo, sin reproducir consultas médicas españolas. La velocidad de Ultra
publicada con B200 y concurrencia 128 tampoco predice nuestra latencia en CPU.

### Conversaciones médicas

[Omi Health](https://github.com/Omi-Health/medical-STT-eval#leaderboard) publica
estas mediciones sobre 55 consultas simuladas PriMock57 en **inglés británico**,
actualizadas el 29 de abril de 2026. Son resultados de los evaluadores, con código
y transcripciones públicos; no son pesos de conversores. Valores en %, menor es
mejor. M-WER evalúa términos médicos y la última columna nombres de medicamentos.

| Modelo | WER general | M-WER médico | M-WER medicamentos |
| --- | ---: | ---: | ---: |
| Qwen3-ASR-0.6B | 9.83 | 4.95 | 13.7 |
| Parakeet TDT 0.6B v2 | 10.75 | 4.60 | 15.7 |
| Parakeet TDT 0.6B v3 | 9.35 | 5.90 | 20.6 |
| Canary 1B v2 | 14.32 | 9.40 | 18.0 |

v2 mejora ligeramente a Qwen0.6 en el agregado de términos médicos; Qwen mejora
en WER general y medicamentos. v2 sigue siendo solo inglés. Esta tabla no establece
un ganador médico en español ni incluye una medición comparable de Ultra, Redux,
MOSS actual o nuestro Nemotron 3.5. La fila del Nemotron-Speech inglés antiguo
corresponde a otro checkpoint. Los tiempos usan distintos equipos y no sirven para
ordenar nuestros recursos locales. Tampoco son mediciones de los quants propuestos.

## Tamaños de Whisper

Según el [repositorio de OpenAI](https://github.com/openai/whisper#available-models-and-languages):

| Variante | Parámetros |
| --- | ---: |
| tiny | 39M |
| base | 74M |
| small, nuestra referencia | 244M |
| medium | 769M |
| large | 1,550M |
| turbo | 809M |

Cambiar a un Whisper medium compatible con whisper.cpp sería principalmente añadir
presets y fijar sus archivos oficiales. Aumenta la capacidad y también los recursos;
la mejora en nuestra nota necesita medición. No inferir RAM solo de los parámetros:
dependen de pesos, dtype efectivo, contexto, buffers y runtime.

La distribución oficial de **whisper.cpp**, aceptada como fuente en este proyecto,
publica estas variantes multilingües. Son conversiones del proyecto del runtime,
no quants publicados por OpenAI. Metadatos de la revisión ya utilizada
`5359861c739e955e79d9a303bcbc70fb988958b1`:

| Variante | F16 con ciertos tensores F32 | Q8_0 | Q5_0 |
| --- | ---: | ---: | ---: |
| small actual | 488 MB | 264 MB | No publicado; hay Q5_1 |
| medium | 1.534 GB | 823 MB | 539 MB |
| large-v3-turbo | 1.625 GB | 874 MB | 574 MB |
| large-v3 | 3.095 GB | No publicado en esta revisión | 1.081 GB |

[Archivos y tamaños del distribuidor oficial](https://huggingface.co/ggerganov/whisper.cpp/tree/5359861c739e955e79d9a303bcbc70fb988958b1).
Los archivos son `ggml-medium-q8_0.bin`, `ggml-large-v3-turbo-q8_0.bin` y sus
variantes `q5_0`; para large-v3 está `ggml-large-v3-q5_0.bin`. No llevan `.en`.
El [descargador de whisper.cpp v1.9.5](https://github.com/ggml-org/whisper.cpp/blob/v1.9.5/models/download-ggml-model.sh)
ya contempla medium, turbo y esos formatos: no necesitan otro runtime en Scribnic.

Q8 en small ahorra unos 223 MB de archivo frente a F16. El número de parámetros
no determina por sí solo la degradación por cuantización. Comparar small F16 con
small Q8 aislaría ese efecto; comparar medium/turbo Q8 con small F16 cambia también
el modelo. Primero interesan medium Q8 y turbo Q8 como opciones de mayor capacidad,
manteniendo una referencia F16 si se quiere separar ambas causas.

## Audio directo a nota

| Modelo | Encaje y coste | Estado y licencia |
| --- | --- | --- |
| [LFM3-Audio-2B](https://huggingface.co/LiquidAI/LFM3-Audio-2B) | Español; 2B, GGUF Q4 y encoder/companion Q8 oficiales; prefill incremental | Implementado en [`lfm3_audio_note`](lfm3.md); LFM Open License 1.0, con umbral de ingresos para uso comercial gratuito |
| [MOSS-Audio-4B-Thinking](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-4B-Thinking) | Audio a texto general; unos 9.73 GiB de pesos BF16, más activaciones | Backend implementado, checkpoint completo sin ejecutar aquí; Apache-2.0; [detalle](moss.md) |
| [Gemma 4 E2B](https://huggingface.co/google/gemma-4-E2B-it) | 2.3B efectivos y unos 5.1B con embeddings; máximo 30 s de audio | Candidato sin workflow; requiere fragmentación y validación del protocolo de audio; Apache-2.0 |
| [Voxtral Mini 3B 2507](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507) | Español y resumen directo; unos 4.68B totales, no solo el decoder 3B | Candidato; Apache-2.0; adaptación de chat de audio, formato de prompt y conversión de respuesta |
| [Nemotron-Labs-Audex-2B](https://huggingface.co/nvidia/Nemotron-Labs-Audex-2B) | Audio general, inglés en su ficha | Descartar para la ruta comercial: NVIDIA OneWay Noncommercial |

Voxtral publica hasta 30 minutos para transcripción y 40 para comprensión, con
unos 9.5 GB de memoria GPU en BF16/FP16 en su ruta documentada. llama.cpp fijado
enumera Voxtral y Gemma 4, pero el adaptador WebSocket de LFM3 es específico:
no convierte estos candidatos en un cambio exclusivo de TOML. Voxtral tampoco
acepta el mismo uso de `system` de nuestra plantilla sin adaptación.
[Ficha de Voxtral](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507#key-features).

LFM2.5-Audio-1.5B es menor, pero su
[ficha oficial](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B) indica inglés;
no lo priorizamos frente a LFM3 para español. Probar primero el LFM3 ya integrado
evita añadir otro backend antes de resolver la calidad de la nota directa.

La evaluación local debe conservar el mismo audio, referencia, generador y prompt
cuando compare ASR. Para comparar generadores, reutilizar exactamente el mismo
transcript. Registrar cifras, dosis, negaciones, seguimiento y atribución, además
de tiempo de carga/inferencia y memoria por proceso. Mantener fijos esos factores
permite atribuir la diferencia a un modelo o quant concreto.
