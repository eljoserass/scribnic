# Modelos de texto para comparar generación de notas

Los presets usan `openai_chat` y servidores locales `llama-server`. Qwen3.5 y
LFM3 de texto seleccionan `llama_cpp` upstream b11946; Bonsai selecciona
`llama_cpp_prism`, con el release Prism ya fijado. No se añaden bibliotecas
Python de modelos ni otros runtimes.

Cada workflow acepta una transcripción plana o diarizada y genera una nota SOAP
en español. Esto permite evaluar varios generadores sobre la misma entrada,
incluido texto de referencia, sin repetir reconocimiento de audio.

## Variantes disponibles

Los tamaños son los archivos de pesos publicados, en GB decimales. La memoria
de ejecución añade caché, activaciones y buffers del runtime. Solo se prepara el
modelo del workflow elegido; registrar todos los presets no descarga todos los
pesos.

| Modelo | Formato | Pesos | Workflow |
| --- | --- | ---: | --- |
| Bonsai 2 2B, referencia existente | PQ2_0 | 0.52 GB | `bonsai_note` |
| Qwen3.5-0.8B | Q4_K_M | 0.53 GB | `qwen35_08b_q4_k_m_note` |
| Qwen3.5-0.8B | Q8_0 | 0.81 GB | `qwen35_08b_q8_0_note` |
| Qwen3.5-2B | Q4_K_M | 1.28 GB | `qwen35_2b_q4_k_m_note` |
| Qwen3.5-2B | Q8_0 | 2.01 GB | `qwen35_2b_q8_0_note` |
| Qwen3.5-4B | Q4_K_M | 2.74 GB | `qwen35_4b_q4_k_m_note` |
| Qwen3.5-4B | Q8_0 | 4.48 GB | `qwen35_4b_q8_0_note` |
| Bonsai 2 4B | PQ2_0 | 1.15 GB | `bonsai_4b_note` |
| LFM3-1.7B, texto | Q4_K_M | 1.10 GB | `lfm3_17b_q4_k_m_note` |
| LFM3-1.7B, texto | Q8_0 | 1.85 GB | `lfm3_17b_q8_0_note` |
| LFM3-1.7B, texto, QAD | Q4_0 | 1.05 GB | `lfm3_17b_qad_q4_0_note` |

QAD identifica un checkpoint entrenado para compensar la cuantización, almacenado
en formato Q4_0. Su nombre, archivo y revisión se conservan separados del Q4_K_M
ordinario. Los formatos PQ2_0 y Q4_K_M tampoco representan el mismo tipo de pesos.

Las nuevas variantes fijan repositorio, revisión y SHA256 del archivo. Las
fuentes elegidas son:

- Qwen: [0.8B](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF),
  [2B](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF) y
  [4B](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF), conversiones publicadas por
  la organización Unsloth AI a partir de los modelos de Qwen. No son publicaciones
  de la organización Qwen ni acreditan validación clínica.
- [Bonsai 2 4B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-4B-gguf), PQ2_0,
  usando los kernels del fork Prism existente.
- [LFM3-1.7B GGUF](https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF), publicado por
  Liquid. Es el generador de texto; el workflow de audio directo sigue siendo
  [`lfm3_audio_note`](lfm3.md).

La [política de procedencia](model_sources.md) excluye conversiones de cuentas
personales, con la excepción explícita del repositorio oficial de whisper.cpp.
La sustitución de las seis variantes Qwen conserva sus nombres y workflows;
actualiza repositorios, revisiones, archivos y hashes. Las siguientes ejecuciones
preparan los archivos de Unsloth; los pesos antiguos que haya en caché no se borran.

Qwen y Bonsai declaran Apache-2.0. LFM3 usa
[LFM Open License v1.0](https://www.liquid.ai/lfm-license), con su umbral de ingresos
anuales para el uso comercial gratuito. Cada modelo conserva su propia licencia.

## Ejecutar desde texto o un transcript guardado

Para ejecutar Whisper small Q8 con VAD y Qwen3.5-0.8B Q8 en un solo comando:

```bash
uv run scribnic run consulta.wav --workflow whisper_qwen35_08b_q8_0 \
  --output-dir runs/whisper-qwen08-q8
```

El informe queda en `runs/whisper-qwen08-q8/report.html`. Este workflow también
acepta texto o una transcripción guardada: en ese caso solo ejecuta el paso de nota.

```bash
# Descargar y verificar una sola variante, sin cargarla en memoria.
uv run scribnic setup --workflow qwen35_08b_q4_k_m_note

# Generar desde texto plano; el report.html queda en el directorio indicado.
uv run scribnic run conversacion.txt \
  --workflow qwen35_08b_q4_k_m_note --output-dir runs/qwen08-q4

# Reutilizar exactamente el mismo texto con otra cuantización.
uv run scribnic run conversacion.txt \
  --workflow qwen35_08b_q8_0_note --output-dir runs/qwen08-q8

uv run scribnic run conversacion.txt \
  --workflow bonsai_4b_note --output-dir runs/bonsai4

uv run scribnic run conversacion.txt \
  --workflow lfm3_17b_qad_q4_0_note --output-dir runs/lfm3-text-qad
```

El texto puede sustituirse por un artefacto `transcript.json` o
`diarized_transcript.json`. Por ejemplo, reconocer el audio una vez y reutilizar
el resultado:

```bash
uv run scribnic run consulta.wav --workflow whisper_bonsai_vad \
  --output transcript --output-dir runs/shared-transcript

uv run scribnic run runs/shared-transcript/transcript.json \
  --workflow qwen35_08b_q4_k_m_note --output-dir runs/qwen08-from-whisper
```

Para insertar cualquiera de estos generadores en otro pipeline, selecciona en el
paso de nota el modelo y runtime de su workflow. Conserva los pasos de voz de ese
pipeline. Ejemplo del paso Qwen:

```toml
[steps.note]
backend = "openai_chat"
model = "Qwen3.5-0.8B-Q4_K_M"
runtime = "llama_cpp"
template = "soap_es"
source = "speech"

[steps.note.model_settings]
temperature = 0.0

[steps.note.runtime_settings]
context_size = 8192
```

En LFM3 hay que conservar juntos `thinking = true` del workflow y su tabla
`server_options`, con presupuesto de razonamiento cero. La plantilla deja abierto
el bloque de protocolo y el servidor lo cierra inmediatamente, sin tokens libres
de razonamiento. Prefijarlo cerrado produjo un `</think>` huérfano con b11946.
Los overrides de tablas sustituyen la tabla completa. El cliente sigue rechazando
etiquetas de pensamiento en el contenido final; no las elimina silenciosamente.

## Condiciones de comparación y validación

Los workflows comparten la plantilla `soap_es`, seed 42, máximo de 768 tokens de
salida y límite de 12,000 caracteres de entrada. Fijan 8,192 posiciones de contexto.
Qwen conserva temperatura cero en sus workflows de comparación. Bonsai 4B usa
el perfil sin pensamiento de Prism: temperatura 0.7, top-p 0.8, top-k 20, min-p 0,
presence penalty 1.5 y repetición 1.0 sobre una ventana de 64 tokens.
[Perfil oficial de Prism](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/SMALL-MODELS.md).

LFM3 usa temperatura 0.1, top-k 50 y repetición 1.1, según el perfil general del
autor. Mantiene top-p 1 y min-p 0 como filtros neutros. Su presupuesto cero controla
la fase de razonamiento, separado del prefijo de la plantilla. El backend rechaza
notas vacías, truncadas o con etiquetas de pensamiento.
[Parámetros del modelo](https://huggingface.co/LiquidAI/LFM3-1.7B#inference).

Las variantes de una misma familia mantienen sus ajustes de generación. Qwen y
Bonsai conservan su perfil de penalizaciones; LFM3 usa el suyo. Para comparar
familias hay que registrar esos ajustes además del prompt, input y cuantización.
La configuración efectiva ya se guarda en cada ejecución.

Se ejecutaron Bonsai 4B PQ2_0 y LFM3-1.7B Q8_0 sobre la misma referencia textual,
con Bonsai 2B como control, sin ASR y usando esos perfiles. El detalle está en la
[evaluación de texto](text_note_evaluation.md). Las otras variantes nuevas siguen
pendientes de inferencia real. Esa prueba local no valida otras plataformas ni
establece calidad clínica general. Para elegir, comparar cifras, dosis, intervalos,
negaciones, seguimiento y atribución de hechos junto con los recursos.
