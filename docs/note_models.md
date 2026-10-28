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
| Qwen3.5-0.8B | Q4_K_M | 0.58 GB | `qwen35_08b_q4_k_m_note` |
| Qwen3.5-0.8B | Q8_0 | 0.84 GB | `qwen35_08b_q8_0_note` |
| Qwen3.5-2B | Q4_K_M | 1.40 GB | `qwen35_2b_q4_k_m_note` |
| Qwen3.5-2B | Q8_0 | 2.08 GB | `qwen35_2b_q8_0_note` |
| Qwen3.5-4B | Q4_K_M | 3.01 GB | `qwen35_4b_q4_k_m_note` |
| Qwen3.5-4B | Q8_0 | 4.62 GB | `qwen35_4b_q8_0_note` |
| Bonsai 2 4B | PQ2_0 | 1.15 GB | `bonsai_4b_note` |
| LFM3-1.7B, texto | Q4_K_M | 1.10 GB | `lfm3_17b_q4_k_m_note` |
| LFM3-1.7B, texto | Q8_0 | 1.85 GB | `lfm3_17b_q8_0_note` |
| LFM3-1.7B, texto, QAD | Q4_0 | 1.05 GB | `lfm3_17b_qad_q4_0_note` |

QAD identifica un checkpoint entrenado para compensar la cuantización, almacenado
en formato Q4_0. Su nombre, archivo y revisión se conservan separados del Q4_K_M
ordinario. Los formatos PQ2_0 y Q4_K_M tampoco representan el mismo tipo de pesos.

Las nuevas variantes fijan repositorio, revisión y SHA256 del archivo. Las
fuentes elegidas son:

- Qwen: [0.8B](https://huggingface.co/bartowski/Qwen_Qwen3.5-0.8B-GGUF),
  [2B](https://huggingface.co/bartowski/Qwen_Qwen3.5-2B-GGUF) y
  [4B](https://huggingface.co/bartowski/Qwen_Qwen3.5-4B-GGUF), conversiones Bartowski
  del mismo distribuidor para mantener consistencia entre formatos y tamaños.
- [Bonsai 2 4B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-4B-gguf), PQ2_0,
  usando los kernels del fork Prism existente.
- [LFM3-1.7B GGUF](https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF), publicado por
  Liquid. Es el generador de texto; el workflow de audio directo sigue siendo
  [`lfm3_audio_note`](lfm3.md).

Qwen y Bonsai declaran Apache-2.0. LFM3 usa
[LFM Open License v1.0](https://www.liquid.ai/lfm-license), con su umbral de ingresos
anuales para el uso comercial gratuito. Cada modelo conserva su propia licencia.

## Ejecutar desde texto o un transcript guardado

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

En LFM3 conviene copiar también la tabla `server_options` de su workflow: incluye
un presupuesto de razonamiento cero y conserva explícitamente las opciones del
runtime. Los overrides de tablas sustituyen la tabla completa.

## Condiciones de comparación y validación

Los nuevos workflows comparten la plantilla `soap_es`, temperatura cero, seed 42,
thinking desactivado, máximo de 768 tokens de salida y límite de 12,000 caracteres
de entrada. Fijan 8,192 posiciones de contexto, como la referencia Bonsai. LFM3
añade `reasoning-budget = 0` en llama-server. El backend existente rechaza notas
vacías, truncadas o que incluyan etiquetas de pensamiento.

Las variantes de una misma familia mantienen sus ajustes de generación. Qwen y
Bonsai conservan su perfil de penalizaciones; LFM3 usa el suyo. Para comparar
familias hay que registrar esos ajustes además del prompt, input y cuantización.
La configuración efectiva ya se guarda en cada ejecución.

Esta incorporación valida resolución de presets y contratos sin descargar ni
ejecutar los nuevos checkpoints. La calidad, memoria y tiempos reales quedan
pendientes de evaluación. La existencia de un preset tampoco acredita validación
en CUDA, Metal o CPU. Para elegir, comparar la conservación de cifras, dosis,
intervalos, negaciones, seguimiento y atribución de hechos junto con los recursos.
