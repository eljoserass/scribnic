# Procedencia de modelos y cuantizaciones

Revisión de fuentes: 28 de octubre de 2026. Esta es la política de selección de
los presets incluidos en Scribnic, acordada para este proyecto.

## Fuentes aceptadas

Preferir pesos publicados por la organización autora del modelo. Para conversiones,
aceptar organizaciones identificables y documentar quién entrenó el modelo y quién
publicó la conversión. No incorporar cuantizaciones de cuentas personales.

Hay una excepción explícita: los pesos del repositorio oficial
[`ggerganov/whisper.cpp`](https://huggingface.co/ggerganov/whisper.cpp), mantenido
en la cuenta del creador del proyecto. Esa excepción no se extiende a otras cuentas.

Que una cuenta sea una organización no certifica la calidad de sus pesos. Fijar
revisión y SHA256 identifica una descarga reproducible; tampoco constituye una
validación del modelo, su cuantización o su calidad clínica. Un preset configura
una ejecución posible; las plataformas y resultados medidos se documentan aparte.

Si no hay una fuente aceptable para un quant, conservar el checkpoint original
cuando el backend lo soporte, o dejar pendiente esa variante. No sustituirlo
silenciosamente por otra conversión. Para introducir una nueva organización hay
que revisar su identidad, relación con el modelo, archivos, licencia y runtime.

Por decisión del proyecto, Handy y transcribe.cpp quedan fuera del plan de
integración. Para Ultra y Redux se contempla una futura rama con Photon y los
pesos originales de Moondream. Las adiciones de presets nativos se mantienen
separadas de esas integraciones.

## Estado del catálogo

| Familia | Publicador de los pesos seleccionados | Decisión |
| --- | --- | --- |
| Qwen3.5 0.8B, 2B y 4B, Q4_K_M / Q8_0 | [Unsloth AI](https://huggingface.co/unsloth) | Seis conversiones de organización, a partir de Qwen; sustituyen las de la cuenta Bartowski |
| Bonsai 2 2B / 4B, PQ2_0 | [Prism ML](https://huggingface.co/prism-ml) | Pesos de la organización autora; conservar |
| LFM3-1.7B y LFM3-Audio-2B, incluidos QAD y mmproj | [LiquidAI](https://huggingface.co/LiquidAI) | Pesos de la organización autora; conservar |
| Nemotron 3.5 ASR, Nemotron 3 Diarization y Parakeet TDT v3, Q8_0 | [NVIDIA](https://huggingface.co/nvidia) | Pesos de la organización autora; conservar |
| Whisper tiny, base, small, medium y turbo Q8_0; large-v3 Q5_0 | Repositorio oficial de whisper.cpp | Distribución del runtime como excepción aprobada; ver [presets](asr_presets.md) |
| Silero VAD, distribución de whisper.cpp | [ggml-org/whisper-vad](https://huggingface.co/ggml-org/whisper-vad) | Organización del runtime; conservar |
| MOSS-Transcribe-Diarize y MOSS-Audio-4B-Thinking | [OpenMOSS-Team](https://huggingface.co/OpenMOSS-Team) | Checkpoints originales; sin añadir un quant de terceros |

En la revisión no se encontraron GGUF Q4/Q8 publicados por Qwen para sus variantes
Qwen3.5 de 0.8B, 2B y 4B. Unsloth publica ambos formatos para las tres escalas.
Los presets fijan cada revisión y el SHA256 anunciado por Hugging Face para cada
archivo. Las referencias y tamaños están en el [catálogo de notas](note_models.md).

La migración cambia la fuente de descarga, sin cambiar nombres de modelos ni
workflows. No se descargan todos los modelos al registrar presets. Las revisiones
anteriores en caché y sus informes históricos se conservan; una comparación futura
debe distinguir los archivos de ambos publicadores, aunque compartan etiqueta Q4/Q8.

## Cuantización propia

Crear un quant local a partir de un checkpoint oficial es un experimento propio:
no pasa a ser un quant oficial del autor. Hay que conservar revisión de origen,
herramienta y versión, método, módulos convertidos, dtype de cómputo, archivos y
resultados de evaluación antes de añadirlo como preset. No requiere publicar los
pesos ni modificar la instalación base del cliente.

La documentación de [MOSS](moss.md) separa ese trabajo pendiente de probar el
checkpoint BF16 existente. La cuantización propia todavía no está implementada.

La revisión de [Unsloth, Whisper y otros publicadores](model_publishers.md) detalla
formatos y licencias. Compartir distribuidor no implica compartir runtime.
