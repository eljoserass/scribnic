# Presets de voz sobre los runtimes existentes

Esta entrega añade únicamente descriptores de modelos y workflows. Reutiliza
`openai_transcription`, whisper.cpp v1.9.5, NeMo-Speech.cpp v0.2.0 y el generador
Bonsai 2 2B PQ2_0 sobre Prism. No añade paquetes Python, backends ni runtimes.

## Modelos y workflows

Todos los Whisper son multilingües, sin sufijo `.en`, y usan VAD Silero, español,
beam 5 y timestamps. Comparten los ajustes del small existente. Los tamaños son
archivos de pesos en unidades decimales, no RAM durante inferencia.

| ASR | Parámetros | Pesos | Workflow |
| --- | ---: | ---: | --- |
| Whisper tiny Q8_0 | 39M | 43.5 MB | `whisper_tiny_q8_0_bonsai` |
| Whisper base Q8_0 | 74M | 81.8 MB | `whisper_base_q8_0_bonsai` |
| Whisper small Q8_0, modelo ya existente | 244M | 264 MB | `whisper_small_q8_0_bonsai` |
| Whisper medium Q8_0 | 769M | 823 MB | `whisper_medium_q8_0_bonsai` |
| Whisper large-v3-turbo Q8_0 | 809M | 874 MB | `whisper_turbo_q8_0_bonsai` |
| Whisper large-v3 Q5_0 | 1,550M | 1.081 GB | `whisper_large_v3_q5_0_bonsai` |
| Parakeet TDT 0.6B v3 Q8_0 | 600M | 714 MB | `parakeet_v3_q8_0_bonsai` |
| El mismo Parakeet con Nemotron 3 Diarization Q8_0 | ASR + diarizador | 714 MB + companion | `parakeet_v3_q8_0_diarized_bonsai` |

`whisper_bonsai`, `whisper_bonsai_vad` y `whisper_qwen35_08b_q8_0` conservan sus
nombres y configuración. El nuevo workflow small reutiliza el preset de pesos
existente; no crea otra copia del modelo.

Whisper procede de la [distribución oficial de whisper.cpp](https://huggingface.co/ggerganov/whisper.cpp/tree/5359861c739e955e79d9a303bcbc70fb988958b1),
con licencia MIT. En esa revisión no hay `ggml-large-v3-q8_0.bin`: se incorpora
large-v3 Q5_0, sin sustituirlo por large-v2 ni una conversión de terceros.
El Q8 de turbo corresponde a otro checkpoint y conserva su identidad completa.

Parakeet procede del [repositorio oficial de NVIDIA](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/tree/541d1f99c6b0c3cd0b11a95167540bb8edefd82b),
con licencia CC BY 4.0. Todos los presets nuevos fijan repositorio, revisión,
archivo, formato y SHA256. Solo se preparan los recursos del workflow elegido.

## Ejecutar y comparar

```bash
uv run scribnic run consulta.wav --workflow whisper_tiny_q8_0_bonsai \
  --output-dir runs/whisper-tiny

uv run scribnic run consulta.wav --workflow whisper_turbo_q8_0_bonsai \
  --output-dir runs/whisper-turbo

uv run scribnic run consulta.wav --workflow whisper_large_v3_q5_0_bonsai \
  --output-dir runs/whisper-large-v3

uv run scribnic run consulta.wav --workflow parakeet_v3_q8_0_bonsai \
  --output-dir runs/parakeet-v3

uv run scribnic run consulta.wav --workflow parakeet_v3_q8_0_diarized_bonsai \
  --output-dir runs/parakeet-v3-diarized
```

El resultado predeterminado es una nota; cada directorio contiene `report.html`,
configuración efectiva, transcripción, nota y métricas. `setup --workflow NOMBRE`
descarga y verifica sin cargar el modelo. `--offline` exige recursos ya en caché.

Para detenerse antes de generar la nota, añadir `--output transcript`, o
`--output diarized_transcript` al workflow diarizado. Los ocho workflows aceptan
también texto o transcripciones suministradas y ejecutan solo la nota. La sesión
todavía prepara los recursos configurados al abrirse; el bypass es de inferencia.

Parakeet detecta el idioma; `es-ES` se conserva para el postproceso compatible y
los metadatos, sin convertirse en un prompt que fuerce el idioma del ASR.
Solo se publican workflows Parakeet offline. Su variante diarizada reutiliza el
companion y la atribución por tiempos de NeMo; no implementa diarización propia.
La ruta ASR plain no carga ese companion.

Para comparar generadores de notas, reconocer una vez y pasar el mismo
`transcript.json` a los [workflows de texto](note_models.md). Para cambiar el
generador de estos pipelines, copiar el workflow y sustituir su paso `note` por
el del generador elegido. No hace falta duplicar todas las combinaciones.

## Alcance de la validación

La prueba local usa los mismos primeros 15 segundos de `data/examples/dialogue8.wav`,
CPU Linux x86-64 en WSL, seis hilos y una ejecución por workflow, con modelos ya
descargados. Revisa carga real, HTTP, tiempos, texto, companion y generación de nota.
Los informes generados se guardan fuera del código en
`runs/native-asr-presets-20261028/`; su `report.html` principal reúne los ocho
workflows, los tiempos y RSS del proceso ASR, transcripciones, notas y enlaces a
los informes individuales.

Es una prueba de integración corta, sin warmup ni repeticiones estadísticas.
No establece un ganador clínico, calidad en consultas completas ni equivalencia
entre quants. La nota Bonsai de algunas pasadas añade afirmaciones que el fragmento
no contiene: que termine la ejecución no valida esos contenidos. Esa revisión
de calidad sigue separada de incorporar los presets.

## Integraciones posteriores

Qwen3-ASR requiere adaptar entrada y salida de audio sobre llama-server. Ultra y
Redux se estudiarían juntos en una rama Photon con los pesos de Moondream.
Canary necesita soporte de su decoder o backend oficial. Estas rutas quedan para
ramas posteriores; la [selección automática de hardware](hardware_selection.md)
también sigue siendo una propuesta. Handy y transcribe.cpp quedan fuera.
