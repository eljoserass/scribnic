# Unsloth, Whisper y otros publicadores

Fuentes revisadas el 28 de octubre de 2026. Esta investigación no añade modelos,
dependencias ni runtimes. La selección de pesos sigue la
[política de procedencia](model_sources.md).

## Qué ofrece Unsloth

Unsloth desarrolla herramientas para ejecutar, ajustar y entrenar modelos,
incluidos fine-tuning y aprendizaje por refuerzo, además de publicar pesos
convertidos y cuantizados. Su biblioteca, sus interfaces Studio/Desktop y sus
archivos de modelos son productos distintos. Consumir uno de sus GGUF en nuestro
`llama-server` no requiere instalar Unsloth ni usar su interfaz.
[Documentación oficial](https://unsloth.ai/docs).

Su catálogo distingue GGUF para motores como llama.cpp y safetensors para sus
rutas de inferencia o entrenamiento. Algunos safetensors están cuantizados con
bitsandbytes; otros contienen pesos en coma flotante. El nombre del publicador o
la extensión `safetensors` no identifica la precisión.
[Catálogo oficial](https://unsloth.ai/docs/get-started/unsloth-model-catalog).

Unsloth Dynamic 2.0 combina precisiones por capa y usa calibración para decidir
dónde reducirlas. No equivale a entrenar un modelo nuevo, ni a que todos los pesos
usen el mismo número de bits. Sus resultados publicados en MMLU y divergencia KL
no validan por sí solos notas clínicas en español.
[Descripción del método](https://unsloth.ai/blog/dynamic-v2).

Los Qwen3.5 seleccionados en Scribnic identifican archivos concretos `Q4_K_M` y
`Q8_0`. No se debe atribuirles automáticamente otra variante Dynamic, un formato
bitsandbytes o resultados medidos en checkpoints distintos.
[Presets y condiciones de comparación](note_models.md).

## Whisper: sí hay dos distribuciones diferentes

La organización principal [`unsloth`](https://huggingface.co/unsloth) publica
Whisper tiny, base, small, large-v3 y large-v3-turbo en safetensors. Los repositorios
revisados no ofrecen archivos Whisper Q4/Q8 para nuestro runtime whisper.cpp.
Por ejemplo, [`whisper-small`](https://huggingface.co/unsloth/whisper-small/tree/023f354567bbe9a875605cb8e7ef40ff9629653a)
y [`large-v3-turbo`](https://huggingface.co/unsloth/whisper-large-v3-turbo/tree/d01a3178cc804b807fed7b3e17531769c47e9665).

También existe la organización secundaria
[`unslothai`](https://huggingface.co/unslothai), denominada Unsloth Backup Account.
Su relación se puede comprobar desde el código oficial de Unsloth: el catálogo de
dictado enumera expresamente esos repositorios y los archivos `.bin`, en el
[commit `4b71ee393d12addf7ee1bc5952273ca05f404d88`](https://github.com/unslothai/unsloth/blob/4b71ee393d12addf7ee1bc5952273ca05f404d88/studio/backend/core/inference/stt_ggml_sidecar.py#L88-L101).
El nombre de una cuenta o una insignia, por sí solos, no habrían acreditado ese
vínculo.

| Distribución secundaria | Archivo | Tamaño decimal | Precisión publicada |
| --- | --- | ---: | --- |
| [tiny](https://huggingface.co/unslothai/whisper-tiny-GGUF) | `whisper-tiny.bin` | 77.7 MB | F16, con ciertos tensores F32 |
| [base](https://huggingface.co/unslothai/whisper-base-GGUF) | `whisper-base.bin` | 148 MB | F16, con ciertos tensores F32 |
| [small](https://huggingface.co/unslothai/whisper-small-GGUF) | `whisper-small.bin` | 488 MB | F16, con ciertos tensores F32 |
| [large-v3-turbo](https://huggingface.co/unslothai/whisper-large-v3-turbo-GGUF) | `whisper-large-v3-turbo.bin` | 1.62 GB | F16, con ciertos tensores F32 |
| [large-v3](https://huggingface.co/unslothai/whisper-large-v3-GGUF) | `whisper-large-v3.bin` | 3.10 GB | F16, con ciertos tensores F32 |

Aunque los repositorios se llamen `-GGUF`, los archivos son GGML `.bin` de
whisper.cpp. Sus fichas especifican que no se aplicó cuantización de pocos bits.
Son candidatos compatibles en formato con nuestro `whisper_cpp`, no con
`llama_cpp`; un preset nuevo necesitaría fijar revisión y hash y ejecutar una
prueba con nuestra versión de whisper-server. No se han descargado ni probado
estos archivos en Scribnic.

Por consistencia de publicador se podría comparar Whisper small F16 de Unsloth
con nuestro small Q8, manteniendo el mismo pipeline. Cambiaría la precisión y
el tamaño de pesos: no sería una sustitución equivalente del archivo Q8. Se
conserva el Q8 oficial de whisper.cpp, cuya procedencia ya está aprobada.

## Las licencias se revisan por archivo y repositorio

OpenAI publica el código y los pesos originales de Whisper bajo MIT.
[Declaración y licencia originales](https://github.com/openai/whisper#license).
Las fichas de `unsloth/whisper-small` y `unsloth/whisper-large-v3` indican Apache-2.0;
la de [`unsloth/whisper-large-v3-turbo`](https://huggingface.co/unsloth/whisper-large-v3-turbo)
indica MIT. Los cinco repositorios secundarios `unslothai/whisper-*-GGUF` indican
Apache-2.0.

En las revisiones consultadas, esos repositorios secundarios contienen solamente
`.gitattributes`, `README.md` y el modelo `.bin`: no incluyen un archivo `LICENSE`
independiente. Apache-2.0 es la declaración de su ficha; no resuelve por sí sola la
trazabilidad de los avisos MIT del original. Antes de distribuir esos archivos
como parte de un producto habría que conservar y comprobar los avisos de origen.
No se presenta esa discrepancia de metadatos como una licencia nueva universal de
Whisper.

Los Qwen de este catálogo declaran Apache-2.0; LFM mantiene su propia licencia y
sus condiciones. La licencia de las herramientas Unsloth es otra cuestión: no se
aplica automáticamente a los pesos convertidos que descargamos.
[Licencias del catálogo de notas](note_models.md).

## Otras organizaciones relevantes

| Organización | Qué publica | Encaje con Scribnic |
| --- | --- | --- |
| [ggml-org](https://huggingface.co/ggml-org) | Organización oficial de ggml y sus proyectos; conversiones GGUF y herramientas | Preferible cuando publica el modelo y formato necesarios; revisar arquitectura y release del runtime |
| [LM Studio Community](https://huggingface.co/lmstudio-community) | Conversiones GGUF y MLX; sus repositorios figuran en la [documentación de LM Studio](https://lmstudio.ai/docs/app/modelyaml) | GGUF puede usar nuestro llama-server si la arquitectura está soportada; no obliga a instalar LM Studio |
| [MLX Community](https://huggingface.co/mlx-community) | Conversiones y quants para MLX, enlazados desde el [proyecto MLX LM](https://github.com/ml-explore/mlx-lm#quantization) | Ruta para Apple Silicon; no hay backend local MLX en Scribnic y sus pesos no son GGUF |
| Autores como [LiquidAI](https://huggingface.co/LiquidAI), [Prism ML](https://huggingface.co/prism-ml) y [NVIDIA](https://huggingface.co/nvidia) | Checkpoints propios, algunos ya cuantizados | Priorizar sus publicaciones cuando existan; varios presets actuales ya las usan |

Que existan estas organizaciones no aprueba automáticamente todo su catálogo.
Cada incorporación identifica autor original, conversor, licencia, precisión y
archivo verificable, con una prueba del modelo y runtime elegidos.

## Compatibilidad de formatos en los runtimes actuales

| Pesos | Ruta actual | Trabajo para usarlos |
| --- | --- | --- |
| Qwen3.5 GGUF Q4/Q8 de Unsloth | `openai_chat` + `llama_cpp` | Presets existentes; no instalar Unsloth |
| Whisper GGML `.bin` F16 de Unsloth | `openai_transcription` + `whisper_cpp` | Preset y workflow, seguidos de prueba del archivo exacto |
| Whisper safetensors de Unsloth | Ninguna integración local Whisper/Transformers | Añadir cargador y backend ASR; el backend Transformers de MOSS es específico de MOSS |
| Safetensors cuantizados con bitsandbytes | No son pesos para llama-server | Cargador compatible con ese método, dependencias opcionales y plataforma validada |
| Pesos MLX cuantizados | Ningún runtime local MLX | Integración MLX o servicio externo compatible, con su contrato de salida |

Un servidor externo de notas con chat HTTP compatible puede reutilizar
`openai_chat`. Eso no establece soporte de audio, timestamps o diarización. Un
contenedor GGUF tampoco convierte Whisper, MOSS y un generador de notas en la
misma arquitectura ni en el mismo protocolo.
