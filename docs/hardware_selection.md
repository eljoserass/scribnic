# Mapa propuesto de selección de hardware y cómputo

Propuesta del 28 de octubre de 2026. Este documento describe cambios posibles;
no implementa detección nueva ni cambia la selección automática actual.

## Qué tenemos y qué falta

Scribnic ya selecciona binarios por sistema/arquitectura en `downloads.py` y
comprueba dispositivos Torch dentro de los backends opcionales. Las decisiones
de dtype y aceleración están repartidas y usan reglas conservadoras:

| Ruta actual | Decisión automática actual | Limitación |
| --- | --- | --- |
| MOSS-Transcribe-Diarize | CUDA si está disponible; BF16 en CUDA, FP32 en CPU | No comprueba BF16 de CPU; CUDA disponible tampoco demuestra BF16 nativo |
| MOSS-Audio | CUDA, luego MPS, luego CPU; BF16/FP16/FP32 respectivamente | Mismo problema en CPU y falta comprobar dtype por dispositivo/modelo |
| llama-server upstream y Prism | Descarga fijada por plataforma; `gpu_layers=auto` pide todas en Darwin y cero en otros sistemas | El sistema operativo no prueba capacidades de la build ni memoria suficiente |
| whisper-server | `use_gpu=auto` habilita GPU por defecto solo en Darwin | Falta comprobar backend compilado, dispositivo utilizable y selección efectiva |
| NeMo-Speech.cpp | `device=cpu` si no se configura otra cosa | Falta consultar capacidades de esa distribución |
| Endpoint externo | Conecta al servicio indicado | El cliente no conoce ni posee su hardware |

La prueba [MOSS CPU BF16 frente a FP32](moss.md#pesos-bf16-cómputo-en-cpu-y-cuantización)
demuestra el caso que interesa resolver: esta CPU admite BF16 y la ruta del modelo
funcionó, pero `auto` eligió FP32. No implica que cualquier CPU o backend mejore.

## Flujo de decisión propuesto

1. **Leer la petición.** Separar dispositivo/dtype explícitos de `auto`. Conservar
   la identidad del modelo, quant y revisión seleccionados. Elegir BF16 de cómputo
   no significa cambiar un modelo Q8 a Q4 ni descargar otros pesos.
2. **Consultar lo visible desde el proceso.** Sistema, arquitectura, afinidad y
   límites del contenedor/VM, memoria disponible y capacidades expuestas. Una GPU
   física del host puede no estar disponible en WSL o para la biblioteca instalada.
3. **Consultar el runtime seleccionado.** Torch CPU, CUDA/ROCm y MPS; o capacidades
   de la build nativa fijada y dispositivos que esa build puede utilizar. Consultar
   el mecanismo de cada release, sin asumir un `--list-devices` común a todos.
4. **Cruzar capacidades y soporte del modelo.** Hardware con BF16, kernels
   disponibles y una ruta del modelo compatible son condiciones distintas.
   Seleccionar solo combinaciones respaldadas por esa evidencia.
5. **Preparar y registrar la ejecución.** Validar la petición explícita antes de
   cargar pesos; leer del runtime el dispositivo y cómputo efectivos cuando los
   exponga. Registrar diferencias con lo solicitado y su motivo.

Una petición explícita incompatible debe producir un error claro. Para `auto`,
una capacidad desconocida permite una elección conservadora con motivo registrado.
Un error arbitrario durante inferencia no justifica cambiar de dtype y repetir
silenciosamente una petición. Los endpoints externos deben indicar `unknown` cuando
el servidor no exponga datos, sin inspeccionar hardware local como sustituto.

## Señales útiles por backend

| Ámbito | Señales y criterio propuesto |
| --- | --- |
| Torch en CPU | API pública `torch.cpu.get_capabilities()` cuando exista; BF16 específico, no solo el nivel AVX512. Disponibilidad de oneDNN y prueba funcional opcional; en versiones sin esa API, capacidad desconocida o consulta de plataforma explícita |
| CUDA/ROCm | Build instalada, disponibilidad, dispositivo, memoria y soporte del dtype; consultar `torch.cuda.is_bf16_supported(including_emulation=False)` cuando lo permita esa versión, distinguiendo cómputo nativo y emulado |
| Apple MPS | Disponibilidad real de MPS y soporte de operaciones/dtype en la combinación de macOS, Torch y modelo; evitar trasladar automáticamente reglas CUDA |
| Motores GGML | Distribución/release, backends compilados, dispositivos y offload efectivo que informe el proceso; conservar el quant de los tensores y distinguir dtype de KV cache y operaciones |
| Memoria | Estimación inferior de pesos, más reservas para contexto, audio y buffers; registrar que compartir memoria, mmap y RSS no equivalen al tamaño del archivo |

En esta máquina, la API pública de Torch confirma `avx512_bf16=True`. También
informa `avx512_fp16=False`: ni siquiera las capacidades de BF16 y FP16 se deben
tratar como equivalentes. La build `2.14.1+cpu` no ofrece CUDA/ROCm aunque el chip
incluya una Radeon; otros motores nativos pueden tener capacidades diferentes.
No se ha validado aquí esa GPU con los runtimes seleccionados.

La consulta de capacidades no demuestra que BF16 sea más rápido para toda la
carga. Proponer una prueba corta opcional en un futuro comando de diagnóstico:
sin descargar modelos y sin convertir el inicio normal de cada sesión en un
benchmark. La prueba real de un modelo seguiría siendo opt-in y separada de las
pruebas unitarias. Si se guardan resultados de diagnóstico, invalidarlos al cambiar
hardware visible, sistema, versión de biblioteca, build o modelo pertinente.

Referencias de las APIs públicas:
[capacidades CPU](https://docs.pytorch.org/docs/2.14/generated/torch.cpu.get_capabilities.html),
[soporte BF16 en CUDA/ROCm](https://docs.pytorch.org/docs/2.14/generated/torch.cuda.is_bf16_supported.html).

## Archivos que tocaría una implementación futura

| Archivo | Responsabilidad concreta |
| --- | --- |
| Nuevo `src/scribnic/runtimes/compute.py` | Dataclasses pequeñas de capacidades y decisión; selección comprobable que separe solicitado, soportado y efectivo. Sin importar Torch al cargar el módulo |
| `src/scribnic/downloads.py` | Compartir la identidad de plataforma y capacidades de la distribución preparada; mantener descargas y builds fijadas |
| `src/scribnic/backends/moss_transcribe_diarize.py` | Consultar Torch solo al cargar este backend; utilizar la decisión de dtype y comprobar inputs del modelo |
| `src/scribnic/backends/moss_audio.py` | Compartir la decisión de cómputo anterior, conservando sus restricciones de versión y cálculo de memoria |
| `src/scribnic/runtimes/{llama_cpp,whisper_cpp,nemo_cpp}.py` | Traducir una selección compatible a los argumentos de cada servidor; recoger su selección efectiva sin incorporar lógica de prompts |
| `src/scribnic/config.py` | Validar valores explícitos y representar únicamente ajustes que realmente se usen; evitar un objeto genérico de opciones en cada capa |
| `src/scribnic/telemetry/otel.py` y persistencia existente | Asociar solicitado, seleccionado, efectivo, evidencia y motivo al run sin confundirlos con el formato de los pesos |
| CLI y documentación | Presentar el diagnóstico y permitir overrides; la decisión debe funcionar también mediante la API Python |
| Nuevo `tests/test_compute.py` y pruebas de fronteras existentes | Fakes para CPU con/sin BF16, capacidades desconocidas, GPU presente sin runtime compatible, overrides, endpoint externo y procedencia de la decisión |

Primera entrega propuesta: resolver exclusivamente BF16/FP32 en los dos backends
Torch usando una decisión compartida y registrable. Después, integrar la detección
específica de los servidores nativos. No hace falta un nuevo backend de inferencia
ni otro runtime para esta selección; la evidencia de cada motor sigue siendo propia.
