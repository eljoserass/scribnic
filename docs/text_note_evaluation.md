# Comparación de LFM3 texto y Bonsai 4B

Evaluación local del 28 de octubre de 2026. Se compara generación de notas, sin
ejecutar reconocimiento de audio. Es un caso y una ejecución por perfil, con
revisión cualitativa contra el guion; no una escala clínica validada.

## Entrada y ejecución

La fuente es `data/examples/original.json`, con 33 turnos y 5,154 caracteres de
conversación. Es un JSON de escenario, no un artefacto Scribnic directamente:
el script temporal extrae los textos y guarda un `Transcript` en
`runs/text-notes-lfm3-bonsai4b-20261028/inputs/plain.json`. No añade el resumen,
contexto del escenario, voces ni duraciones estimadas al prompt.

La comparación principal entrega exactamente ese mismo artefacto a tres
workflows de texto. Cada modelo tiene su propia sesión y una única llamada a
`session.run`; `session.start(source_kind="transcript")` mide la carga aparte.
Solo se ejecuta el paso `note`. Los pesos se descargaron y verificaron antes de
medir; los modelos se ejecutaron secuencialmente en CPU Linux x86-64/WSL, seis hilos.

Se conserva la plantilla `soap_es`, seed 42 y contexto 8192. Los parámetros de
muestreo siguen las recomendaciones de cada autor, en lugar de imponer
temperatura cero a todos. El límite experimental es 2048 tokens, con petición de
máximo 250 palabras: todas las respuestas terminaron antes de 768 tokens, que sigue
siendo el límite de los presets. No se cortó la entrada ni la salida.

| Perfil | Temperatura | Top-p / top-k | Penalizaciones |
| --- | ---: | --- | --- |
| Bonsai 2B y 4B, sin pensamiento | 0.7 | 0.8 / 20 | Presencia 1.5; repetición 1.0; ventana 64; min-p 0 |
| LFM3-1.7B Q8_0, presupuesto de razonamiento cero | 0.1 | 1.0 / 50 | Repetición 1.1; ventana 64; min-p 0 |

Fuentes: [Prism](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/SMALL-MODELS.md)
y [LiquidAI](https://huggingface.co/LiquidAI/LFM3-1.7B#inference). El perfil general
de Liquid especifica temperatura, top-k y repetición; top-p 1 y min-p 0 mantienen
los filtros adicionales neutros. Son perfiles de uso, no resultados clínicos.

## Resultados principales

| Modelo | Generación, incluida prefill | Carga del servicio | Pico RSS cliente + servidor durante generación | Palabras aproximadas |
| --- | ---: | ---: | ---: | ---: |
| Bonsai 2B PQ2_0, control | 9.41 s | 2.22 s | 1.46 GB | 120 |
| Bonsai 4B PQ2_0 | 26.91 s | 2.37 s | 3.03 GB | 184 |
| LFM3-1.7B Q8_0 | 30.41 s | 0.63 s | 2.20 GB | 242 |

RSS es una suma muestreada cada 250 ms en la misma pasada, con posibles páginas
compartidas contadas varias veces. No es RAM física exclusiva. Los tiempos no son
promedios ni incluyen descargas; las tres peticiones principales tienen cero tokens
de prompt reutilizados de una generación anterior. Las palabras se cuentan
aproximadamente por espacios. Formatos, modelos y runtimes difieren: esta comparación
no aísla el efecto de una cuantización por sí sola.

Prefiero provisionalmente **Bonsai 4B como nota de este caso**: mejor síntesis y
organización que el control 2B, y conserva la advertencia de no dejar el tratamiento
de golpe. LFM3 cubre más detalles y usa menos RSS observado que Bonsai 4B Q2, pero
su redacción y algunas instrucciones son menos precisas. Ninguna nota es plenamente
fiel al guion.

| Hecho del guion | Bonsai 2B | Bonsai 4B | LFM3 |
| --- | --- | --- | --- |
| Propranolol 40 mg cada 12 horas | Conservado | Conservado | Conservado |
| Colesterol medido 240 mg/dL | Conservado | Conservado | Conservado |
| Objetivo de colesterol menor de 200 | Omitido | Omitido | Conservado |
| Revisión en un mes | Conservada | Conservada | Conservada |
| Analítica dentro de 15 días | Incluida en un cierre impreciso | Conservada | Conservada |
| No suspender el medicamento de golpe | Omitido | Conservado | Omitido |
| Considerar fármaco para colesterol si no mejora en tres meses | Omitido | Omitido | Conservado |
| Reducción de café: dos tazas diarias una semana, después una diaria | Parcial | Distorsionada | Conservada |
| Parar inmediatamente ante síntomas durante ejercicio | Omitido | Omitido | Distorsionado |

Errores relevantes de las notas conservadas:

- Bonsai 2B añade una «respuesta favorable al tratamiento farmacológico» cuando
  este se está iniciando. También introduce una causa metabólica no establecida.
- Bonsai 4B escribe «de 3-4 a 2 tazas/semana»: cambia la frecuencia del plan de café.
  Presenta café y tabaco como desencadenantes con más certeza que el guion.
- LFM3 mezcla la recomendación general de evitar intensidad extrema con la
  actuación ante palpitaciones, mareo o dolor: no conserva «pare inmediatamente».
  Omite la retirada no brusca y contiene «Constantes: EPCOTEGRAMO» en Objetivo.

## Formato de razonamiento de LFM3

El primer perfil, `enable_thinking=false` junto a `reasoning-budget=0`, produjo
un `</think>` huérfano al principio de `message.content` con llama.cpp b11946.
Las ejecuciones fueron rechazadas por el guard existente. Cambiar únicamente
`reasoning-format` a `deepseek` no lo resolvió.

El ajuste validado deja abierto el prefijo de protocolo mediante `thinking=true`
y conserva `reasoning-budget=0`: el servidor fuerza el cierre inmediato, sin
tokens libres de razonamiento. La respuesta final llega limpia. No se hace strip
de etiquetas ni se debilita el guard del cliente. Este ajuste está en los tres
workflows LFM3 de texto; la inferencia real se comprobó con Q8_0.

La interacción corresponde al parser y sampler de la versión fijada:
[parser](https://github.com/ggml-org/llama.cpp/blob/b11946/common/parsers/qwen3-coder.cpp),
[presupuesto](https://github.com/ggml-org/llama.cpp/blob/b11946/common/reasoning-budget.cpp).
Las recomendaciones de muestreo mejoran las condiciones de la comparación, pero
no eran la causa del cierre de protocolo huérfano.

## Artefactos y reproducción

El informe conjunto queda en `runs/text-notes-lfm3-bonsai4b-20261028/report.html`.
Incluye las tres notas principales, métricas, perfiles efectivos, matriz de hechos,
referencia y enlaces a cada run. `manifest.json` registra hashes de origen y entrada.
Los intentos iniciales greedy y con etiquetas permanecen como ensayos secundarios;
no se mezclan con las cifras principales de perfiles recomendados.

El script temporal `.tmp/run_recommended_text_notes.py` usa la API pública, con
una llamada por modelo. Los runs y sus scripts son artefactos locales, no source.
La referencia exportada también funciona con la CLI:

```bash
uv run scribnic run runs/text-notes-lfm3-bonsai4b-20261028/inputs/plain.json \
  --workflow bonsai_4b_note --output-dir runs/bonsai4-reference-new

uv run scribnic run runs/text-notes-lfm3-bonsai4b-20261028/inputs/plain.json \
  --workflow lfm3_17b_q8_0_note --output-dir runs/lfm3-text-reference-new
```
