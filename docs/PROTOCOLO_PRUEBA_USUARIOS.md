# Protocolo de prueba con usuarios (sección 4.6 del enunciado)

Objetivo: medir con personas reales los supuestos del modelo de valor que hoy son estimaciones
(`t_manual`, `p_manual`, `t_rev3`, `t_grabar`) y comparar **búsqueda manual vs. SeñaBusca**.
Duración: ~15 min por persona. Meta mínima: 5–8 participantes (compañeros de la U).

## Preparación

1. Levantar la app: `.venv/bin/uvicorn app.server:app --port 8000` y abrir http://localhost:8000.
2. Elegir **6 señas** del vocabulario (las 50 están en el desplegable "Ver las 50 señas…").
   Sugeridas (mezcla de fáciles y difíciles según `reports/tablas/por_sena.csv`):
   HELLO, THANKYOU, SCHOOL, UNDERSTAND, BATHROOM, TOMORROW.
3. Tener a mano un video corto de cada seña hecho por **otra persona** (puede ser de un
   diccionario en línea, p. ej. Lifeprint o Handspeak, abierto en otra pestaña). El participante NO
   debe saber qué palabra es.
4. Cronómetro (el del celular) y la hoja de registro de abajo.

## Tarea A — búsqueda manual (3 señas)

"Te voy a mostrar una seña. Encuentra qué significa usando el diccionario en línea que quieras.
Tienes máximo 5 minutos."  → anotar tiempo y si la encontró.

## Tarea B — con SeñaBusca (las otras 3 señas)

"Mira la seña, imítala frente a la cámara y elige la opción correcta."  → la app registra sola en
`logs/app_eventos.jsonl`: confianza, modo (seguro / no seguro), top-3 y, al tocar "Es esta",
el rank elegido y `ms_hasta_elegir` (= `t_rev3` real). Anotar también el tiempo total con
cronómetro (desde que ve la seña hasta que elige).

> Alternar el orden A/B entre participantes (la mitad empieza con B) para evitar el efecto de
> aprendizaje. Usar señas distintas en A y B para cada persona, rotándolas.

## Preguntas finales (escala 1–5)

1. ¿Qué tan fácil fue encontrar la seña con cada método?
2. ¿Confiarías en la respuesta de la app sin revisar el video de ejemplo?
3. ¿Lo usarías en tu curso? ¿Pagarías por algo así (cuánto al mes)?

## Hoja de registro

| Persona | Orden | Seña | Método | Tiempo (s) | ¿Encontró? | Rank en la app | Comentarios |
|---|---|---|---|---|---|---|---|
| P1 | A→B | HELLO | manual | | | – | |
| P1 | A→B | SCHOOL | app | | | | |

## Cómo usar los resultados

* Sustituir en `asl/valor.py` → `Supuestos` los valores medidos (`t_manual` = mediana de la tarea
  A, `p_manual` = fracción encontrada, `t_rev3` = mediana de `ms_hasta_elegir`/1000) y volver a
  correr `scripts/05_modelo_valor.py`. El informe debe decir explícitamente cuáles supuestos
  vienen de la prueba y cuáles siguen siendo estimaciones.
* Los registros de `logs/` pueden contener datos de personas: no subirlos al repositorio
  (ya están en `.gitignore`) y pedir consentimiento verbal antes de la prueba.
