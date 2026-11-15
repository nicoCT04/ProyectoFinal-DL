# Guion de presentación (pitch 8–10 min + demo 3–5 min) y video (2–4 min)

> Números de `reports/tablas/tabla_versiones.csv`, `resumen_valor.json`, `prueba_usuarios.json`,
> `validacion_propia.json` y `prueba_camara_simulada.json` (15-nov-2026). Si se regenera el modelo de
> valor, revisar las cifras marcadas con †.

## Pitch (≈ 9 min, ~11 diapositivas)

| # | Diapositiva | Quién | Mensaje clave |
|---|---|---|---|
| 1 | Portada: SeñaBusca | ambos | "Haz la seña, te decimos qué significa." |
| 2 | Problema y usuario | Esteban | Un estudiante de ASL ve una seña y no sabe qué es. Los diccionarios van de palabra → seña, no al revés. **Lo medimos:** buscando a mano en Handspeak/SignASL/Lifeprint tardamos una mediana de **274 s** por seña y no encontramos **ninguna de 3**. Con SeñaBusca: **6 de 6 en 10 s**. |
| 3 | ¿Por qué deep learning? | Esteban | La misma seña cambia entre personas, cámaras y velocidad. Sin DL (regresión logística sobre estadísticos): **79.9 % top-3**. Con DL: **98.1 %**. El error top-3 baja de 20.1 % a 1.9 % y eso vale **1,427 USD/mes** † más. **No es un LLM:** el modelo lo diseñamos y entrenamos nosotros. |
| 4 | Datos | Nicolás | ASL Citizen: 52 personas sordas, 50 señas cotidianas (1 538 videos) + 450 señas extra para preentrenar. Split por PERSONA: el usuario nunca está en entrenamiento. Licencia no comercial: no publicamos videos. |
| 5 | Pipeline | Nicolás | Cámara → MediaPipe en el navegador (el video no sale del equipo) → 64 × 161 → modelo → top-3 + confianza → umbral → modo seguro / "no estoy seguro". 1.8 ms de inferencia en CPU; ≈ 0.0025 USD por búsqueda. |
| 6 | Iteraciones | Nicolás | Clásico → LSTM (0.656 top-1) → + aumentos (0.813) → Transformer (0.895) → transfer learning desde 450 señas → **it7** (0.906 / 0.981). TL frente al modelo base con la misma arquitectura: con solo 9 personas de entrenamiento, 0.926 de top-3 contra 0.775 desde cero. |
| 7 | Iteración guiada por el producto | Esteban | Probamos la app como la usaría una persona (cámara simulada en tiempo real con 616 videos) y no daba lo del notebook. Dos causas: **(1)** la segmentación descartaba señas cortas → corregido: v2 pasó de 48/60 a 58/60 en top-3; **(2)** el modelo sufría con la seña cortada o la cámara cerca → **it7**, aumentos que imitan la cámara: 0.912 → 0.933 con la seña cortada, sin perder nada en prueba limpia. Antes también: el desajuste IMAGE/VIDEO de MediaPipe (50/60 → 57/60). Lección: **probar el producto, no solo el modelo**. |
| 8 | Modelo de valor | Esteban | Valor = (tiempo manual − tiempo con la app) × búsquedas × valor de la hora − costo de operación, con los engaños y las búsquedas fallidas como costo. Supuestos **medidos** en la prueba piloto: búsqueda manual 274 s, revisar 3 opciones 2.4 s, grabar 7 s. p_manual se deja conservador (65 %) porque 3 búsquedas no bastan. |
| 9 | Curva de valor y umbral | Esteban | Valor vs. umbral: el punto se elige en validación. **Medir cambió la decisión:** revisar 3 opciones cuesta 2.4 s (no 15), así que conviene mostrarlas más: el umbral óptimo de v2 bajó de 0.83 a **0.67**. Cada punto de top-3 vale ≈ 82 USD/mes †. |
| 10 | Sensibilidad y riesgos | ambos | Tornado ±50 %: ningún supuesto solo vuelve negativo el valor (aun con la búsqueda manual a la mitad: 5,928 USD/mes †). Escenario pesimista conjunto: 1,031 USD/mes †. Riesgos: sesgo por detección de manos (iluminación, cámara), señas fuera de las 50, no sustituye a un intérprete. |
| 11 | Conclusión de negocio | ambos | v2 genera **9,873 USD/mes** † (IC 95 % [9,782, 9,950]), **788 USD/mes más que v0** † (IC [513, 1,115]) con el mismo costo de cómputo (≈ 25 USD/mes). Quien paga es la institución (curso o programa de ASL), no el estudiante. Siguiente paso: más señas (sube p_vocab, el supuesto más sensible) y medir con más estudiantes. |

## Demo en vivo (≈ 4 min)

Antes de entrar: ver `docs/CHECKLIST_PRESENTACION.md` (arranque, luz, plan B).

1. App abierta en Chrome en **http://localhost:8000** (sin `?prueba`), cámara encendida y modelo **v2**.
2. **HELLO** (fácil) → "hola" en modo seguro con confianza alta. Mostrar el esqueleto animado de la tarjeta.
3. **Una seña del vocabulario que proponga el catedrático** (lista en "Ver las 50 señas…"). Si sale mal a la
   primera, repetirla con calma: en la validación propia acertó 65 % al primer intento y 100 % al repetir.
4. Cambiar el selector a **v0** y repetir la misma seña: comparación en vivo de versiones (v0 suele dar
   menos confianza o fallar).
5. **Una seña que NO está entre las 50** (ej. "computadora") → idealmente "No estoy seguro" con 10
   opciones. Ojo: con el umbral 0.67 eso pasa en ~64 % de las señas fuera de vocabulario; si muestra
   opciones con seguridad, explicarlo: es el trade-off del umbral elegido por valor (diapositiva 9).
6. Monitoreo: `logs/app_eventos.jsonl` (confianza, "Ninguna es", posición elegida) = plan de monitoreo.

**Plan B si la cámara falla:** botón "Subir video" con un video de prueba de ASL Citizen
(`data/raw/ASL_Citizen/videos/`, personas no vistas) o uno grabado por ustedes. Pasa por el mismo flujo.

## Video de demostración (2–4 min, .mp4)

1. (0:00–0:20) Problema en una frase + quién lo usa.
2. (0:20–1:40) Grabación de pantalla de la app: 3 señas (una fácil, una fuera del vocabulario con el
   modo "no estoy seguro", una con el selector v0 vs v2).
3. (1:40–2:40) Diapositiva de la curva de valor + 9,873 USD/mes † y +788 USD/mes de v2 sobre v0 †.
4. (2:40–3:00) Cierre: limitaciones y siguiente paso.

Grabar la pantalla (QuickTime en macOS, OBS o la grabadora del sistema en Linux) y exportar a .mp4.
