# Guion de presentación (pitch 8–10 min + demo 3–5 min) y video (2–4 min)

> Borrador. Los números entre ⟨⟩ se llenan con `reports/tablas/tabla_versiones.csv` y
> `reports/tablas/resumen_valor.json` (ver informe). Ajusten el reparto de diapositivas.

## Pitch (≈ 9 min, ~10 diapositivas)

| # | Diapositiva | Quién | Mensaje clave |
|---|---|---|---|
| 1 | Portada: SeñaBusca | ambos | "Haz la seña, te decimos qué significa." |
| 2 | Problema y usuario | Esteban | Un estudiante de ASL ve una seña y no sabe qué es. Los diccionarios van de palabra → seña. Buscar por rasgos tarda ~3 min y falla 1 de cada 3 veces. |
| 3 | ¿Por qué deep learning? | Esteban | Variación entre personas, cámaras y velocidad. Baseline clásico sin DL: ⟨top-3⟩. El DL lo supera por ⟨x⟩ puntos y eso son ⟨USD⟩/mes. |
| 4 | Datos | Nicolás | ASL Citizen: 52 personas sordas, 50 señas, ~1 500 videos. Split por PERSONA: el usuario nunca está en entrenamiento. Licencia no comercial: no publicamos videos. |
| 5 | Pipeline | Nicolás | Cámara → MediaPipe en el navegador (el video no sale del equipo) → 64×161 → modelo → top-3 + confianza → umbral → modo seguro / "no estoy seguro". |
| 6 | Iteraciones | Nicolás | Tabla: clásico → LSTM → +aumentos → Transformer → transfer learning. Qué cambió en cada una. |
| 6b | Iteración de datos (historia de "dominio técnico") | Nicolás | Entrenamos con MediaPipe modo IMAGE, la app usaba VIDEO → en la app acertaba menos (50/60). Lo detectamos probando de punta a punta, re-extrajimos todo y re-entrenamos → 57/60. Lección: el preprocesamiento de entrenamiento y producción debe ser idéntico. |
| 7 | Modelo de valor | Esteban | La fórmula, en una línea. Supuestos clave y de dónde salen. |
| 8 | Curva de valor y umbral | Esteban | Valor vs. umbral (el punto se elige en validación). Valor vs. top-3: punto de equilibrio y rendimientos decrecientes. |
| 9 | Sensibilidad y riesgos | ambos | Tornado ±20/50 %: la conclusión se mantiene salvo ⟨supuesto⟩. Sesgos por persona, mano zurda (aumento espejo), qué NO hace el sistema. |
| 10 | Conclusión de negocio | ambos | v2 genera ⟨USD⟩/mes más que v0 por ⟨costo extra⟩ de cómputo. Siguiente paso: más señas, prueba con usuarios reales. |

## Demo en vivo (≈ 4 min)

1. Abrir http://localhost:8000 (levantar antes: `.venv/bin/uvicorn app.server:app --port 8000`).
2. Encender la cámara. Hacer **HELLO** (fácil) → aparece "hola" con confianza alta (modo seguro).
3. Hacer **una seña difícil** (ver `reports/tablas/por_sena.csv`) → mostrar el modo "no estoy
   seguro" con 10 opciones: así se comunica la incertidumbre.
4. Cambiar el selector a **v0** y repetir la misma seña: comparación en vivo de versiones.
5. Pedirle al catedrático que proponga una de las 50 señas (lista en "Ver las 50 señas…").
6. Mostrar `logs/app_eventos.jsonl`: así se monitorea en producción (confianza promedio, tasa de
   "ninguna es"), que es el plan de monitoreo.
7. **Plan B si la cámara falla:** "Subir video" con un video grabado antes con el celular
   (llevar 3–4 grabados de ustedes mismos haciendo señas del vocabulario).

Practicar antes: luz de frente, fondo liso, hombros visibles, empezar y terminar con las manos
abajo. Las señas se aprenden en minutos con un diccionario en línea (Lifeprint / Handspeak).

## Video de demostración (2–4 min, .mp4)

1. (0:00–0:20) Problema en una frase + quién lo usa.
2. (0:20–1:40) Grabación de pantalla de la app: 3 señas (una fácil, una difícil con el modo "no
   estoy seguro", una con el selector v0 vs v2).
3. (1:40–2:40) Diapositiva de la curva de valor + la cifra de USD/mes y la diferencia entre
   versiones.
4. (2:40–3:00) Cierre: limitaciones y siguiente paso.

Grabar con QuickTime (Archivo → Nueva grabación de pantalla), exportar a .mp4.
