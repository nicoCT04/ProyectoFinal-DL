# Traspaso — estado del proyecto final (SeñaBusca)

Documento para Esteban (y Nicolás) con lo que ya está hecho, cómo correrlo, qué está flojo y qué
partes quedan abiertas para repartir. Todo está **local, sin commits**.

## 1. Qué hay

| Pieza | Dónde | Estado |
|---|---|---|
| Descarga parcial de ASL Citizen (50 + 450 señas) | `scripts/01_*` | ✅ hecho (~15 300 videos, ~10 GB en `data/raw/`) |
| Extracción de landmarks MediaPipe | `scripts/02_*`, `asl/features.py` | ✅ hecho (`data/landmarks/`) |
| Baseline clásico (sin DL) | `scripts/04_*` | ✅ |
| v0 LSTM, v1 Transformer, v2 transfer learning (3 semillas c/u) | `scripts/03_*`, `asl/models.py` | ✅ ver `reports/tablas/tabla_iteraciones.csv` |
| Señas fuera de vocabulario | `scripts/03b_*` | ✅ |
| Modelo de valor (función, umbral, sensibilidad, IC) | `asl/valor.py`, `scripts/05_*` | ✅ supuestos ESTIMADOS (falta medirlos) |
| Explicabilidad y sesgos | `scripts/06_*` | ✅ |
| App web (cámara, auto-grabación, top-3, v0/v1/v2, feedback) | `app/` | ✅ probada con videos; **falta probarla con cámara real** |
| Notebook ejecutado | `notebook/proyecto_final.ipynb` | ✅ |
| Borrador del trabajo escrito (.docx) | `reports/Informe_ProyectoFinal_ASL.docx` | ✅ borrador para revisar |
| Guion de pitch, demo y video | `docs/GUION_PRESENTACION_Y_VIDEO.md` | 📝 borrador |
| Protocolo de prueba con usuarios | `docs/PROTOCOLO_PRUEBA_USUARIOS.md` | 📝 listo para ejecutar |

Resultados principales: ver `reports/tablas/tabla_versiones.md` y `README.md` (sección de resultados).

## 2. Cómo correr la app (5 minutos)

```bash
cd ProyectoFinal-DL
source .venv/bin/activate          # o crear el venv: ver README
uvicorn app.server:app --port 8000
```

Abrir http://localhost:8000 → "Encender cámara" → hacer una seña empezando y terminando con las
manos **abajo, fuera de cuadro**. La lista de las 50 señas está en la misma página.
Para aprender rápido cómo se hace cada seña: Lifeprint (lifeprint.com) o Handspeak.

Esteban necesita los datos/modelos: el repo NO incluye datos (licencia). Opciones: (a) copiar la
carpeta completa por USB/Drive (pesa ~11 GB por `data/raw/`; sin `data/raw` pesa poco y la app
funciona igual), o (b) correr `bash scripts/run_all.sh` (≈ 2–3 h).

## 2b. Lo que pasó en la noche (vale la pena contarlo en la presentación)

* **Desajuste entrenamiento ↔ producción detectado y corregido.** Primero extrajimos los puntos
  con MediaPipe en modo IMAGE (cada cuadro por separado), pero el navegador usa modo VIDEO (con
  seguimiento). Al probar la app con videos de prueba, acertaba menos que offline. Comparando un
  mismo video cuadro a cuadro: VIDEO detecta más manos y la muñeca tiembla ~4× menos. Se
  re-extrajo todo en VIDEO y se re-entrenó. En la app (60 videos de prueba, flujo completo del
  navegador): **top-1 50/60 → 57/60**, top-3 56/60 → 59/60. Tabla en
  `reports/tablas/tabla_iteracion_datos.csv` y `prueba_app_e2e.json`; lo viejo quedó archivado en
  `data/landmarks_image/`, `models/image_mode/`, `reports/iteracion_datos_image/`.
* **Señas fuera del vocabulario medidas, no supuestas**: con videos de las 450 señas extra. Por
  eso el umbral óptimo es alto (~0.9): el modo "no estoy seguro" atrapa la mayoría de señas
  desconocidas.
* **Regla de "mano activa"** en la app: la grabación solo cuenta manos por encima de la zona de
  reposo (hombros + 1.3·ancho de hombros), validada en el dataset (conserva 99.9 % de los cuadros
  de señas). Evita grabaciones en bucle si la persona deja las manos sobre la mesa.

## 3. Decisiones tomadas (y por qué) — revisarlas juntos

1. **Puntos clave (MediaPipe) en vez de video crudo.** Más rápido de entrenar en una laptop,
   invariante a fondo/iluminación, y el video del usuario no sale del navegador (privacidad).
   Contra: si MediaPipe no ve la mano, el modelo tampoco.
2. **Split oficial por participante.** El usuario nunca está en el entrenamiento (uso real).
3. **Transfer learning "propio"**: preentrenar con 450 señas distintas de ASL Citizen y ajustar a
   las 50. Cumple la excepción del catedrático (fine-tuning/TL documentando base vs. ajustado):
   base = it2 (Transformer desde cero), ajustado = it4 (congelado) / it5 (fine-tuning).
4. **Umbral elegido por valor en validación**, reportado en prueba.
5. **Señas fuera de vocabulario medidas**, no supuestas (videos de prueba de las 450 señas extra).

## 4. Lo que está flojo / riesgos para la presentación

* **Los supuestos del modelo de valor son estimaciones** (t_manual, p_manual, p_engano, c_error).
  La rúbrica da 4 pts al "vínculo métrica ↔ negocio": medir aunque sea 5 personas con el
  protocolo sube mucho la credibilidad. **← la tarea más valiosa que queda.**
* **La app no se ha probado con cámara real** (se probó con videos del dataset pasando por el
  mismo flujo del navegador). Probarla con distintas luces, distancias y con alguien zurdo.
* **Licencia:** el dataset es no comercial. Los pesos del modelo están en `models/`; decidir si se
  suben al repo público (es práctica común, pero la licencia prohíbe redistribuir "modificaciones
  de los datos"; los pesos no son los datos, pero conviene mencionarlo en la presentación).
* El vocabulario es de 50 señas: si el catedrático propone una seña fuera de la lista, la app
  debería decir "no estoy seguro" — buen momento para explicar el umbral y las señas fuera de
  vocabulario. Practicarlo.

## 5. Partes que quedan abiertas (propuesta de reparto)

| Tarea | Propuesta | Por qué |
|---|---|---|
| Prueba con usuarios (5–8 personas) y actualizar supuestos | **Esteban** | Es lo que más sube la nota de "vínculo métrica ↔ negocio" y "viabilidad". |
| Revisar y pulir el informe (.docx → PDF) | ambos (Esteban 1, 2, 6; Nicolás 3, 4, 5, 7) | Que los dos dominen todo para las preguntas técnicas. |
| Diapositivas | **Esteban** (con el guion) | |
| Video de demostración 2–4 min (.mp4) | **Nicolás** | |
| Probar la app con cámara y ajustar parámetros de segmentación (`CFG` en `app/static/app.js`) | Nicolás | |
| (Opcional) Ampliar vocabulario a 100–200 señas | quien tenga tiempo | Sube `p_vocab`, el supuesto más sensible; el pipeline ya escala. |
| (Opcional) Desplegar en la nube (Render/Railway) | opcional | La app es un solo proceso CPU; ojo con la licencia. |
| Subir a GitHub (commits) | Nicolás | Decidir qué entra (ver `.gitignore`). |

## 6. Preguntas técnicas probables (practicar)

* ¿Por qué Transformer y no LSTM? ¿Qué aprende la atención aquí? (ver figura de saliencia/atención)
* ¿Qué pasa si la persona es zurda? (aumento de espejo + prueba de robustez)
* ¿Cómo eligieron el umbral? ¿Por qué no en prueba?
* ¿Cuánto cuesta por predicción? ¿Qué pasa con 100× usuarios?
* ¿Hay fuga de información en el preentrenamiento? (no: otras señas y solo participantes de train)
* ¿Por qué top-3 y no top-1? (el usuario verifica visualmente; refs. de usabilidad de diccionarios)
* ¿Qué supuesto, si está mal, cambia la conclusión? (tornado)
