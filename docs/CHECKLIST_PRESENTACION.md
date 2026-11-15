# Checklist para la presentación (semana del 16-nov) y la entrega (miércoles 18)

## 1. La rúbrica y dónde está la evidencia

| Criterio (pts) | Qué se evalúa | Evidencia |
|---|---|---|
| Trabajo escrito (10) | Secciones 4.1–4.9 del enunciado | `reports/Informe_ProyectoFinal_ASL.docx` (secciones 1–8 cubren 4.1–4.9; se genera con `scripts/informe/build_informe.js`) |
| Creatividad (3) | Ir más allá de un modelo estándar sobre un dataset público | Diccionario **inverso** (seña → palabra) con cámara; preentrenamiento propio con 450 señas; iteraciones guiadas por probar el producto (IMAGE/VIDEO, segmentación, it7) |
| Viabilidad (4) | Datos, costos, latencia, mantenimiento, riesgos, ¿lo pagaría alguien? | 1.8 ms en CPU, ≈ 0.0025 USD por búsqueda, servidor de 25 USD/mes; funciona sin internet; licencia no comercial y privacidad (el video no sale del navegador); quien paga es la institución |
| Vínculo métrica ↔ negocio (4) | Supuestos justificados, curva de valor, sensibilidad, punto de operación por valor | `asl/valor.py`; supuestos medidos (`docs/RESULTADOS_PRUEBA_USUARIOS.md`); `valor_vs_umbral.png`, `valor_vs_top3.png`, `sensibilidad_tornado.png`; el umbral cambió de 0.83 a 0.67 al medir |
| Dominio técnico (4) | Arquitectura, entrenamiento, decisiones, demo, explicar fallas | Sección 5 de este documento; `tabla_iteraciones.md`; `robustez_webcam.csv`; `prueba_camara_simulada.json` |

**Restricción del curso (sin LLM):** cumple. Modelos propios (LSTM y Transformer) y transfer learning
propio, comparado contra el modelo base con la misma arquitectura (it2 vs. it4/it5/it7, sección 5 del informe).

## 2. Números para tener en la cabeza

| | v0 LSTM | v1 Transformer | **v2 TL + aumentos de cámara (it7)** |
|---|---|---|---|
| Top-1 / top-3 en prueba (11 personas no vistas) | 0.656 / 0.862 | 0.883 / 0.969 | **0.906 / 0.981** |
| Cámara en tiempo real (616 videos) top-1 / top-3 | 60.2 % / 81.5 % | 81.5 % / 90.3 % | **84.4 % / 93.0 %** |
| Valor neto USD/mes | 9,085 | 9,668 | **9,873** [9,782, 9,950] |

* Sin deep learning (regresión logística): 0.617 / 0.799 de top-1 / top-3.
* v2 − v0: **+788 USD/mes** (IC 95 % [513, 1,115]); el costo de operar es el mismo (≈ 25 USD/mes).
* Prueba piloto: a mano, 274 s y 0 de 3 encontradas; con la app, 6 de 6 en 10 s.
* Validación propia (20 señas, cámara real): 65 % en la primera opción al primer intento, 80 % en top-3, 100 % al repetir.

## 3. Antes de salir de casa

- [ ] `git pull` en las dos laptops y `bash scripts/00_descargar_mediapipe.sh` (copia local de MediaPipe).
- [ ] Probar la app en la laptop de la demo con **Chrome** y la cámara del aula si se puede.
- [ ] Plan B: `data/raw/plan_b/` tiene HELLO, FATHER y WATER (videos de prueba que la app acierta al subirlos con 96–98 %). Copiarlos a la laptop de la demo si es otra.
- [ ] Diapositivas en Canva exportadas a PDF por si falla internet.
- [ ] Video de demostración (.mp4, 2–4 min) — Nicolás.

## 4. Arranque de la demo (5 min antes)

```bash
cd ProyectoFinal-DL
.venv/bin/uvicorn app.server:app --port 8000
```

1. Chrome → **http://localhost:8000** (sin `?prueba` ni `?debug`). Recargar con Ctrl+Shift+R.
2. Cerrar cualquier otra pestaña de SeñaBusca (dos pestañas usan la cámara a la vez).
3. "Encender cámara". Modelo **v2**. Luz de frente, hombros visibles, la línea amarilla abajo.
4. Hacer HELLO una vez para calentar (la primera detección tarda un poco más).
5. Si "Error al iniciar": la app reintenta con CPU sola; si aun así falla, plan B (subir video).

Durante la demo: empezar y terminar cada seña con las manos abajo; si no sale a la primera, repetirla
con calma. El guion paso a paso está en `docs/GUION_PRESENTACION_Y_VIDEO.md`.

## 5. Preguntas técnicas probables (con respuesta corta)

* **¿Por qué Transformer y no LSTM?** La LSTM sobreajusta (0.999 en train, 0.689 en prueba); la
  atención relaciona cualquier par de cuadros (inicio y final de la seña) sin pasar por un estado
  recurrente. Pasar a Transformer explica 583 de los 788 USD/mes que separan v2 de v0 †.
* **¿Qué hace el transfer learning y cómo lo comparan con el base?** Se preentrena el mismo Transformer
  para clasificar 450 señas distintas (solo personas de entrenamiento: sin fuga) y se ajusta a las 50.
  Base it2: 0.895 / 0.966; fine-tuning it5: 0.899 / 0.972; it7: 0.900 / 0.977. La ganancia grande es con
  pocos datos: con 9 personas, 0.926 de top-3 contra 0.775 desde cero.
* **¿Qué es it7?** El mismo fine-tuning con aumentos que imitan la cámara en vivo (seña cortada, encuadre,
  mano perdida). Lo motivó probar la app en tiempo real; se eligió por validación, no por prueba.
* **¿Cómo eligieron el umbral? ¿Por qué 0.67?** Maximizando el valor en VALIDACIÓN; se reporta en prueba.
  Al medir que revisar 3 opciones toma 2.4 s (no 15), mostrar 3 opciones conviene más y el óptimo bajó.
  Costo: marca "no estoy seguro" en 64 % de las señas fuera de vocabulario (antes 87 %).
* **¿Los supuestos son reales?** t_manual, t_rev3 y t_grabar medidos en una prueba piloto (1 persona; lo
  decimos así). p_manual sigue estimado (65 %, conservador; en la prueba fue 0 de 3). Ningún supuesto
  ±50 % vuelve negativo el valor; con la búsqueda manual a la mitad, 5,928 USD/mes †.
* **¿Por qué top-3 y no top-1?** El usuario verifica visualmente con el esqueleto animado; los estudios de
  diccionarios por video [2], [3] muestran que importa que la seña aparezca en las primeras posiciones.
* **¿Qué pasa si la persona es zurda?** Aumento de espejo (intercambia manos y lados del cuerpo): con la
  seña reflejada el modelo desplegado prácticamente no cambia.
* **¿Hay fuga de información?** No: split oficial por persona; el preentrenamiento usa otras señas y solo
  personas de entrenamiento; umbral y selección de epoch en validación.
* **¿Por qué en la app acierta menos que en el notebook?** El notebook usa segmentos perfectos del dataset;
  en vivo se suma la segmentación automática y la detección del navegador (≈ 5 puntos menos). Por eso
  medimos con la cámara simulada y corregimos la segmentación (señas cortas descartadas).
* **¿Cuánto cuesta? ¿Y con 100× usuarios?** ≈ 0.0025 USD por búsqueda con 10 000 búsquedas/mes; un proceso
  CPU atiende decenas de búsquedas por segundo, así que 100× sigue cabiendo en el mismo servidor.
* **¿Dónde falla?** BAD, WHO, SCHOOL y HOT (top-3 0.77–0.91); confusiones PLEASE ↔ HAPPY y GOOD ↔ THANKYOU
  (misma forma o ubicación). En vivo, señas rápidas a la altura del pecho (STOP, WORK) cuando MediaPipe
  pierde la mano. Personas con peor detección de manos (luz, cámara) sufren más errores.
* **¿Lo pagaría alguien?** El estudiante no (lo dijo en la prueba piloto); la institución sí: 500
  estudiantes × 20 búsquedas ahorran ≈ 825 horas al mes †. La licencia de ASL Citizen es no comercial:
  un producto real necesitaría datos propios o licenciados.

† Cifras del modelo de valor al 15-nov-2026 (`reports/tablas/resumen_valor.json`).

## 6. Entrega del miércoles 18

- [ ] Revisar el informe juntos (Word) y exportar a PDF. El .docx no se versiona (`.gitignore`).
- [x] Enlace al repositorio en el informe y repositorio público (ya lo es).
- [ ] Video .mp4 y diapositivas.
