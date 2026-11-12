# SeñaBusca — diccionario inverso de ASL con cámara

**CC3092 · Deep Learning y Sistemas Inteligentes** — Universidad del Valle de Guatemala
**Esteban Cárcamo (23016) · Nicolás Concuá (23197)** · Proyecto final, noviembre de 2026

El usuario hace una seña de lengua de señas americana (ASL) frente a la cámara y la app le muestra
las 3 coincidencias más probables con su significado en español. El prototipo reconoce 50 señas
de uso cotidiano para estudiantes de ASL. El núcleo es un modelo de deep learning **entrenado por
nosotros** sobre secuencias de puntos clave (MediaPipe): LSTM (v0), Transformer (v1) y Transformer
con **transfer learning** desde 450 señas distintas (v2). Ninguna parte del sistema usa un LLM.

## Resultados principales (prueba: 11 personas nunca vistas)

| Versión | Modelo | Top-1 | Top-3 | Seg/búsqueda* | Valor neto USD/mes* [IC 95 %] | Latencia CPU |
|---|---|---|---|---|---|---|
| — | Regresión logística (sin DL) | 0.617 | 0.799 | — | 5,607 | 0.1 ms |
| v0 | LSTM bidireccional desde cero | 0.656 | 0.862 | 99 s | 6,169 [5,891, 6,390] | 2.4 ms |
| v1 | Transformer desde cero + aumentos | 0.883 | 0.969 | 85 s | 6,637 [6,541, 6,739] | 1.0 ms |
| **v2** | **Transformer preentrenado en 450 señas + fine-tuning** | **0.894** | **0.972** | **82 s** | **6,754 [6,667, 6,838]** | **1.0 ms** |

\* Modelo de valor (`asl/valor.py`): 500 estudiantes × 20 búsquedas/mes, búsqueda manual = 285 s
equivalentes, 15 % de búsquedas fuera del vocabulario (medidas con videos reales de otras señas),
umbral de confianza elegido en validación. v2 − v0 = +585 USD/mes (IC 95 % [377, 865]).

* **Prueba de punta a punta en la app** (60 videos de prueba por el flujo completo del navegador):
  v2 acierta 57/60 en la primera opción y 59/60 en las 3 mostradas; v0, 44/60 y 54/60.
* **Iteración de datos:** extraer los landmarks en modo IMAGE (la primera versión) mientras la app
  usa modo VIDEO bajaba el top-1 del Transformer de 0.887 a 0.837 en datos como los de la app;
  re-extraer en VIDEO lo corrigió (0.895). Ver `reports/tablas/tabla_iteracion_datos.csv`.
* **Transfer learning** gana poco con todos los datos, pero con solo 9 personas de entrenamiento da
  0.926 de top-3 contra 0.775 desde cero (`reports/figuras/curva_aprendizaje.png`).

Tablas completas: `reports/tablas/tabla_iteraciones.md` (una fila por iteración, 3 semillas) y
`reports/tablas/tabla_versiones.md`.

## Estructura

```
ProyectoFinal-DL/
├── asl/                      # librería compartida por entrenamiento y app
│   ├── features.py           # landmarks -> tensor (64 cuadros x 161): recorte, normalización, remuestreo
│   ├── data.py               # carga del dataset y aumentos (espejo, afín, deformación temporal)
│   ├── models.py             # LSTM bidireccional y Transformer encoder
│   ├── valor.py              # MODELO DE VALOR (función explícita y supuestos)
│   └── extraccion.py         # MediaPipe sobre un archivo de video (respaldo de la app)
├── scripts/
│   ├── 00_descargar_mediapipe.sh
│   ├── 01_seleccionar_y_descargar.py   # elige las señas y baja SOLO esos videos del zip remoto
│   ├── 02_extraer_landmarks.py         # MediaPipe en paralelo -> data/landmarks/*.npy
│   ├── 03_entrenar.py                  # todas las iteraciones (3 semillas c/u)
│   ├── 03b_predecir_fuera_vocabulario.py  # comportamiento con señas que el modelo no conoce
│   ├── 04_baseline_clasico.py          # alternativa sin deep learning
│   ├── 05_modelo_valor.py              # tablas, curva de valor, sensibilidad, umbral, despliegue.json
│   ├── 06_explicabilidad_y_plantillas.py
│   ├── 07_diagrama_arquitectura.py  08_numeros_informe.py  09_iteracion_datos.py  10_latencia.py
│   ├── informe/build_informe.js        # genera el .docx del trabajo escrito
│   ├── construir_notebook.py
│   └── run_all.sh                      # reproduce todo
├── app/
│   ├── server.py             # FastAPI: /api/predict, /api/predict_video, /api/feedback, /api/info
│   └── static/               # interfaz web (MediaPipe Tasks en el navegador)
├── notebook/proyecto_final.ipynb       # EDA, resultados, modelo de valor, explicabilidad (ejecutado)
├── reports/
│   ├── figuras/  tablas/  runs/        # resultados (métricas por corrida, curvas)
│   └── Informe_ProyectoFinal_ASL.docx  # trabajo escrito
├── docs/                     # traspaso, guion de demo/video, protocolo de prueba con usuarios
└── models/                   # pesos entrenados + despliegue.json (modelo y umbral que usa la app)
```

## Instalación

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash scripts/00_descargar_mediapipe.sh
```

## Ejecutar la app

```bash
.venv/bin/uvicorn app.server:app --port 8000
```

Abrir http://localhost:8000, encender la cámara y hacer una seña empezando y terminando con las
manos abajo. También se puede subir un video. El selector de arriba cambia entre v0, v1 y v2 para
comparar en vivo.

La app necesita los 3 pesos listados en `models/despliegue.json` (~20 MB, los únicos que se versionan; el resto se regenera con `scripts/run_all.sh`). Las animaciones de
ejemplo de cada seña (`models/plantillas.npz`) se generan localmente con el script 06 porque se
derivan del dataset y su licencia no permite redistribuirlo; sin ese archivo la app funciona igual
pero sin el esqueleto animado.

## Reproducir las métricas

```bash
bash scripts/run_all.sh        # ≈ 2–3 h en Apple M4 Pro (descarga ~10 GB parciales)
```

El orden y cada paso están en `scripts/run_all.sh`. Las semillas están fijas. El notebook
`notebook/proyecto_final.ipynb` carga los resultados (o re-entrena con `ENTRENAR = True`).

## Datos y licencia

[ASL Citizen](https://www.microsoft.com/en-us/research/project/asl-citizen/) (Desai et al., 2023,
Microsoft Research): 83 399 videos de 2 731 señas grabados por 52 personas sordas o con
discapacidad auditiva. Licencia de investigación **no comercial**; los videos y todo lo derivado
de ellos (landmarks, cachés) **no se versionan** (ver `.gitignore`). El script 01 descarga solo el
subconjunto usado (~15 300 videos) leyendo el zip oficial por rangos HTTP.
