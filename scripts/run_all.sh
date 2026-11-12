#!/usr/bin/env bash
# Reproduce todo el proyecto de punta a punta (≈ 2–3 h en un Apple M4 Pro).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

bash scripts/00_descargar_mediapipe.sh
$PY scripts/01_seleccionar_y_descargar.py              # ~10 GB parciales del zip de ASL Citizen
GLOG_minloglevel=2 $PY scripts/02_extraer_landmarks.py --procesos 10

# Baseline clásico y modelos desde cero (3 semillas cada uno)
$PY scripts/04_baseline_clasico.py
$PY scripts/03_entrenar.py it0_lstm_base it1_lstm_aug it2_tf_aug it3_tf_aug_small
# Preentrenamiento (450 señas distintas) y transfer learning
$PY scripts/03_entrenar.py pre_tf pre_lstm
$PY scripts/03_entrenar.py it4_tl_tf_congelado it5_tl_tf_finetune it6_tl_lstm_finetune
# Curva de aprendizaje (fracciones de participantes)
$PY scripts/03_entrenar.py lc_tf_f25 lc_tf_f50 lc_tf_f75 lc_tl_f25 lc_tl_f50 lc_tl_f75

# Señas fuera de vocabulario y latencia medida con la máquina en reposo
$PY scripts/03b_predecir_fuera_vocabulario.py
$PY scripts/10_latencia.py
# Modelo de valor, figuras, despliegue, explicabilidad
$PY scripts/05_modelo_valor.py
$PY scripts/06_explicabilidad_y_plantillas.py
$PY scripts/07_diagrama_arquitectura.py
# (09_iteracion_datos.py compara con la extracción IMAGE antigua: requiere data/landmarks_image y models/image_mode)
# Notebook ejecutado e informe
$PY scripts/construir_notebook.py
(cd notebook && ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace proyecto_final.ipynb)
$PY scripts/08_numeros_informe.py
(cd scripts/informe && npm install docx >/dev/null && node build_informe.js)
echo "Listo. Ejecuta la app con:  .venv/bin/uvicorn app.server:app --port 8000"
