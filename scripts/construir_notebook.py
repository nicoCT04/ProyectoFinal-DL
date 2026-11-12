"""Genera notebook/proyecto_final.ipynb (luego se ejecuta con nbconvert para dejar salidas)."""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))  # noqa: E731
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))    # noqa: E731

md("""
# Proyecto Final — SeñaBusca: diccionario inverso de ASL con cámara

**CC3092 · Deep Learning y Sistemas Inteligentes** — Universidad del Valle de Guatemala
**Esteban Cárcamo (23016) · Nicolás Concuá (23197)** — noviembre de 2026

Enlace al repositorio: https://github.com/nicoCT04/ProyectoFinal-DL

Este notebook documenta el flujo completo: datos → preprocesamiento → modelos (3 versiones +
baseline clásico) → modelo de valor → explicabilidad. **No es el producto final**: el producto es
la app web en `app/` (FastAPI + MediaPipe en el navegador). El entrenamiento pesado está en
`scripts/` para poder reproducirlo por terminal; aquí se cargan sus resultados (con la opción de
re-entrenar poniendo `ENTRENAR = True`).
""")

code("""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import Image, display, Markdown

# Rutas robustas: funciona desde la raíz del proyecto o desde notebook/
ROOT = Path.cwd() if (Path.cwd() / "asl").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))
from asl.features import N_POSE, N_HAND, N_PTS, T_FIJO, preparar, normalizar, recortar_actividad
from asl.data import cargar_grupo, aumentar, reflejar, vocabulario
from asl.valor import SUPUESTOS_BASE, valor, mejor_umbral, costo_manual, supuestos_dict
SEED = 23197
ENTRENAR = False          # True = re-ejecuta scripts/03_entrenar.py (≈ 1–2 h en M4 Pro)
FIG, TAB = ROOT / "reports" / "figuras", ROOT / "reports" / "tablas"
pd.set_option("display.max_colwidth", 80)
""")

md("""
## 1. Contexto y problema

Un estudiante de ASL ve una seña (en clase, en un video, en una conversación) y no sabe qué
significa. Los diccionarios están organizados **por palabra**: sirven para ir de *palabra → seña*,
no de *seña → palabra*. Hoy la alternativa es buscar por rasgos (forma de la mano, ubicación,
movimiento) en sitios especializados o preguntarle a alguien; es lento y muchas veces no se
encuentra. SeñaBusca invierte la búsqueda: el usuario hace la seña frente a la cámara y recibe las
3 coincidencias más probables.

**¿Por qué deep learning?** La misma seña varía entre personas (velocidad, amplitud, mano
dominante), cámaras e iluminación. Las reglas no capturan esa variación y un modelo clásico sobre
estadísticos pierde el orden temporal. Lo comprobamos abajo con un baseline sin deep learning.
""")

md("## 2. Datos y análisis exploratorio")
code("""
voc = vocabulario()
sub = pd.read_csv(ROOT / "data" / "subconjunto.csv")
obj = sub[sub.grupo == "objetivo"]
print("Señas objetivo:", len(voc["objetivo"]), "| señas de preentrenamiento:", len(voc["preentreno"]))
display(sub.groupby(["grupo", "split"]).agg(videos=("Video file", "size"),
        participantes=("Participant ID", "nunique"), senas=("Gloss", "nunique")))
""")
code("""
# ¿La separación es por persona? Ningún participante aparece en dos splits.
parts = obj.groupby("split")["Participant ID"].apply(set)
print("train∩val:", parts["train"] & parts["val"], "| train∩test:", parts["train"] & parts["test"],
      "| val∩test:", parts["val"] & parts["test"])
fig, axs = plt.subplots(1, 2, figsize=(11, 3.2))
obj.groupby("Gloss").size().sort_values().plot.bar(ax=axs[0], color="#2f5bd3", width=0.8)
axs[0].set_title("Videos por seña (objetivo, todos los splits)"); axs[0].tick_params(axis="x", labelsize=6)
obj.groupby(["split", "Participant ID"]).size().unstack(0).plot.bar(ax=axs[1], stacked=True, width=0.8)
axs[1].set_title("Videos por participante"); axs[1].tick_params(axis="x", labelsize=6)
plt.tight_layout(); plt.savefig(FIG / "eda_distribuciones.png"); plt.show()
""")
code("""
# Calidad de la detección: largo de los videos y fracción de cuadros con manos detectadas.
filas = []
for _, r in obj.iterrows():
    a = np.load(ROOT / "data" / "landmarks" / (r["Video file"][:-4] + ".npy")).astype(np.float32)
    manos = ~np.isnan(a[:, N_POSE:, 0]).all(1)
    filas.append(dict(split=r.split, gloss=r.Gloss, part=r["Participant ID"], cuadros=len(a),
                      frac_manos=manos.mean(), frac_pose=(~np.isnan(a[:, 0, 0])).mean(),
                      frac_izq=(~np.isnan(a[:, N_POSE:N_POSE+N_HAND, 0]).all(1)).mean(),
                      frac_der=(~np.isnan(a[:, N_POSE+N_HAND:, 0]).all(1)).mean()))
q = pd.DataFrame(filas)
display(q.describe().round(3))
print("Videos sin ninguna mano detectada:", (q.frac_manos == 0).sum())
fig, axs = plt.subplots(1, 3, figsize=(12, 3))
q.cuadros.hist(ax=axs[0], bins=30, color="#2f5bd3"); axs[0].set_title("Cuadros por video (30 fps)")
q.frac_manos.hist(ax=axs[1], bins=30, color="#1f8a4c"); axs[1].set_title("Fracción de cuadros con mano")
q.groupby("part")[["frac_izq", "frac_der"]].mean().plot.bar(ax=axs[2], width=0.8)
axs[2].set_title("Mano detectada por participante"); axs[2].tick_params(axis="x", labelsize=6)
plt.tight_layout(); plt.savefig(FIG / "eda_calidad_deteccion.png"); plt.show()
""")
md("""
**Lectura del EDA.** Las clases están prácticamente balanceadas (~30 videos por seña), por lo que
la accuracy y el F1 macro cuentan casi lo mismo. El desbalance real está en las **personas**: cada
participante grabó un número distinto de videos y algunos usan la mano izquierda como
dominante; por eso el split oficial (por participante) es el correcto para imitar el uso real: el
usuario de la app nunca está en el entrenamiento.
""")

md("""
### 2.1 Preprocesamiento

1. **MediaPipe** (manos + pose) en modo VIDEO (con seguimiento entre cuadros, igual que en el
   navegador) → 53 puntos (x, y, z) por cuadro: 11 de la parte superior del cuerpo y 21 por mano.
   El video nunca sale del navegador en la app: solo viajan estos números. (Una primera versión
   usó modo IMAGE y eso creó un desajuste entre entrenamiento y producción; ver sección 4.1.)
2. **Recorte de reposo**: se quitan los cuadros iniciales/finales sin manos.
3. **Normalización**: centro en el punto medio de los hombros y escala por el ancho de hombros →
   invariancia a la posición y a la distancia a la cámara.
4. **Remuestreo** a 64 cuadros e indicadores de presencia de cada mano → tensor (64, 161).
""")
code("""
r = obj[obj.Gloss == "HELLO"].iloc[0]
cr = np.load(ROOT / "data" / "landmarks" / (r["Video file"][:-4] + ".npy")).astype(np.float32)
x = preparar(cr)
rng = np.random.default_rng(SEED)
MANO = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),
        (14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]
POSE = [(5,6),(5,7),(7,9),(6,8),(8,10),(1,2),(3,4)]
def esqueleto(ax, pts, titulo):
    # Tres instantes de la seña: inicio (claro), mitad y final (oscuro).
    validos = [t for t in range(len(pts)) if not np.isnan(pts[t, N_POSE:, 0]).all()]
    for k, t in enumerate([validos[len(validos)//5], validos[len(validos)//2], validos[-len(validos)//5]]):
        p = pts[t]; a = 0.3 + 0.35 * k
        for i, j in POSE:
            ax.plot(p[[i, j], 0], p[[i, j], 1], color="#b4690e", alpha=a, lw=1.5)
        for off in (N_POSE, N_POSE + N_HAND):
            h = p[off:off+N_HAND, :2]
            if np.isnan(h).all() or np.abs(h).max() < 1e-6 or np.abs(h).std() < 1e-3:
                continue   # mano no detectada en ese cuadro
            for i, j in MANO:
                ax.plot(p[[off+i, off+j], 0], p[[off+i, off+j], 1], color="#2f5bd3", alpha=a, lw=1)
    ax.invert_yaxis(); ax.set_aspect("equal"); ax.set_title(titulo)
fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
esqueleto(axs[0], cr, "Crudo (coords. de imagen)")
esqueleto(axs[1], x[:, :N_PTS*3].reshape(64, N_PTS, 3), "Normalizado (centro y escala: hombros)")
esqueleto(axs[2], aumentar(x, rng)[:, :N_PTS*3].reshape(64, N_PTS, 3), "Aumentado (espejo/afín/tiempo)")
plt.suptitle("Seña HELLO en tres instantes (más oscuro = más tarde)")
plt.tight_layout(); plt.savefig(FIG / "preprocesamiento.png"); plt.show()
""")

md("""
## 3. Modelos e iteraciones

| Versión | Idea | Por qué |
|---|---|---|
| Baseline clásico | Regresión logística / k-NN sobre estadísticos | ¿Hace falta deep learning? |
| **v0** | LSTM bidireccional desde cero (Lab 3) | Modelo secuencial base |
| **v1** | Transformer encoder desde cero (HT 2) | La atención relaciona inicio y fin de la seña |
| **v2** | Transformer **preentrenado en 450 señas distintas** + fine-tuning (Lab 6) | Transfer learning: aprender "cómo se mueven las manos" con más datos |

El preentrenamiento usa solo participantes de **entrenamiento** y señas que **no** están entre las
50 objetivo, así que no hay fuga de información hacia prueba. Todas las iteraciones se corren con
3 semillas; el mejor epoch se elige con validación y prueba se evalúa una sola vez.
""")
code("""
if ENTRENAR:
    import subprocess
    subprocess.run([sys.executable, "scripts/03_entrenar.py", "it0_lstm_base", "it1_lstm_aug", "it2_tf_aug",
                    "it3_tf_aug_small", "pre_tf", "pre_lstm", "it4_tl_tf_congelado", "it5_tl_tf_finetune",
                    "it6_tl_lstm_finetune"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "scripts/05_modelo_valor.py"], cwd=ROOT, check=True)
tabla = pd.read_csv(TAB / "tabla_iteraciones.csv")
display(tabla)
""")
code("display(Image(FIG / 'curvas_entrenamiento.png'))")

md("""
## 4. Modelo de valor

**Escenario.** Un programa de enseñanza de ASL con 500 estudiantes activos que hacen ~20
búsquedas de señas desconocidas al mes (10 000 búsquedas/mes). Valoramos el tiempo de un
estudiante en 12 USD/h.

**Función de valor (por búsqueda, en segundos equivalentes):**

* *Sin la app:* `C_manual = t_manual + (1 − p_manual) · c_no_resuelta`.
* *Con la app*, según la confianza `c` (masa de probabilidad del top-3) y el umbral `τ`:
  * `c ≥ τ` → **modo seguro** (3 opciones): `t_grabar + t_rev3` si la seña correcta está en el
    top-3; si no, con prob. `p_engano` el usuario acepta una opción equivocada (`c_error`) y si no,
    revisa 10 opciones (`t_rev10`) y, si tampoco está, busca a mano.
  * `c < τ` → **modo "no estoy seguro"** (10 opciones con aviso): `t_grabar + t_rev10`; el aviso
    baja el engaño a `p_engano_aviso`.
* `Valor neto/mes = (C_manual − E[C_app]) · búsquedas/mes · valor_hora / 3600 − costo_operación`.

El umbral **traduce la confianza del modelo en experiencia de usuario**: mostrar 3 opciones es más
rápido, pero si el modelo se equivoca con seguridad el usuario puede aprender una seña mal.
""")
code("""
display(pd.DataFrame([supuestos_dict()]).T.rename(columns={0: "valor"}))
print(f"Costo esperado de una búsqueda manual: {costo_manual(SUPUESTOS_BASE):.0f} s")
""")
code("""
tv = pd.read_csv(TAB / "tabla_versiones.csv")
display(tv)
resumen = json.loads((TAB / "resumen_valor.json").read_text())
print("IC 95 % de las diferencias de valor (USD/mes):", resumen["ic95_diferencias_valor"])
print("Escenario pesimista:", json.dumps(resumen["escenarios"], indent=1))
""")
code("""
for f in ["valor_vs_umbral.png", "valor_vs_top3.png", "curva_aprendizaje.png", "sensibilidad_tornado.png"]:
    if (FIG / f).exists(): display(Image(FIG / f))
""")
code("display(pd.read_csv(TAB / 'sensibilidad.csv').round(0))")

md("""
### 4.1 Iteración de datos: modo IMAGE vs. modo VIDEO de MediaPipe

A: entrenado y evaluado con extracción IMAGE (lo que se creía). B: el mismo modelo evaluado con
extracción VIDEO, que es lo que produce la app en el navegador. C: re-entrenado con VIDEO.
""")
code("""
if (TAB / "tabla_iteracion_datos.csv").exists():
    display(pd.read_csv(TAB / "tabla_iteracion_datos.csv"))
if (TAB / "prueba_app_e2e.json").exists():
    print(json.dumps(json.loads((TAB / "prueba_app_e2e.json").read_text()), indent=1, ensure_ascii=False))
""")

md("## 5. Explicabilidad, sesgos y robustez")
code("""
for f in ["explicabilidad_oclusion.png", "explicabilidad_saliencia.png", "por_participante.png",
          "senas_dificiles.png", "calibracion.png"]:
    if (FIG / f).exists(): display(Image(FIG / f))
display(pd.read_csv(TAB / "confusiones_top.csv"))
print(json.dumps(json.loads((TAB / "explicabilidad_oclusion.json").read_text()), indent=1, ensure_ascii=False))
""")

md("## 6. Despliegue: latencia del flujo completo")
code("""
# Mide la API real en proceso (sin red): landmarks crudos de un video de prueba -> top-10.
from fastapi.testclient import TestClient
import app.server as srv
cli = TestClient(srv.app)
r = obj[obj.split == "test"].iloc[0]
cr = np.load(ROOT / "data" / "landmarks" / (r["Video file"][:-4] + ".npy")).astype(np.float32)
def a_frames(cr):
    out = []
    for f in cr:
        pose = [[0, 0, 0]] * 33
        pose = np.zeros((33, 3)); pose[[0, 2, 5, 9, 10, 11, 12, 13, 14, 15, 16]] = np.nan_to_num(f[:N_POSE])
        hands = [h.tolist() for h in (f[N_POSE:N_POSE+N_HAND], f[N_POSE+N_HAND:]) if not np.isnan(h).any()]
        out.append({"pose": pose.tolist(), "hands": hands})
    return out
payload = {"frames": a_frames(cr), "modo": "en_vivo"}
lat = {}
for alias in srv.MODELOS:
    payload["modelo"] = alias
    ts = []
    for _ in range(30):
        t0 = time.perf_counter(); res = cli.post("/api/predict", json=payload).json(); ts.append((time.perf_counter() - t0) * 1000)
    lat[alias] = dict(p50_ms=np.median(ts), p95_ms=np.percentile(ts, 95), modelo_ms=res["ms_modelo"],
                      top3=[c["glosa"] for c in res["candidatos"][:3]], real=r.Gloss)
display(pd.DataFrame(lat).T)
""")

md("""
## 7. Conclusiones

Las conclusiones de negocio y técnicas (qué cambio movió más el valor, punto de rendimientos
decrecientes, trade-off complejidad/costo/valor y limitaciones) están desarrolladas en el informe
(`reports/Informe_ProyectoFinal_ASL.docx`) y en `docs/TRASPASO_ESTEBAN.md`.
""")

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
(ROOT / "notebook").mkdir(exist_ok=True)
nbf.write(nb, ROOT / "notebook" / "proyecto_final.ipynb")
print("ok")
