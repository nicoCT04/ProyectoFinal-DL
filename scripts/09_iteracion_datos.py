"""
Paso 9 — Iteración de DATOS: extracción de MediaPipe en modo IMAGE vs. modo VIDEO.

Primera versión: los landmarks de entrenamiento se extrajeron en modo IMAGE (cada cuadro
independiente), pero la app usa modo VIDEO (con seguimiento) en el navegador. Al probar la app
con videos de prueba, el acierto bajó respecto a lo medido offline. Aquí se cuantifica:

  A. modelo entrenado con IMAGE, evaluado con IMAGE      (lo que se creía que rendía)
  B. modelo entrenado con IMAGE, evaluado con VIDEO      (lo que realmente veía la app)
  C. modelo entrenado con VIDEO, evaluado con VIDEO      (después de corregir)

Salida: reports/tablas/tabla_iteracion_datos.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from asl.models import construir  # noqa: E402

TAB = ROOT / "reports" / "tablas"
VIEJO = ROOT / "reports" / "iteracion_datos_image"


def cargar_cache(ruta):
    z = np.load(ruta, allow_pickle=True)
    return z["X_test"], z["y_test"]


def evaluar(pt, X, y):
    ck = torch.load(pt, map_location="cpu")
    m = construir(ck["arq"], len(ck["clases"]), **ck["kw"]); m.load_state_dict(ck["state_dict"]); m.eval()
    with torch.no_grad():
        P = F.softmax(torch.cat([m(torch.from_numpy(X[i:i + 512])) for i in range(0, len(X), 512)]), 1).numpy()
    top = np.argsort(-P, 1)
    return float((top[:, 0] == y).mean()), float((top[:, :3] == y[:, None]).any(1).mean())


def main():
    Xi, yi = cargar_cache(ROOT / "data" / "cache_image" / "objetivo.npz")
    Xv, yv = cargar_cache(ROOT / "data" / "cache" / "objetivo.npz")
    filas = []
    for it in ["it0_lstm_base", "it1_lstm_aug", "it2_tf_aug", "it3_tf_aug_small"]:
        for s in [0, 1, 2]:
            vieja = ROOT / "models" / "image_mode" / f"{it}_s{s}.pt"
            nueva = ROOT / "models" / f"{it}_s{s}.pt"
            if not (vieja.exists() and nueva.exists()):
                continue
            a = evaluar(vieja, Xi, yi); b = evaluar(vieja, Xv, yv); c = evaluar(nueva, Xv, yv)
            filas.append(dict(iteracion=it, seed=s, A_top1=a[0], A_top3=a[1], B_top1=b[0], B_top3=b[1],
                              C_top1=c[0], C_top3=c[1]))
    df = pd.DataFrame(filas)
    g = df.groupby("iteracion").agg(["mean", "std"]).drop(columns="seed")
    out = pd.DataFrame({
        "Iteración": g.index,
        "A: entrena IMAGE / evalúa IMAGE (top-1 | top-3)": [f"{g.loc[i, ('A_top1', 'mean')]:.3f} | {g.loc[i, ('A_top3', 'mean')]:.3f}" for i in g.index],
        "B: entrena IMAGE / evalúa VIDEO = app (top-1 | top-3)": [f"{g.loc[i, ('B_top1', 'mean')]:.3f} | {g.loc[i, ('B_top3', 'mean')]:.3f}" for i in g.index],
        "C: entrena VIDEO / evalúa VIDEO (top-1 | top-3)": [f"{g.loc[i, ('C_top1', 'mean')]:.3f} | {g.loc[i, ('C_top3', 'mean')]:.3f}" for i in g.index],
    })
    out.to_csv(TAB / "tabla_iteracion_datos.csv", index=False)
    df.to_csv(TAB / "iteracion_datos_detalle.csv", index=False)
    print(out.to_string())
    # Valor de negocio de la versión desplegada en cada escenario (sin OOV, para comparar solo el efecto de los datos).
    from asl.valor import valor
    res = {}
    for it in ["it3_tf_aug_small", "it2_tf_aug"]:
        sub = df[df.iteracion == it]
        if len(sub):
            res[it] = {k: float(sub[k].mean()) for k in ["A_top3", "B_top3", "C_top3", "A_top1", "B_top1", "C_top1"]}
    (TAB / "iteracion_datos_resumen.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
