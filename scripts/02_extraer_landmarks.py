"""
Paso 2 — Extracción de landmarks con MediaPipe (HandLandmarker + PoseLandmarker).

Por cada video del subconjunto se guarda data/landmarks/<video>.npy con forma (T, 53, 3)
(T = cuadros del video, NaN donde no se detectó el punto). Se paraleliza por procesos;
cada proceso crea sus propios detectores (asl/extraccion.py, el mismo código que usa la API).
Es reanudable: salta los .npy existentes.

Uso:  GLOG_minloglevel=2 python scripts/02_extraer_landmarks.py [--procesos 10] [--grupo objetivo]
Tiempo: ≈ 1 h para 15 300 videos con 9–10 procesos en un Apple M4 Pro.
"""
import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
VIDEOS = ROOT / "data" / "raw" / "ASL_Citizen" / "videos"
OUT = ROOT / "data" / "landmarks"


def trabajo(nombre: str):
    from asl.extraccion import extraer  # cada proceso carga sus detectores una sola vez

    dst = OUT / (nombre[:-4] + ".npy")
    if dst.exists():
        return nombre, 0.0, -1
    t0 = time.time()
    try:
        arr = extraer(VIDEOS / nombre)
    except Exception as e:  # video corrupto o ilegible
        print("ERROR", nombre, e, flush=True)
        return nombre, 0.0, -2
    np.save(dst, arr.astype(np.float16))
    return nombre, time.time() - t0, len(arr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procesos", type=int, default=10)
    ap.add_argument("--grupo", default=None, help="objetivo | preentreno | (todos)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    sub = pd.read_csv(ROOT / "data" / "subconjunto.csv")
    if args.grupo:
        sub = sub[sub["grupo"] == args.grupo]
    # Primero las señas objetivo, para poder entrenar v0/v1 cuanto antes.
    sub = sub.sort_values("grupo", key=lambda s: s != "objetivo")
    nombres = [f for f in sub["Video file"] if (VIDEOS / f).exists()]
    print(f"{len(nombres)} videos disponibles de {len(sub)}", flush=True)

    t0, hechos = time.time(), 0
    with Pool(args.procesos) as pool:
        for i, (n, dt, T) in enumerate(pool.imap_unordered(trabajo, nombres, chunksize=4), 1):
            hechos += T > 0
            if i % 250 == 0 or i == len(nombres):
                print(f"[{i}/{len(nombres)}] {(time.time() - t0) / 60:.1f} min, nuevos={hechos}", flush=True)


if __name__ == "__main__":
    main()
