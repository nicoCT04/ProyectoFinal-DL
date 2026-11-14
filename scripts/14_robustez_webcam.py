"""
Robustez ante fallas típicas de la cámara en vivo, medida sobre los landmarks CRUDOS del conjunto de
prueba (antes de preparar): seña cortada por la segmentación, manos perdidas, menos FPS, hombros sin
detectar, cámara más cerca (manos que salen del cuadro) y reposo extra al final.

Compara las iteraciones indicadas (media de sus semillas). Motivó la iteración it7 (aumentos de cámara).

Uso:  python scripts/14_robustez_webcam.py [it5_tl_tf_finetune it7_tl_tf_webcam]
Salida: reports/tablas/robustez_webcam.csv
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
from asl.features import N_POSE, P_LSH, P_RSH, preparar  # noqa: E402
from asl.models import construir  # noqa: E402


def active_span(s):
    idx = np.where(~np.isnan(s[:, N_POSE:, 0]).all(1))[0]
    return (idx[0], idx[-1] + 1) if len(idx) else (0, len(s))


def cut(frac, keep_start):
    def f(s):
        a, b = active_span(s)
        k = max(int((b - a) * frac), 2)
        return s[a:a + k] if keep_start else s[b - k:b]
    return f


def drop_hands(p, rng):
    def f(s):
        s = s.copy()
        s[rng.random(len(s)) < p, N_POSE:] = np.nan
        return s
    return f


def no_shoulders(s):
    s = s.copy()
    s[:, [P_LSH, P_RSH]] = np.nan
    return s


def closer(z):
    def f(s):
        s = s.copy()
        s[..., :2] = (s[..., :2] - 0.5) * z + 0.5
        out = (s[..., 0] < 0) | (s[..., 0] > 1) | (s[..., 1] < 0) | (s[..., 1] > 1)
        for side in (0, 1):
            hand = slice(N_POSE + side * 21, N_POSE + (side + 1) * 21)
            s[out[:, hand].mean(1) > 0.3, hand] = np.nan
        return s
    return f


def rest_tail(k):
    def f(s):
        _, b = active_span(s)
        last = s[b - 1:b].copy()
        last[..., 1] += 0.15
        return np.concatenate([s[:b], np.repeat(last, k, 0)])
    return f


def main():
    its = sys.argv[1:] or ["it5_tl_tf_finetune", "it7_tl_tf_webcam"]
    rng = np.random.default_rng(0)
    voc = json.loads((ROOT / "data" / "vocabulario.json").read_text())
    c2i = {c: i for i, c in enumerate(voc["objetivo"])}
    sub = pd.read_csv(ROOT / "data" / "subconjunto.csv")
    sub = sub[(sub.grupo == "objetivo") & (sub.split == "test")]
    raw = [np.load(ROOT / "data" / "landmarks" / (f[:-4] + ".npy")).astype(np.float32) for f in sub["Video file"]]
    y = np.array([c2i[g] for g in sub.Gloss])
    perturbations = {
        "original": lambda s: s,
        "seña cortada: primer 60 %": cut(0.6, True), "seña cortada: último 60 %": cut(0.6, False),
        "seña cortada: primer 80 %": cut(0.8, True),
        "manos perdidas en 20 % de cuadros": drop_hands(0.2, rng), "manos perdidas en 40 % de cuadros": drop_hands(0.4, rng),
        "15 fps": lambda s: s[::2], "hombros sin detectar": no_shoulders,
        "cámara 1.4× más cerca": closer(1.4), "15 cuadros de reposo extra al final": rest_tail(15),
    }
    models = {}
    for it in its:
        for ck_path in sorted((ROOT / "models").glob(f"{it}_s*.pt")):
            ck = torch.load(ck_path, map_location="cpu")
            m = construir(ck["arq"], len(ck["clases"]), **ck["kw"])
            m.load_state_dict(ck["state_dict"]); m.eval()
            models.setdefault(it, []).append(m)
    rows = []
    with np.errstate(all="ignore"):
        for name, fn in perturbations.items():
            X = torch.from_numpy(np.stack([preparar(fn(s)) for s in raw]))
            row = {"perturbación": name}
            for it, ms in models.items():
                t1, t3 = [], []
                for m in ms:
                    with torch.no_grad():
                        order = np.argsort(-F.softmax(m(X), 1).numpy(), 1)
                    t1.append((order[:, 0] == y).mean()); t3.append((order[:, :3] == y[:, None]).any(1).mean())
                row[f"{it} top1"] = round(float(np.mean(t1)), 3)
                row[f"{it} top3"] = round(float(np.mean(t3)), 3)
            rows.append(row)
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(ROOT / "reports" / "tablas" / "robustez_webcam.csv", index=False)


if __name__ == "__main__":
    main()
