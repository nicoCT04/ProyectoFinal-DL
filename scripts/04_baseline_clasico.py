"""
Paso 4 — Alternativa "más simple" (requisito 4.1): modelos clásicos sin deep learning.

Representación: estadísticos por coordenada sobre el tiempo (media, desviación, mín, máx y
4 cuadros clave) -> vector fijo. Clasificadores: regresión logística y k-NN.
Sirve para justificar si la complejidad de deep learning vale la pena.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from asl.data import cargar_grupo  # noqa: E402

RUNS = ROOT / "reports" / "runs"


def resumen(X):
    clave = X[:, [8, 24, 40, 56]].reshape(len(X), -1)
    return np.concatenate([X.mean(1), X.std(1), X.min(1), X.max(1), clave], 1)


def topk(P, y, k):
    return float((np.argsort(-P, 1)[:, :k] == y[:, None]).any(1).mean())


def main():
    data, clases = cargar_grupo("objetivo")
    (Xtr, ytr), (Xva, yva), (Xte, yte) = [data[s][:2] for s in ["train", "val", "test"]]
    Ftr, Fva, Fte = resumen(Xtr), resumen(Xva), resumen(Xte)
    modelos = {
        "clasico_logreg": make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)),
        "clasico_knn": make_pipeline(StandardScaler(), KNeighborsClassifier(5, weights="distance")),
    }
    RUNS.mkdir(parents=True, exist_ok=True)
    for nombre, m in modelos.items():
        t0 = time.time(); m.fit(Ftr, ytr); t = time.time() - t0
        res = dict(iteracion=nombre, seed=0, tiempo_train_s=t, n_clases=len(clases),
                   params=None, latencia_cpu_ms=None)
        for s, F, y in [("train", Ftr, ytr), ("val", Fva, yva), ("test", Fte, yte)]:
            P = m.predict_proba(F)
            res[s] = dict(top1=topk(P, y, 1), top3=topk(P, y, 3), top5=topk(P, y, 5),
                          f1_macro=float(f1_score(y, P.argmax(1), average="macro")), loss=None)
            if s == "test":
                np.savez_compressed(RUNS / f"{nombre}_s0_test_probs.npz", probs=P, y=y,
                                    participantes=data["test"][2], archivos=data["test"][3])
            if s == "val":
                np.savez_compressed(RUNS / f"{nombre}_s0_val_probs.npz", probs=P, y=y)
        # Comportamiento ante señas fuera del vocabulario (ver 03b).
        if (ROOT / "data" / "cache" / "preentreno.npz").exists():
            oov, _ = cargar_grupo("preentreno")
            for s in ["val", "test"]:
                np.savez_compressed(RUNS / f"{nombre}_s0_oov_{s}_probs.npz",
                                    probs=m.predict_proba(resumen(oov[s][0])), participantes=oov[s][2])
        x1 = Fte[:1]; t0 = time.perf_counter()
        for _ in range(200):
            m.predict_proba(x1)
        res["latencia_cpu_ms"] = (time.perf_counter() - t0) / 200 * 1000
        (RUNS / f"{nombre}_s0.json").write_text(json.dumps(res, indent=2))
        print(nombre, {s: round(res[s]["top1"], 3) for s in ["train", "val", "test"]},
              "top3 test", round(res["test"]["top3"], 3))


if __name__ == "__main__":
    main()
