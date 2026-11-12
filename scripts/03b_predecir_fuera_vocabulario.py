"""
Paso 3b — Comportamiento ante señas FUERA del vocabulario (OOV).

Un usuario real a veces hará una seña que el prototipo no conoce. Para medir qué hace cada
modelo en ese caso usamos los videos de VALIDACIÓN y PRUEBA de las 450 señas de preentrenamiento
(participantes que ningún modelo vio). Se guardan las probabilidades sobre las 50 clases:
  reports/runs/<it>_s<seed>_oov_val_probs.npz   (para elegir el umbral)
  reports/runs/<it>_s<seed>_oov_test_probs.npz  (para reportar el valor)

Nota: el preentrenamiento (pre_*) usó esas 450 señas pero SOLO con participantes de
entrenamiento, así que los videos OOV de val/test tampoco los vio el modelo v2.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from asl.data import cargar_grupo  # noqa: E402
from asl.models import construir  # noqa: E402

RUNS = ROOT / "reports" / "runs"
MODELS = ROOT / "models"


def main():
    data, _ = cargar_grupo("preentreno")
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    for js in sorted(RUNS.glob("*.json")):
        r = json.loads(js.read_text())
        it, seed = r["iteracion"], r["seed"]
        if it.startswith(("pre_", "clasico")):
            continue
        dst = RUNS / f"{it}_s{seed}_oov_test_probs.npz"
        if dst.exists():
            continue
        ck = torch.load(MODELS / f"{it}_s{seed}.pt", map_location="cpu")
        m = construir(ck["arq"], len(ck["clases"]), **ck["kw"]); m.load_state_dict(ck["state_dict"])
        m.to(dev).eval()
        for split in ["val", "test"]:
            X, _, parts, _ = data[split]
            out = []
            with torch.no_grad():
                for i in range(0, len(X), 512):
                    out.append(F.softmax(m(torch.from_numpy(X[i:i + 512]).to(dev)), 1).float().cpu())
            P = torch.cat(out).numpy()
            np.savez_compressed(RUNS / f"{it}_s{seed}_oov_{split}_probs.npz", probs=P, participantes=parts)
        print(it, seed, "ok", flush=True)


if __name__ == "__main__":
    main()
