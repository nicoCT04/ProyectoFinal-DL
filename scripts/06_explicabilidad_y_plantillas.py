"""
Paso 6 — Explicabilidad del modelo desplegado y plantillas de ejemplo para la app.

  * Oclusión por parte del cuerpo: se "apaga" (pone en reposo) cada grupo de puntos en prueba
    y se mide cuánto cae el top-3. Revela en qué se fija el modelo (mano dominante, no dominante,
    cara, brazos).
  * Saliencia por gradiente (|∂logit/∂x|) agregada por articulación y por cuadro.
  * Atención del token [CLS] sobre el tiempo (solo Transformer).
  * Robustez: espejo (persona zurda), ruido y velocidad distinta.
  * Plantillas: para cada seña, la secuencia de entrenamiento más "típica" (medoide), que la app
    dibuja como esqueleto animado. Se guarda en models/plantillas.npz (derivada del dataset:
    uso local, no se publica).
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from asl.data import cargar_grupo, reflejar  # noqa: E402
from asl.features import N_HAND, N_POSE, N_PTS, T_FIJO  # noqa: E402
from asl.models import construir  # noqa: E402

FIG = ROOT / "reports" / "figuras"; TAB = ROOT / "reports" / "tablas"; MODELS = ROOT / "models"
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

GRUPOS = {
    "cara (nariz, ojos, boca)": list(range(0, 5)),
    "hombros y codos": [5, 6, 7, 8],
    "muñecas (pose)": [9, 10],
    "mano izquierda": list(range(N_POSE, N_POSE + N_HAND)),
    "mano derecha": list(range(N_POSE + N_HAND, N_PTS)),
}


def cargar(alias):
    cfg = json.loads((MODELS / "despliegue.json").read_text())
    info = cfg["modelos"][alias]
    ck = torch.load(MODELS / f"{info['archivo']}.pt", map_location="cpu")
    m = construir(ck["arq"], len(ck["clases"]), **ck["kw"]); m.load_state_dict(ck["state_dict"]); m.eval()
    return m, ck, info


def top3(m, X):
    with torch.no_grad():
        P = F.softmax(m(torch.from_numpy(X)), 1).numpy()
    return P


def acc3(P, y):
    return float((np.argsort(-P, 1)[:, :3] == y[:, None]).any(1).mean())


def ocluir(X, puntos):
    X = X.copy()
    T = X.shape[1]
    pts = X[:, :, : N_PTS * 3].reshape(len(X), T, N_PTS, 3)
    pts[:, :, puntos] = 0.0
    X[:, :, : N_PTS * 3] = pts.reshape(len(X), T, -1)
    if any(p >= N_POSE for p in puntos):
        lado = 0 if puntos[0] < N_POSE + N_HAND else 1
        X[:, :, N_PTS * 3 + lado] = 0.0
    return X


def main():
    data, clases = cargar_grupo("objetivo")
    Xte, yte = data["test"][:2]
    cfg = json.loads((MODELS / "despliegue.json").read_text())
    res = {}
    for alias in cfg["modelos"]:
        m, ck, info = cargar(alias)
        base = acc3(top3(m, Xte), yte)
        r = {"base_top3": base}
        for nombre, pts in GRUPOS.items():
            r[f"sin {nombre}"] = acc3(top3(m, ocluir(Xte, pts)), yte)
        # Robustez
        r["espejo (zurdo)"] = acc3(top3(m, np.stack([reflejar(x) for x in Xte])), yte)
        rng = np.random.default_rng(0)
        r["ruido σ=0.05"] = acc3(top3(m, (Xte + rng.normal(0, 0.05, Xte.shape) *
                                         (np.arange(Xte.shape[2]) < N_PTS * 3)).astype(np.float32)), yte)
        idx_lento = np.clip(np.round(np.linspace(0, T_FIJO * 0.8, T_FIJO)).astype(int), 0, T_FIJO - 1)
        r["seña cortada (80 %)"] = acc3(top3(m, Xte[:, idx_lento]), yte)
        res[alias] = r
    (TAB / "explicabilidad_oclusion.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=1, ensure_ascii=False))

    dep = cfg["default"]
    r = res[dep]
    claves = [k for k in r if k != "base_top3"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh(claves[::-1], [r[k] for k in claves][::-1], color="#2f5bd3", alpha=0.85)
    ax.axvline(r["base_top3"], color="black", lw=1)
    ax.text(r["base_top3"], len(claves) - 0.4, f" sin cambios: {r['base_top3']:.2f}", fontsize=8)
    ax.set_xlim(0, 1); ax.set_xlabel("Top-3 en prueba")
    ax.set_title(f"Oclusión y robustez ({dep}): ¿qué información usa el modelo?")
    fig.tight_layout(); fig.savefig(FIG / "explicabilidad_oclusion.png"); plt.close(fig)

    # Saliencia por gradiente: importancia por punto y por cuadro.
    m, ck, info = cargar(dep)
    x = torch.from_numpy(Xte).requires_grad_(True)
    logit = m(x).gather(1, torch.from_numpy(yte)[:, None]).sum()
    logit.backward()
    g = x.grad.abs().numpy()[:, :, : N_PTS * 3].reshape(len(Xte), T_FIJO, N_PTS, 3).sum(-1)  # (N, T, 53)
    por_punto = g.mean((0, 1)); por_t = g.sum(2).mean(0)
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.2))
    cols = ["#b4690e"] * N_POSE + ["#2f5bd3"] * N_HAND + ["#1f8a4c"] * N_HAND
    axs[0].bar(range(N_PTS), por_punto / por_punto.sum(), color=cols)
    axs[0].set_title("Saliencia por punto (naranja pose · azul mano izq. · verde mano der.)", fontsize=9)
    axs[0].set_xlabel("índice de punto")
    axs[1].plot(np.arange(T_FIJO), por_t / por_t.sum(), color="#2f5bd3", label="gradiente")
    if hasattr(m, "atencion_cls"):
        with torch.no_grad():
            att = m.atencion_cls(torch.from_numpy(Xte)).mean(0).numpy()
        axs[1].plot(np.arange(T_FIJO), att / att.sum(), color="#1f8a4c", label="atención [CLS] (última capa)")
    axs[1].set_title("Importancia a lo largo de la seña", fontsize=9); axs[1].set_xlabel("cuadro (remuestreado)")
    axs[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "explicabilidad_saliencia.png"); plt.close(fig)

    # Plantillas (medoide de entrenamiento por clase).
    Xtr, ytr = data["train"][:2]
    plant = {}
    for c, nombre in enumerate(clases):
        S = Xtr[ytr == c][:, :, : N_PTS * 3]
        med = S.mean(0)
        i = np.argmin(((S - med) ** 2).sum((1, 2)))
        plant[nombre] = S[i].reshape(T_FIJO, N_PTS, 3)[:, :, :2].astype(np.float32)
    np.savez_compressed(MODELS / "plantillas.npz", **plant)
    print("plantillas:", len(plant))


if __name__ == "__main__":
    main()
