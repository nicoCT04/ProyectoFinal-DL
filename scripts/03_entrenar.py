"""
Paso 3 — Entrenamiento y evaluación de las iteraciones del modelo.

Cada iteración se define en ITERACIONES y se entrena con varias semillas (para reportar
media ± desviación). La selección del mejor epoch se hace SOLO con validación; el conjunto
de prueba (participantes nunca vistos) se evalúa una vez con el checkpoint elegido.

Salidas por iteración y semilla:
  models/<it>_s<seed>.pt              pesos + configuración
  reports/runs/<it>_s<seed>.json      métricas (train/val/test), tiempo, parámetros, latencia
  reports/runs/<it>_s<seed>_hist.csv  curvas de entrenamiento
  reports/runs/<it>_s<seed>_test_probs.npz  probabilidades en prueba (para el modelo de valor)

Uso:  python scripts/03_entrenar.py it0_lstm_base it1_lstm_aug ...  [--seeds 0 1 2]
"""
import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from asl.data import SenasDataset, cargar_grupo  # noqa: E402
from asl.features import N_FEAT, T_FIJO  # noqa: E402
from asl.models import construir  # noqa: E402

RUNS = ROOT / "reports" / "runs"
MODELS = ROOT / "models"

BASE = dict(grupo="objetivo", arq="lstm", kw={}, aug=False, epochs=120, lr=1e-3, wd=1e-4,
            ls=0.0, bs=64, warmup=5, init=None, congelar=False, lr_backbone=None, paciencia=40, frac=1.0)

ITERACIONES = {
    # ---- v0: LSTM desde cero
    "it0_lstm_base": dict(arq="lstm"),
    "it1_lstm_aug": dict(arq="lstm", aug=True, ls=0.1, epochs=200, paciencia=60),
    # ---- v1: Transformer desde cero
    "it2_tf_aug": dict(arq="transformer", aug=True, ls=0.1, epochs=250, lr=5e-4, wd=0.05, paciencia=80),
    "it3_tf_aug_small": dict(arq="transformer", kw=dict(d=128, capas=3, heads=4, dropout=0.3),
                             aug=True, ls=0.1, epochs=250, lr=5e-4, wd=0.05, paciencia=80),
    # ---- Preentrenamiento (450 señas distintas, solo participantes de entrenamiento)
    "pre_tf": dict(grupo="preentreno", arq="transformer", aug=True, ls=0.1, epochs=70, lr=5e-4,
                   wd=0.05, bs=128, paciencia=20),
    "pre_lstm": dict(grupo="preentreno", arq="lstm", aug=True, ls=0.1, epochs=60, lr=1e-3,
                     bs=128, paciencia=20),
    # ---- v2: transfer learning sobre las 50 señas objetivo
    "it4_tl_tf_congelado": dict(arq="transformer", aug=True, ls=0.1, epochs=150, lr=1e-3,
                                init="pre_tf", congelar=True, paciencia=50),
    "it5_tl_tf_finetune": dict(arq="transformer", aug=True, ls=0.1, epochs=200, lr=5e-4, wd=0.05,
                               init="pre_tf", lr_backbone=1e-4, paciencia=60),
    "it6_tl_lstm_finetune": dict(arq="lstm", aug=True, ls=0.1, epochs=200, lr=1e-3,
                                 init="pre_lstm", lr_backbone=3e-4, paciencia=60),
    # ---- v3: v2 + aumentos que imitan la cámara en vivo (grabación cortada, encuadre, mano perdida)
    "it7_tl_tf_webcam": dict(arq="transformer", aug="webcam", ls=0.1, epochs=200, lr=5e-4, wd=0.05,
                             init="pre_tf", lr_backbone=1e-4, paciencia=60),
}
# Curva de aprendizaje (valor vs. cantidad de datos) para v1 y v2.
for _f in [0.25, 0.5, 0.75]:
    ITERACIONES[f"lc_tf_f{int(_f*100)}"] = {**ITERACIONES["it2_tf_aug"], "frac": _f}
    ITERACIONES[f"lc_tl_f{int(_f*100)}"] = {**ITERACIONES["it5_tl_tf_finetune"], "frac": _f}


def semilla_worker(_):
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(info.seed % 2**32)


def dispositivo():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def predecir(model, X, dev, bs=256):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        out.append(model(torch.from_numpy(X[i:i + bs]).to(dev)).float().cpu())
    return torch.cat(out)


def metricas(logits: torch.Tensor, y: np.ndarray) -> dict:
    y_t = torch.from_numpy(y)
    top = logits.topk(5, 1).indices
    pred = top[:, 0].numpy()
    return dict(
        loss=float(F.cross_entropy(logits, y_t)),
        top1=float((top[:, 0] == y_t).float().mean()),
        top3=float((top[:, :3] == y_t[:, None]).any(1).float().mean()),
        top5=float((top == y_t[:, None]).any(1).float().mean()),
        f1_macro=float(f1_score(y, pred, average="macro")),
    )


def latencia_cpu(model, n=200):
    """Milisegundos por seña en CPU (batch 1), como en el servidor de la app."""
    m = copy.deepcopy(model).cpu().eval()
    x = torch.randn(1, T_FIJO, N_FEAT)
    with torch.no_grad():
        for _ in range(20):
            m(x)
        t0 = time.perf_counter()
        for _ in range(n):
            m(x)
    return (time.perf_counter() - t0) / n * 1000


def entrenar(nombre: str, seed: int, dev):
    cfg = {**BASE, **ITERACIONES[nombre]}
    torch.manual_seed(seed); np.random.seed(seed)
    data, clases = cargar_grupo(cfg["grupo"])
    Xtr, ytr = data["train"][:2]
    Xva, yva = data["val"][:2]
    Xte, yte = data["test"][:2]
    if cfg["frac"] < 1.0:
        # Curva de aprendizaje: se usa una fracción de los PARTICIPANTES de entrenamiento
        # (simula tener menos personas grabando señas).
        parts = np.unique(data["train"][2])
        rng = np.random.default_rng(seed)
        elegidos = rng.choice(parts, max(1, int(round(cfg["frac"] * len(parts)))), replace=False)
        m = np.isin(data["train"][2], elegidos)
        Xtr, ytr = Xtr[m], ytr[m]
    model = construir(cfg["arq"], len(clases), **cfg["kw"]).to(dev)

    if cfg["init"]:
        # Transfer learning: se copia todo el backbone preentrenado; la cabeza es nueva (50 clases).
        ck = torch.load(MODELS / f"{cfg['init']}_s{seed}.pt", map_location="cpu")
        sd = {k: v for k, v in ck["state_dict"].items() if not k.startswith("head.")}
        faltan = model.load_state_dict(sd, strict=False)
        assert set(faltan.missing_keys) <= {"head.weight", "head.bias"}, faltan
    backbone = [p for n, p in model.named_parameters() if not n.startswith("head.")]
    head = [p for n, p in model.named_parameters() if n.startswith("head.")]
    if cfg["congelar"]:
        for p in backbone:
            p.requires_grad_(False)
        grupos = [dict(params=head, lr=cfg["lr"])]
    elif cfg["lr_backbone"]:
        grupos = [dict(params=backbone, lr=cfg["lr_backbone"]), dict(params=head, lr=cfg["lr"])]
    else:
        grupos = [dict(params=list(model.parameters()), lr=cfg["lr"])]
    opt = torch.optim.AdamW(grupos, weight_decay=cfg["wd"])
    lrs0 = [g["lr"] for g in opt.param_groups]

    # Con muchos datos (preentrenamiento) los aumentos en numpy son el cuello de botella:
    # se reparten en procesos auxiliares, cada uno con su propia semilla.
    nw = 6 if len(ytr) > 3000 else 0
    dl = DataLoader(SenasDataset(Xtr, ytr, aug=cfg["aug"], seed=seed), batch_size=cfg["bs"],
                    shuffle=True, drop_last=len(ytr) > cfg["bs"], num_workers=nw,
                    persistent_workers=nw > 0, worker_init_fn=semilla_worker if nw else None)
    pasos_ep = len(dl)
    total = cfg["epochs"] * pasos_ep
    hist, mejor, mejor_ep, sin_mejora = [], None, -1, 0
    t0 = time.time()
    paso = 0
    for ep in range(cfg["epochs"]):
        model.train()
        if cfg["congelar"]:
            model.eval()  # sin dropout en el backbone congelado
        tl, n = 0.0, 0
        for xb, yb in dl:
            # Calentamiento lineal + decaimiento coseno de la tasa de aprendizaje.
            fr = paso / max(cfg["warmup"] * pasos_ep, 1)
            f = fr if fr < 1 else 0.5 * (1 + math.cos(math.pi * (paso - cfg["warmup"] * pasos_ep) /
                                                       max(total - cfg["warmup"] * pasos_ep, 1)))
            for g, l0 in zip(opt.param_groups, lrs0):
                g["lr"] = l0 * f
            xb, yb = xb.to(dev), yb.to(dev)
            loss = F.cross_entropy(model(xb), yb, label_smoothing=cfg["ls"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item() * len(yb); n += len(yb); paso += 1
        mv = metricas(predecir(model, Xva, dev), yva)
        hist.append(dict(epoch=ep + 1, train_loss=tl / n, val_loss=mv["loss"], val_top1=mv["top1"],
                         val_top3=mv["top3"], val_f1=mv["f1_macro"]))
        # Criterio de selección: top-1 de validación (desempate por pérdida).
        clave = (mv["top1"], -mv["loss"])
        if mejor is None or clave > mejor[0]:
            mejor = (clave, copy.deepcopy(model.state_dict())); mejor_ep = ep + 1; sin_mejora = 0
        else:
            sin_mejora += 1
        if (ep + 1) % (10 if cfg["grupo"] == "preentreno" else 20) == 0:
            print(f"  {nombre} s{seed} ep{ep+1}: train_loss={tl/n:.3f} val_top1={mv['top1']:.3f} "
                  f"val_top3={mv['top3']:.3f}", flush=True)
        if sin_mejora >= cfg["paciencia"]:
            break
    t_train = time.time() - t0
    model.load_state_dict(mejor[1])

    res = dict(iteracion=nombre, seed=seed, config={k: v for k, v in cfg.items()},
               n_clases=len(clases), n_train=len(ytr), n_val=len(yva), n_test=len(yte),
               mejor_epoch=mejor_ep, epochs_corridos=len(hist), tiempo_train_s=t_train,
               dispositivo=str(dev),
               params=sum(p.numel() for p in model.parameters()),
               params_entrenables=sum(p.numel() for p in model.parameters() if p.requires_grad),
               latencia_cpu_ms=latencia_cpu(model))
    for s, X, y in [("train", Xtr, ytr), ("val", Xva, yva), ("test", Xte, yte)]:
        lg = predecir(model, X, dev)
        res[s] = metricas(lg, y)
        if s == "test":
            np.savez_compressed(RUNS / f"{nombre}_s{seed}_test_probs.npz",
                                probs=F.softmax(lg, 1).numpy(), y=y, participantes=data["test"][2],
                                archivos=data["test"][3])
        if s == "val":
            np.savez_compressed(RUNS / f"{nombre}_s{seed}_val_probs.npz",
                                probs=F.softmax(lg, 1).numpy(), y=y)
    torch.save(dict(state_dict=model.state_dict(), arq=cfg["arq"], kw=cfg["kw"], clases=clases,
                    iteracion=nombre, seed=seed), MODELS / f"{nombre}_s{seed}.pt")
    pd.DataFrame(hist).to_csv(RUNS / f"{nombre}_s{seed}_hist.csv", index=False)
    (RUNS / f"{nombre}_s{seed}.json").write_text(json.dumps(res, indent=2))
    print(f"== {nombre} s{seed}: val_top1={res['val']['top1']:.3f} test_top1={res['test']['top1']:.3f} "
          f"test_top3={res['test']['top3']:.3f} f1={res['test']['f1_macro']:.3f} "
          f"({t_train/60:.1f} min, best ep {mejor_ep})", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("iteraciones", nargs="+")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    RUNS.mkdir(parents=True, exist_ok=True); MODELS.mkdir(exist_ok=True)
    dev = dispositivo()
    print("Dispositivo:", dev, flush=True)
    for it in args.iteraciones:
        for s in args.seeds:
            entrenar(it, s, dev)


if __name__ == "__main__":
    main()
