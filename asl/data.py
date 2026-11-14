"""Carga del dataset de landmarks y aumentos de datos."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .features import N_POSE, N_HAND, N_PTS, T_FIJO, preparar

ROOT = Path(__file__).resolve().parents[1]
LMK = ROOT / "data" / "landmarks"
CACHE = ROOT / "data" / "cache"

# Pares de puntos de pose que se intercambian al reflejar horizontalmente
# (índices dentro de POSE_IDX: ojos, boca, hombros, codos, muñecas).
_PARES_POSE = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)]


def vocabulario():
    return json.loads((ROOT / "data" / "vocabulario.json").read_text())


def cargar_grupo(grupo: str, usar_cache: bool = True):
    """Devuelve dict split -> (X [N, T, F], y [N], participantes [N], archivos [N]) y la lista de clases."""
    CACHE.mkdir(parents=True, exist_ok=True)
    voc = vocabulario()
    clases = voc["objetivo"] if grupo == "objetivo" else voc["preentreno"]
    ruta = CACHE / f"{grupo}.npz"
    if usar_cache and ruta.exists():
        z = np.load(ruta, allow_pickle=True)
        out = {s: (z[f"X_{s}"], z[f"y_{s}"], z[f"p_{s}"], z[f"f_{s}"]) for s in ["train", "val", "test"]}
        return out, clases
    sub = pd.read_csv(ROOT / "data" / "subconjunto.csv")
    sub = sub[sub["grupo"] == grupo]
    c2i = {c: i for i, c in enumerate(clases)}
    out, guardar = {}, {}
    for s in ["train", "val", "test"]:
        d = sub[sub["split"] == s]
        X, y, p, f = [], [], [], []
        for _, r in d.iterrows():
            npy = LMK / (r["Video file"][:-4] + ".npy")
            if not npy.exists():
                continue
            X.append(preparar(np.load(npy).astype(np.float32)))
            y.append(c2i[r["Gloss"]]); p.append(r["Participant ID"]); f.append(r["Video file"])
        out[s] = (np.stack(X), np.array(y), np.array(p), np.array(f))
        for k, v in zip("Xypf", out[s]):
            guardar[f"{k}_{s}"] = v
    np.savez_compressed(ruta, **guardar)
    return out, clases


# ----------------------------------------------------------------------------- aumentos
def reflejar(x: np.ndarray) -> np.ndarray:
    """Espejo horizontal: simula a una persona zurda (o una cámara reflejada)."""
    T = x.shape[0]
    pts = x[:, : N_PTS * 3].reshape(T, N_PTS, 3).copy()
    pres = x[:, N_PTS * 3:].copy()
    pts[..., 0] *= -1
    for a, b in _PARES_POSE:
        pts[:, [a, b]] = pts[:, [b, a]]
    izq = pts[:, N_POSE:N_POSE + N_HAND].copy()
    pts[:, N_POSE:N_POSE + N_HAND] = pts[:, N_POSE + N_HAND:]
    pts[:, N_POSE + N_HAND:] = izq
    pres = pres[:, ::-1]
    return np.concatenate([pts.reshape(T, -1), pres], axis=1)


def aumentar(x: np.ndarray, rng: np.random.Generator, webcam: bool = False) -> np.ndarray:
    T = x.shape[0]
    if rng.random() < 0.5:
        x = reflejar(x)
    pts = x[:, : N_PTS * 3].reshape(T, N_PTS, 3).copy()
    pres = x[:, N_PTS * 3:].copy()
    # Transformación afín 2D: rotación, escala y traslación pequeñas (variación de cámara/persona).
    ang = np.deg2rad(rng.uniform(-15, 15))
    s = rng.uniform(0.85, 1.15)
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]) * s
    pts[..., :2] = pts[..., :2] @ R.T + rng.uniform(-0.15, 0.15, size=2)
    pts[..., 2] *= s
    # Deformación temporal: recorte aleatorio y re-interpolación (velocidad de ejecución distinta).
    a = rng.uniform(0, 0.15) * (T - 1)
    b = (T - 1) - rng.uniform(0, 0.15) * (T - 1)
    pos = np.clip(np.linspace(a, b, T) + rng.normal(0, 0.3, T), 0, T - 1)
    pos.sort()
    i0 = np.floor(pos).astype(int); i1 = np.minimum(i0 + 1, T - 1); w = (pos - i0)[:, None, None]
    pts = (1 - w) * pts[i0] + w * pts[i1]
    pres = pres[np.round(pos).astype(int)]
    # Ruido gaussiano (jitter del detector) y "apagado" de cuadros completos.
    pts += rng.normal(0, 0.01, pts.shape)
    drop = rng.random(T) < 0.05
    pts[drop] = pts[np.maximum(np.where(drop)[0] - 1, 0)]
    if webcam:
        pts, pres = webcam_augment(pts, pres, rng)
    return np.concatenate([pts.reshape(T, -1), pres], axis=1).astype(np.float32)


def webcam_augment(pts: np.ndarray, pres: np.ndarray, rng: np.random.Generator):
    """Fallas típicas de la cámara en vivo, medidas con la app: grabación cortada por un lado,
    hombros mal estimados (encuadre cercano) y la mano perdida varios cuadros seguidos."""
    T = pts.shape[0]
    if rng.random() < 0.5:
        keep = rng.uniform(0.6, 1.0) * (T - 1)
        a = 0.0 if rng.random() < 0.5 else (T - 1) - keep
        pos = np.linspace(a, a + keep, T)
        i0 = np.floor(pos).astype(int); i1 = np.minimum(i0 + 1, T - 1); w = (pos - i0)[:, None, None]
        pts = (1 - w) * pts[i0] + w * pts[i1]
        pres = pres[np.round(pos).astype(int)]
    if rng.random() < 0.3:
        pts[..., :2] = pts[..., :2] * rng.uniform(0.7, 1.3) + rng.uniform(-0.4, 0.4, size=2)
    if rng.random() < 0.3:
        side = rng.integers(2)
        hand = slice(N_POSE + side * N_HAND, N_POSE + (side + 1) * N_HAND)
        start = rng.integers(0, T - 4); end = min(T, start + rng.integers(4, 16))
        pts[start:end, hand] = pts[max(start - 1, 0), hand]
        pres[start:end, side] = 0
    return pts, pres


class SenasDataset(Dataset):
    def __init__(self, X, y, aug: bool = False, seed: int = 0):
        self.X, self.y, self.aug = X, y, aug
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.X[i]
        if self.aug:
            x = aumentar(x, self.rng, webcam=self.aug == "webcam")
        return torch.from_numpy(np.ascontiguousarray(x)), int(self.y[i])
