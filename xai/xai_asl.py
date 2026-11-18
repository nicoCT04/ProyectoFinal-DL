"""
Laboratorio 10 — Explainable AI sobre el modelo desplegado de SeñaBusca (v2).

Funciones compartidas por `xai/generar_xai.py` y `xai/lab10.ipynb`:

  * Carga del modelo v2 (`models/despliegue.json`) sin reentrenarlo.
  * Líneas base para Integrated Gradients (reposo, ceros, media de entrenamiento).
  * Integrated Gradients (captum) agregado por parte del cuerpo y por cuadro.
  * Oclusión por bloques (parte del cuerpo × ventana de tiempo): el método alternativo.
  * Curva de borrado (deletion) para medir la fidelidad de cada explicación.
  * Prueba de cordura (Adebayo et al., 2018): aleatorizar pesos y ver si la explicación cambia.

Recordatorio del formato de entrada (asl/features.py): cada seña es un tensor (64 cuadros × 161).
Las primeras 159 columnas son 53 puntos × (x, y, z) — 11 de pose (cara, hombros, codos, muñecas),
21 de la mano izquierda y 21 de la mano derecha — normalizados respecto a los hombros. Las 2
últimas son banderas de presencia (¿MediaPipe detectó la mano izquierda / derecha en ese cuadro?).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from asl.features import N_HAND, N_POSE, N_PTS, T_FIJO  # noqa: E402
from asl.models import construir  # noqa: E402

SEED = 23197
N_FEAT = N_PTS * 3 + 2
OUT = ROOT / "reports" / "xai"

# ------------------------------------------------------------------ grupos de variables
# Índices de punto dentro de los 53 (ver POSE_IDX en asl/features.py).
PUNTOS = {
    "cara": list(range(0, 5)),                       # nariz, ojos, comisuras de la boca
    "brazos": list(range(5, 11)),                    # hombros, codos y muñecas (pose)
    "mano izq.": list(range(N_POSE, N_POSE + N_HAND)),
    "mano der.": list(range(N_POSE + N_HAND, N_PTS)),
}
COL_PRES = {"mano izq.": N_PTS * 3, "mano der.": N_PTS * 3 + 1}


def columnas(puntos):
    return [3 * p + k for p in puntos for k in range(3)]


# Grupos para AGREGAR atribuciones (las banderas de presencia van aparte para buscar atajos).
GRUPOS_IG = {g: columnas(p) for g, p in PUNTOS.items()}
GRUPOS_IG["presencia de mano"] = [N_PTS * 3, N_PTS * 3 + 1]

# Grupos para OCLUIR: al quitar una mano también se apaga su bandera (así lo ve el modelo
# cuando MediaPipe no detecta la mano).
GRUPOS_OCL = {g: columnas(p) + ([COL_PRES[g]] if g in COL_PRES else []) for g, p in PUNTOS.items()}
VENTANA = 8                                   # 64 cuadros -> 8 ventanas de 8 cuadros
N_VENT = T_FIJO // VENTANA


# ------------------------------------------------------------------ modelo
def cargar_modelo(alias: str = "v2"):
    """Carga el modelo tal como lo usa la app (mismo archivo y umbral de despliegue.json)."""
    cfg = json.loads((ROOT / "models" / "despliegue.json").read_text())
    info = cfg["modelos"][alias]
    ck = torch.load(ROOT / "models" / f"{info['archivo']}.pt", map_location="cpu")
    m = construir(ck["arq"], len(ck["clases"]), **ck["kw"])
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, ck["clases"], info


def probabilidades(m, X, lote: int = 512) -> np.ndarray:
    out = []
    with torch.no_grad():
        for i in range(0, len(X), lote):
            out.append(F.softmax(m(torch.from_numpy(np.ascontiguousarray(X[i:i + lote]))), 1).numpy())
    return np.concatenate(out)


# ------------------------------------------------------------------ líneas base
def linea_base(X: np.ndarray, tipo: str, media_train: np.ndarray | None = None) -> np.ndarray:
    """Devuelve una línea base por instancia (misma forma que X).

    * "reposo": la misma persona quieta y sin manos detectadas. La pose (cara y brazos) se
      reemplaza por su promedio en el tiempo (se conserva el cuerpo, se quita el movimiento) y
      las manos quedan como las codifica el preprocesamiento cuando MediaPipe no ve ninguna:
      coordenadas 0 y bandera 0. Es una entrada que el modelo sí puede recibir de la cámara.
    * "ceros": todo en 0 (el cuerpo colapsado en el centro de los hombros): fuera de distribución.
    * "media": la seña promedio del conjunto de entrenamiento (igual para todas las instancias).
    """
    if tipo == "ceros":
        return np.zeros_like(X)
    if tipo == "media":
        return np.broadcast_to(media_train, X.shape).astype(np.float32).copy()
    if tipo == "reposo":
        B = np.zeros_like(X)
        pose = columnas(PUNTOS["cara"] + PUNTOS["brazos"])
        B[:, :, pose] = X[:, :, pose].mean(1, keepdims=True)
        return B
    raise ValueError(tipo)


# ------------------------------------------------------------------ Integrated Gradients
def integrated_gradients(m, X, objetivo, base, n_steps: int = 64, lote: int = 64):
    """Atribución IG (captum) para la clase `objetivo` (logit, antes del softmax).

    Devuelve (atribuciones [N, 64, 161], error de completitud [N]): por completitud,
    sum(atribuciones) = logit(x) - logit(línea base) salvo el error de la aproximación numérica.
    """
    from captum.attr import IntegratedGradients

    ig = IntegratedGradients(m)
    A, D = [], []
    for i in range(0, len(X), lote):
        a, d = ig.attribute(torch.from_numpy(np.ascontiguousarray(X[i:i + lote])),
                            baselines=torch.from_numpy(np.ascontiguousarray(base[i:i + lote])),
                            target=torch.from_numpy(np.asarray(objetivo[i:i + lote])),
                            n_steps=n_steps, method="gausslegendre", internal_batch_size=1024,
                            return_convergence_delta=True)
        A.append(a.detach().numpy()); D.append(d.detach().numpy())
    return np.concatenate(A), np.concatenate(D)


def agregar(A: np.ndarray, grupos=None) -> np.ndarray:
    """Suma atribuciones por grupo de variables y por cuadro -> [N, n_grupos, 64] (con signo)."""
    grupos = grupos or GRUPOS_IG
    return np.stack([A[:, :, cols].sum(2) for cols in grupos.values()], 1)


def por_bloque(A: np.ndarray) -> np.ndarray:
    """Atribución por bloque de oclusión (parte × ventana) -> [N, 4 * 8], para comparar métodos."""
    G = np.stack([A[:, :, cols].sum(2) for cols in GRUPOS_OCL.values()], 1)  # [N, 4, 64]
    return G.reshape(len(A), len(GRUPOS_OCL), N_VENT, VENTANA).sum(-1).reshape(len(A), -1)


def mascara_bloque(b: int) -> tuple[slice, list[int]]:
    g, w = divmod(b, N_VENT)
    return slice(w * VENTANA, (w + 1) * VENTANA), list(GRUPOS_OCL.values())[g]


# ------------------------------------------------------------------ oclusión (alternativa)
def oclusion(m, X, objetivo, base) -> np.ndarray:
    """Caída de probabilidad de la clase objetivo al reemplazar cada bloque por la línea base.

    Devuelve [N, 32] (positivo = el bloque ayudaba a la predicción). 32 pasadas hacia adelante
    por instancia; no necesita gradientes (caja negra).
    """
    p0 = probabilidades(m, X)[np.arange(len(X)), objetivo]
    S = np.zeros((len(X), len(GRUPOS_OCL) * N_VENT), dtype=np.float32)
    for b in range(S.shape[1]):
        t, cols = mascara_bloque(b)
        Xo = X.copy()
        Xo[:, t, cols] = base[:, t, cols]
        S[:, b] = p0 - probabilidades(m, Xo)[np.arange(len(X)), objetivo]
    return S


# ------------------------------------------------------------------ fidelidad
def curva_borrado(m, X, objetivo, base, orden: np.ndarray) -> np.ndarray:
    """Quita bloques en el orden dado (más importante primero) y registra la probabilidad de la
    clase objetivo. Devuelve [n_bloques + 1] promediado sobre instancias. Un área baja indica
    que la explicación encontró lo que de verdad usa el modelo (Samek et al., 2017)."""
    n_b = orden.shape[1]
    Xo = X.copy()
    idx = np.arange(len(X))
    curva = [probabilidades(m, Xo)[idx, objetivo].mean()]
    for k in range(n_b):
        for i in idx:
            t, cols = mascara_bloque(int(orden[i, k]))
            Xo[i, t, cols] = base[i, t, cols]
        curva.append(probabilidades(m, Xo)[idx, objetivo].mean())
    return np.array(curva)


def modelo_aleatorizado(m, semilla: int = SEED):
    """Copia del modelo con la cabeza y la última capa del encoder reinicializadas (prueba de
    cordura en cascada de Adebayo et al.). Si la explicación casi no cambia, no depende de lo
    que el modelo aprendió y no sería confiable."""
    import copy

    torch.manual_seed(semilla)
    r = copy.deepcopy(m)
    for mod in [r.head, r.enc.layers[-1]]:
        for sub in mod.modules():
            if hasattr(sub, "reset_parameters"):
                sub.reset_parameters()
    r.eval()
    return r


# ------------------------------------------------------------------ utilidades del dominio
def mano_dominante(X: np.ndarray) -> np.ndarray:
    """0 = izquierda, 1 = derecha: la mano que más se mueve en la seña.

    MediaPipe asigna izquierda/derecha según el cuerpo; una persona zurda o un video espejado
    aparecen como "mano izquierda dominante" (no se pueden distinguir con estos datos)."""
    pts = X[:, :, : N_PTS * 3].reshape(len(X), T_FIJO, N_PTS, 3)
    mov = lambda sl: np.abs(np.diff(pts[:, :, sl, :2], axis=1)).sum((1, 2, 3))  # noqa: E731
    return (mov(slice(N_POSE + N_HAND, N_PTS)) > mov(slice(N_POSE, N_POSE + N_HAND))).astype(int)


def puntos_xy(x: np.ndarray, t: int) -> np.ndarray:
    return x[t, : N_PTS * 3].reshape(N_PTS, 3)[:, :2]


# Conexiones para dibujar el esqueleto (índices dentro de los 53 puntos).
CONEX_POSE = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (1, 2), (3, 4)]
_DEDOS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
          (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
CONEX_MANOS = [(N_POSE + a, N_POSE + b) for a, b in _DEDOS] + \
              [(N_POSE + N_HAND + a, N_POSE + N_HAND + b) for a, b in _DEDOS]
