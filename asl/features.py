"""
Representación de una seña como secuencia de puntos clave (landmarks).

Por cuadro guardamos 53 puntos (x, y, z) en coordenadas normalizadas de imagen:
  * 11 puntos de pose de la parte superior del cuerpo (nariz, ojos, boca, hombros,
    codos, muñecas) — dan la ubicación de la seña respecto al cuerpo.
  * 21 puntos de la mano izquierda y 21 de la derecha — forma y movimiento de la mano.

La asignación izquierda/derecha se hace por cercanía a las muñecas de la pose y NO con la
etiqueta "handedness" de MediaPipe: así el resultado es idéntico en el video (Python) y en
la cámara del navegador (JavaScript), aunque la imagen se muestre en espejo.

Este mismo módulo lo usan el entrenamiento y la API, para que el preprocesamiento
sea exactamente el mismo en ambos lados.
"""
from __future__ import annotations

import numpy as np

# Índices de la pose de MediaPipe que conservamos (parte superior del cuerpo).
POSE_IDX = [0, 2, 5, 9, 10, 11, 12, 13, 14, 15, 16]
N_POSE = len(POSE_IDX)          # 11
N_HAND = 21
N_PTS = N_POSE + 2 * N_HAND     # 53
L_HAND = slice(N_POSE, N_POSE + N_HAND)
R_HAND = slice(N_POSE + N_HAND, N_PTS)
# Posiciones (dentro de POSE_IDX) de hombros y muñecas.
P_LSH, P_RSH, P_LWR, P_RWR = 5, 6, 9, 10

T_FIJO = 64  # longitud a la que se remuestrea cada seña


def frame_desde_resultados(pose_lms, hands_lms) -> np.ndarray:
    """Construye el arreglo (53, 3) de un cuadro a partir de listas de landmarks.

    pose_lms: lista de 33 puntos con .x .y .z (o None si no hay persona).
    hands_lms: lista de manos, cada una con 21 puntos con .x .y .z.
    Los puntos ausentes quedan en NaN.
    """
    out = np.full((N_PTS, 3), np.nan, dtype=np.float32)
    if pose_lms is not None:
        for j, i in enumerate(POSE_IDX):
            p = pose_lms[i]
            out[j] = (p.x, p.y, p.z)
    manos = [np.array([(p.x, p.y, p.z) for p in h], dtype=np.float32) for h in (hands_lms or [])]
    asignar_manos(out, manos)
    return out


def asignar_manos(out: np.ndarray, manos: list[np.ndarray]) -> None:
    """Coloca cada mano detectada en el bloque izquierdo o derecho según la muñeca más cercana."""
    if not manos:
        return
    lw, rw = out[P_LWR, :2], out[P_RWR, :2]
    tiene_pose = not (np.isnan(lw).any() or np.isnan(rw).any())
    usados = set()
    # Ordenamos para que la asignación sea determinista.
    for h in manos[:2]:
        muneca = h[0, :2]
        if tiene_pose:
            dl, dr = np.linalg.norm(muneca - lw), np.linalg.norm(muneca - rw)
            lado = "L" if dl <= dr else "R"
        else:
            # Sin pose: en la imagen sin espejo, la mano izquierda del señante aparece a la derecha.
            lado = "L" if muneca[0] >= 0.5 else "R"
        if lado in usados:
            lado = "R" if lado == "L" else "L"
        usados.add(lado)
        out[L_HAND if lado == "L" else R_HAND] = h


def normalizar(seq: np.ndarray) -> np.ndarray:
    """Centra en el punto medio de los hombros y escala por el ancho de hombros.

    seq: (T, 53, 3) con NaN en los puntos faltantes. Devuelve el mismo tamaño.
    Esto hace al modelo invariante a la posición de la persona y a su distancia a la cámara.
    """
    seq = seq.copy()
    sh = seq[:, [P_LSH, P_RSH], :2]                      # (T, 2, 2)
    centro = np.nanmean(sh.reshape(-1, 2), axis=0)       # promedio en todo el clip
    ancho = np.nanmedian(np.linalg.norm(sh[:, 0] - sh[:, 1], axis=-1))
    if not np.isfinite(centro).all():
        centro = np.array([0.5, 0.5], dtype=np.float32)
    if not np.isfinite(ancho) or ancho < 1e-3:
        ancho = 0.3
    seq[..., :2] = (seq[..., :2] - centro) / ancho
    seq[..., 2] = seq[..., 2] / ancho
    return seq


def recortar_actividad(seq: np.ndarray, margen: int = 3) -> np.ndarray:
    """Quita cuadros iniciales/finales sin manos visibles (reposo antes y después de la seña)."""
    hay_mano = ~np.isnan(seq[:, N_POSE:, 0]).all(axis=1)
    idx = np.where(hay_mano)[0]
    if len(idx) == 0:
        return seq
    a, b = max(idx[0] - margen, 0), min(idx[-1] + margen + 1, len(seq))
    return seq[a:b]


def remuestrear(seq: np.ndarray, t: int = T_FIJO) -> np.ndarray:
    """Interpolación lineal en el tiempo a t cuadros (maneja NaN rellenando hacia adelante/atrás)."""
    seq = rellenar(seq)
    T = len(seq)
    if T == 1:
        return np.repeat(seq, t, axis=0)
    pos = np.linspace(0, T - 1, t)
    i0 = np.floor(pos).astype(int)
    i1 = np.minimum(i0 + 1, T - 1)
    w = (pos - i0)[:, None, None]
    return (1 - w) * seq[i0] + w * seq[i1]


def rellenar(seq: np.ndarray) -> np.ndarray:
    """Rellena NaN por punto con el último valor visto (y luego el siguiente); lo que quede, a 0."""
    seq = seq.copy()
    T = len(seq)
    for t in range(1, T):
        m = np.isnan(seq[t])
        seq[t][m] = seq[t - 1][m]
    for t in range(T - 2, -1, -1):
        m = np.isnan(seq[t])
        seq[t][m] = seq[t + 1][m]
    return np.nan_to_num(seq, nan=0.0)


def mascara_presencia(seq: np.ndarray) -> np.ndarray:
    """(T, 2): 1 si la mano izquierda / derecha está detectada en el cuadro."""
    return np.stack(
        [~np.isnan(seq[:, L_HAND, 0]).all(1), ~np.isnan(seq[:, R_HAND, 0]).all(1)], axis=1
    ).astype(np.float32)


def preparar(seq_cruda: np.ndarray, t: int = T_FIJO) -> np.ndarray:
    """Pipeline completo: crudo (T, 53, 3) -> entrada del modelo (t, 53*3 + 2)."""
    seq = recortar_actividad(seq_cruda)
    pres = mascara_presencia(seq)
    seq = normalizar(seq)
    x = remuestrear(seq, t).reshape(t, -1)
    # La presencia de cada mano también se remuestrea (vecino más cercano).
    idx = np.round(np.linspace(0, len(pres) - 1, t)).astype(int)
    x = np.concatenate([x, pres[idx]], axis=1)
    return x.astype(np.float32)


N_FEAT = N_PTS * 3 + 2  # 161
