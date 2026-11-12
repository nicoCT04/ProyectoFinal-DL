"""
Modelo de valor: traduce las predicciones del modelo (top-k y confianza) en minutos ahorrados,
búsquedas resueltas y dólares al mes.

Escenario: un programa de enseñanza de ASL (curso universitario o plataforma de aprendizaje)
cuyos estudiantes ven una seña y no saben qué significa. Hoy la buscan "a mano" en un
diccionario por palabras o por rasgos (forma de mano, ubicación), o preguntan a alguien.

Flujo con la app, por búsqueda:
  1. El usuario graba la seña (t_grabar).
  2. Si la confianza del modelo (masa de probabilidad del top-3) >= umbral -> MODO SEGURO:
     se muestran 3 opciones; revisarlas cuesta t_rev3. Si la correcta está ahí, se resuelve.
     Si no está, el usuario puede creer una opción equivocada (p_engano) -> costo c_error;
     si no se engaña, abre "ver más" (top-10, t_rev10 extra) y, si tampoco está, busca a mano.
  3. Si la confianza < umbral -> MODO "NO ESTOY SEGURO": se muestran 10 opciones con aviso
     (t_rev10). El aviso reduce el engaño (p_engano_aviso). Si no está, busca a mano.
Búsqueda manual: dura t_manual y se resuelve con probabilidad p_manual; si no se resuelve,
se suma c_no_resuelta (preguntar al docente, aprendizaje retrasado).

Todos los tiempos en segundos. El valor se expresa en minutos y se monetiza con valor_hora.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np


@dataclass(frozen=True)
class Supuestos:
    # --- tiempos de interacción (s)
    t_grabar: float = 8.0          # levantar manos, hacer la seña, bajar manos
    t_rev3: float = 15.0           # mirar 3 videos de ejemplo y elegir
    t_rev10: float = 45.0          # revisar una lista de 10 opciones
    t_manual: float = 180.0        # búsqueda manual por rasgos en un diccionario en línea
    # --- probabilidades de comportamiento
    p_manual: float = 0.65         # probabilidad de que la búsqueda manual encuentre la seña
    p_engano: float = 0.30         # aceptar una opción equivocada mostrada "con seguridad"
    p_engano_aviso: float = 0.05   # ídem cuando la app avisa que no está segura
    # --- costos de error (s equivalentes)
    c_error: float = 600.0         # aprender/usar un significado equivocado (corregirlo después)
    c_no_resuelta: float = 300.0   # búsqueda que no se resuelve (preguntar, retrasar la tarea)
    # --- cobertura del vocabulario
    p_vocab: float = 0.85          # fracción de búsquedas cuya seña SÍ está en el vocabulario
                                   # (la app se alinea al vocabulario del curso; el resto es "fuera")
    # --- escala y dinero
    busquedas_mes: float = 10_000  # 500 estudiantes x 20 búsquedas al mes
    valor_hora: float = 12.0       # USD por hora de tiempo de un estudiante
    costo_fijo_mes: float = 25.0   # servidor CPU pequeño (USD/mes)
    costo_pred: float = 0.00002    # USD por predicción (CPU-tiempo + transferencia)


SUPUESTOS_BASE = Supuestos()


def costo_manual(s: Supuestos) -> float:
    """Costo esperado (s) de una búsqueda sin la app."""
    return s.t_manual + (1 - s.p_manual) * s.c_no_resuelta


def costos_por_busqueda(probs: np.ndarray, y: np.ndarray, umbral: float, s: Supuestos) -> dict:
    """Costo esperado (s) de cada búsqueda con la app, a partir de las probabilidades del modelo."""
    orden = np.argsort(-probs, 1)
    hit = orden == y[:, None]                           # y = -1 -> seña fuera del vocabulario
    rank = np.where(hit.any(1), hit.argmax(1), probs.shape[1])   # posición de la correcta (0 = top-1)
    en3, en10 = rank < 3, rank < 10
    conf = np.take_along_axis(probs, orden[:, :3], 1).sum(1)
    seguro = conf >= umbral
    fallback = s.t_manual + (1 - s.p_manual) * s.c_no_resuelta

    # Modo seguro
    c_seg = s.t_grabar + s.t_rev3 + np.where(
        en3, 0.0,
        s.p_engano * s.c_error + (1 - s.p_engano) * (s.t_rev10 + np.where(en10, 0.0, fallback)))
    # Modo "no estoy seguro"
    c_ins = s.t_grabar + s.t_rev10 + np.where(
        en10, 0.0, s.p_engano_aviso * s.c_error + (1 - s.p_engano_aviso) * fallback)
    costo = np.where(seguro, c_seg, c_ins)

    resuelta = np.where(seguro, np.where(en3, 1.0, (1 - s.p_engano) * np.where(en10, 1.0, s.p_manual)),
                        np.where(en10, 1.0, (1 - s.p_engano_aviso) * s.p_manual))
    engano = np.where(seguro, ~en3 * s.p_engano, ~en10 * s.p_engano_aviso)
    return dict(costo=costo, resuelta=resuelta, engano=engano, seguro=seguro, rank=rank, conf=conf)


def valor(probs: np.ndarray, y: np.ndarray, umbral: float = 0.5, s: Supuestos = SUPUESTOS_BASE,
          probs_oov: np.ndarray | None = None) -> dict:
    """Resumen de negocio de un modelo con un umbral dado.

    probs_oov: probabilidades del modelo sobre videos de señas FUERA del vocabulario (medidas con
    las 450 señas de prueba que el modelo no clasifica). Se mezclan con peso (1 - p_vocab).
    """
    c = costos_por_busqueda(probs, y, umbral, s)
    if probs_oov is not None and s.p_vocab < 1:
        co = costos_por_busqueda(probs_oov, np.full(len(probs_oov), -1), umbral, s)
        w = s.p_vocab
        c = {k: (np.concatenate([c[k], co[k]]) if k in ("rank", "seguro", "conf") else None) for k in c} | {
            "costo": np.array([w * c["costo"].mean() + (1 - w) * co["costo"].mean()]),
            "resuelta": np.array([w * c["resuelta"].mean() + (1 - w) * co["resuelta"].mean()]),
            "engano": np.array([w * c["engano"].mean() + (1 - w) * co["engano"].mean()]),
            "seguro_in": c["seguro"], "seguro_oov": co["seguro"]}
        c["seguro"] = np.array([w * c["seguro_in"].mean() + (1 - w) * c["seguro_oov"].mean()])
    ahorro_s = costo_manual(s) - c["costo"].mean()
    horas_mes = ahorro_s * s.busquedas_mes / 3600
    costo_op = s.costo_fijo_mes + s.costo_pred * s.busquedas_mes
    return dict(
        umbral=umbral,
        seg_por_busqueda_app=float(c["costo"].mean()),
        seg_por_busqueda_manual=float(costo_manual(s)),
        ahorro_seg_por_busqueda=float(ahorro_s),
        resueltas=float(c["resuelta"].mean()),
        resueltas_manual=s.p_manual,
        tasa_engano=float(c["engano"].mean()),
        frac_modo_seguro=float(c["seguro"].mean()),
        horas_ahorradas_mes=float(horas_mes),
        valor_bruto_mes=float(horas_mes * s.valor_hora),
        costo_operacion_mes=float(costo_op),
        valor_neto_mes=float(horas_mes * s.valor_hora - costo_op),
    )


def mejor_umbral(probs, y, s: Supuestos = SUPUESTOS_BASE, grid=None, probs_oov=None):
    grid = np.linspace(0.0, 1.0, 101) if grid is None else grid
    vals = [valor(probs, y, u, s, probs_oov)["valor_neto_mes"] for u in grid]
    i = int(np.argmax(vals))
    return float(grid[i]), np.array(vals), grid


def con_cambio(s: Supuestos, campo: str, factor: float) -> Supuestos:
    v = getattr(s, campo) * factor
    if campo.startswith("p_"):
        v = float(np.clip(v, 0, 1))
    return replace(s, **{campo: v})


def supuestos_dict(s: Supuestos = SUPUESTOS_BASE) -> dict:
    return asdict(s)
