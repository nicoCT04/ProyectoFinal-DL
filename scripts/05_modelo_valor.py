"""
Paso 5 — Tabla de iteraciones, modelo de valor, curvas, sensibilidad y punto de operación.

Lee reports/runs/*.json y *_probs.npz (generados por 03/04) y produce:
  reports/tablas/tabla_iteraciones.csv      una fila por iteración (media ± desv. de 3 semillas)
  reports/tablas/tabla_versiones.csv        v0 / v1 / v2 en valor de negocio (con IC 95 %)
  reports/tablas/sensibilidad.csv           ±20 % y ±50 % por supuesto
  reports/figuras/*.png                     figuras para informe y presentación
  models/despliegue.json                    qué pesos y umbral usa la app

Reglas para no hacer trampa: el umbral de operación se elige en VALIDACIÓN y el valor se
reporta en PRUEBA (participantes que el modelo nunca vio).
"""
import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from asl.valor import (SUPUESTOS_BASE, Supuestos, con_cambio, costo_manual, mejor_umbral,  # noqa: E402
                       supuestos_dict, valor)

RUNS = ROOT / "reports" / "runs"
FIG = ROOT / "reports" / "figuras"
TAB = ROOT / "reports" / "tablas"
MODELS = ROOT / "models"
for d in (FIG, TAB):
    d.mkdir(parents=True, exist_ok=True)

COL = {"v0": "#8a8f98", "v1": "#2f5bd3", "v2": "#1f8a4c", "clasico": "#b4690e"}
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})

DESCRIPCION = {
    "clasico_logreg": "Regresión logística sobre estadísticos (sin DL)",
    "clasico_knn": "k-NN sobre estadísticos (sin DL)",
    "it0_lstm_base": "LSTM bidireccional, sin aumentos",
    "it1_lstm_aug": "+ aumentos (espejo, afín, tiempo) y label smoothing",
    "it2_tf_aug": "Transformer encoder (4 capas, d=192) + aumentos",
    "it3_tf_aug_small": "Transformer más pequeño (3 capas, d=128, dropout 0.3)",
    "it4_tl_tf_congelado": "TL: Transformer preentrenado en 450 señas, backbone congelado",
    "it5_tl_tf_finetune": "TL: Transformer preentrenado + fine-tuning completo (lr backbone 1e-4)",
    "it6_tl_lstm_finetune": "TL: LSTM preentrenada en 450 señas + fine-tuning",
    "it7_tl_tf_webcam": "it5 + aumentos de cámara en vivo (seña cortada, encuadre, mano perdida)",
    "pre_tf": "Preentrenamiento Transformer (450 señas, otra tarea)",
    "pre_lstm": "Preentrenamiento LSTM (450 señas, otra tarea)",
}


# ----------------------------------------------------------------------------- carga
def cargar_runs():
    filas = []
    for f in sorted(RUNS.glob("*.json")):
        r = json.loads(f.read_text())
        fila = dict(iteracion=r["iteracion"], seed=r["seed"], params=r.get("params"),
                    tiempo_train_min=r["tiempo_train_s"] / 60, latencia_cpu_ms=r.get("latencia_cpu_ms"),
                    mejor_epoch=r.get("mejor_epoch"), n_train=r.get("n_train"), n_clases=r.get("n_clases"))
        for s in ["train", "val", "test"]:
            for k in ["top1", "top3", "f1_macro"]:
                fila[f"{s}_{k}"] = r[s][k]
        filas.append(fila)
    return pd.DataFrame(filas)


def probs(it, seed, split):
    p = RUNS / f"{it}_s{seed}_{split}_probs.npz"
    return dict(np.load(p, allow_pickle=True)) if p.exists() else None


def paquete(it, seed):
    """Probabilidades de una corrida: val/test dentro del vocabulario y val/test fuera (OOV)."""
    d = dict(it=it, seed=seed, pv=probs(it, seed, "val"), pt=probs(it, seed, "test"),
             ov=probs(it, seed, "oov_val"), ot=probs(it, seed, "oov_test"))
    if d["pt"] is None or d["pv"] is None:
        return None
    if d["ot"] is None or d["ov"] is None:
        # Sin predicciones fuera de vocabulario el valor no es comparable: correr 03b primero.
        print(f"AVISO: {it} s{seed} sin probabilidades OOV; se omite del modelo de valor", flush=True)
        return None
    return d


def _oov(x):
    return None if x is None else x["probs"]


def TAU(d, s=SUPUESTOS_BASE):
    """Umbral que maximiza el valor en VALIDACIÓN (incluye señas fuera del vocabulario)."""
    return mejor_umbral(d["pv"]["probs"], d["pv"]["y"], s, probs_oov=_oov(d["ov"]))[0]


def VT(d, tau, s=SUPUESTOS_BASE):
    """Valor en PRUEBA con umbral tau."""
    return valor(d["pt"]["probs"], d["pt"]["y"], tau, s, probs_oov=_oov(d["ot"]))


def VV(d, tau, s=SUPUESTOS_BASE):
    return valor(d["pv"]["probs"], d["pv"]["y"], tau, s, probs_oov=_oov(d["ov"]))


def valor_run(it, seed, s=SUPUESTOS_BASE):
    """Umbral óptimo en validación, valor en prueba."""
    d = paquete(it, seed)
    if d is None:
        return None
    tau = TAU(d, s)
    v = VT(d, tau, s)
    v["valor_neto_mes_sin_umbral"] = VT(d, 0.0, s)["valor_neto_mes"]   # sin modo "no estoy seguro"
    return v


# ----------------------------------------------------------------------------- bootstrap
def _indices(d):
    pt, ot = d["pt"], d["ot"]
    parts = np.unique(pt["participantes"])
    idx = {p: np.where(pt["participantes"] == p)[0] for p in parts}
    idx_o = {p: (np.where(ot["participantes"] == p)[0] if ot is not None else None) for p in parts}
    return parts, idx, idx_o


def _valor_sel(d, tau, sel, sel_o, s):
    Po = d["ot"]["probs"][sel_o] if d["ot"] is not None else None
    return valor(d["pt"]["probs"][sel], d["pt"]["y"][sel], tau, s, probs_oov=Po)["valor_neto_mes"]


def bootstrap_participantes(d, tau, s=SUPUESTOS_BASE, B=1000, seed=0):
    """IC 95 % remuestreando PARTICIPANTES de prueba (los errores de una persona están correlacionados).
    Los videos fuera de vocabulario se remuestrean con las mismas personas."""
    rng = np.random.default_rng(seed)
    parts, idx, idx_o = _indices(d)
    pt = d["pt"]
    t1, t3, vn = [], [], []
    for _ in range(B):
        elegidos = rng.choice(parts, len(parts), replace=True)
        sel = np.concatenate([idx[p] for p in elegidos])
        sel_o = np.concatenate([idx_o[p] for p in elegidos]) if d["ot"] is not None else None
        P, y = pt["probs"][sel], pt["y"][sel]
        top = np.argsort(-P, 1)
        t1.append((top[:, 0] == y).mean()); t3.append((top[:, :3] == y[:, None]).any(1).mean())
        vn.append(_valor_sel(d, tau, sel, sel_o, s))
    q = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))  # noqa: E731
    return dict(top1=q(t1), top3=q(t3), valor=q(vn))


def bootstrap_diferencia(da, db, s=SUPUESTOS_BASE, B=1000, seed=0):
    """IC 95 % de valor(B) − valor(A) con remuestreo pareado de participantes."""
    rng = np.random.default_rng(seed)
    parts, idx, idx_o = _indices(da)
    out = []
    for _ in range(B):
        elegidos = rng.choice(parts, len(parts), replace=True)
        sel = np.concatenate([idx[p] for p in elegidos])
        sel_o = np.concatenate([idx_o[p] for p in elegidos]) if da["ot"] is not None else None
        out.append(_valor_sel(db, db["tau"], sel, sel_o, s) - _valor_sel(da, da["tau"], sel, sel_o, s))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def pm(m, sd, f="{:.3f}"):
    return f.format(m) if not np.isfinite(sd) or sd == 0 else (f + " ± " + f).format(m, sd)


# ----------------------------------------------------------------------------- main
def main():
    df = cargar_runs()
    df = df[~df["iteracion"].str.startswith("pre_")].copy()
    vals = []
    for _, r in df.iterrows():
        v = valor_run(r["iteracion"], r["seed"])
        vals.append({} if v is None else {"umbral": v["umbral"], "valor_neto_mes": v["valor_neto_mes"],
                                          "valor_sin_umbral": v["valor_neto_mes_sin_umbral"],
                                          "resueltas": v["resueltas"], "tasa_engano": v["tasa_engano"],
                                          "horas_mes": v["horas_ahorradas_mes"]})
    df = pd.concat([df.reset_index(drop=True), pd.DataFrame(vals)], axis=1)
    df.to_csv(TAB / "runs_detalle.csv", index=False)

    # Costo de preentrenamiento atribuido a las iteraciones de TL.
    pre = cargar_runs()
    pre = pre[pre["iteracion"].str.startswith("pre_")].groupby("iteracion")["tiempo_train_min"].mean().to_dict()

    # ---------------- tabla de iteraciones (media ± desviación sobre semillas)
    orden = ["clasico_logreg", "clasico_knn", "it0_lstm_base", "it1_lstm_aug", "it2_tf_aug",
             "it3_tf_aug_small", "it4_tl_tf_congelado", "it5_tl_tf_finetune", "it6_tl_lstm_finetune",
             "it7_tl_tf_webcam"]
    g = df.groupby("iteracion")
    filas = []
    for it in [o for o in orden if o in g.groups]:
        x = g.get_group(it)
        extra_pre = pre.get("pre_tf", 0) if "tl_tf" in it else pre.get("pre_lstm", 0) if "tl_lstm" in it else 0
        filas.append({
            "Iteración": it, "Cambio": DESCRIPCION.get(it, ""), "Semillas": len(x),
            "Parámetros": int(x["params"].iloc[0]) if pd.notna(x["params"].iloc[0]) else None,
            "Top-1 train": pm(x.train_top1.mean(), x.train_top1.std()),
            "Top-1 val": pm(x.val_top1.mean(), x.val_top1.std()),
            "Top-1 test": pm(x.test_top1.mean(), x.test_top1.std()),
            "Top-3 test": pm(x.test_top3.mean(), x.test_top3.std()),
            "F1 macro test": pm(x.test_f1_macro.mean(), x.test_f1_macro.std()),
            "Umbral (val)": pm(x.umbral.mean(), x.umbral.std(), "{:.2f}"),
            "Valor neto USD/mes (test)": pm(x.valor_neto_mes.mean(), x.valor_neto_mes.std(), "{:,.0f}"),
            "Búsquedas resueltas": pm(x.resueltas.mean(), x.resueltas.std(), "{:.3f}"),
            "Entrenamiento (min, M4 Pro)": f"{x.tiempo_train_min.mean():.1f}" +
                                          (f" + {extra_pre:.1f} preentreno" if extra_pre else ""),
            "Latencia CPU (ms)": f"{x.latencia_cpu_ms.mean():.1f}" if pd.notna(x.latencia_cpu_ms.mean()) else "",
        })
    tabla = pd.DataFrame(filas)
    tabla.to_csv(TAB / "tabla_iteraciones.csv", index=False)
    (TAB / "tabla_iteraciones.md").write_text(tabla.to_markdown(index=False))
    print(tabla[["Iteración", "Top-1 val", "Top-1 test", "Top-3 test", "Valor neto USD/mes (test)"]].to_string())

    # ---------------- versiones: v0 = LSTM base, v1 = mejor Transformer desde cero, v2 = mejor TL
    medias = g[["val_top1", "val_top3"]].mean()
    def mejor(cands):
        cands = [c for c in cands if c in medias.index]
        return max(cands, key=lambda c: (medias.loc[c, "val_top3"], medias.loc[c, "val_top1"]))
    # v2 = transfer learning con la MISMA arquitectura que v1 (Transformer), para comparar modelo base vs.
    # ajustado como pide el enunciado; it6 (LSTM + TL) se reporta como evidencia adicional.
    tl = [c for c in ["it4_tl_tf_congelado", "it5_tl_tf_finetune", "it7_tl_tf_webcam"] if c in medias.index]
    versiones = {"v0": "it0_lstm_base", "v1": "it2_tf_aug",
                 # (si aún no hay corridas de TL, se usa provisionalmente LSTM+aumentos)
                 "v2": mejor(tl) if tl else "it1_lstm_aug"}
    sel = {}
    for v, it in versiones.items():
        x = g.get_group(it)
        seed = int(x.sort_values(["val_top3", "val_top1"], ascending=False).iloc[0]["seed"])
        sel[v] = paquete(it, seed)
        sel[v]["tau"] = TAU(sel[v])

    filas = []
    for v, d in sel.items():
        res = VT(d, d["tau"])
        ic = bootstrap_participantes(d, d["tau"])
        oov_det = float((np.sort(d["ot"]["probs"], 1)[:, -3:].sum(1) < d["tau"]).mean()) if d["ot"] is not None else np.nan
        x = g.get_group(d["it"])
        filas.append({"Versión": v, "Iteración": d["it"], "Semilla": d["seed"], "Umbral": round(d["tau"], 2),
                      "Top-1 test": round(float((d["pt"]["probs"].argmax(1) == d["pt"]["y"]).mean()), 3),
                      "IC95 top-1": f"[{ic['top1'][0]:.3f}, {ic['top1'][1]:.3f}]",
                      "Top-3 test": round(float((np.argsort(-d["pt"]["probs"], 1)[:, :3] == d["pt"]["y"][:, None]).any(1).mean()), 3),
                      "IC95 top-3": f"[{ic['top3'][0]:.3f}, {ic['top3'][1]:.3f}]",
                      "Seg/búsqueda app": round(res["seg_por_busqueda_app"], 1),
                      "Seg/búsqueda manual": round(res["seg_por_busqueda_manual"], 1),
                      "Resueltas": round(res["resueltas"], 3), "Engaños": round(res["tasa_engano"], 3),
                      "Modo seguro": round(res["frac_modo_seguro"], 3),
                      "OOV detectadas": round(oov_det, 3),
                      "Horas ahorradas/mes": round(res["horas_ahorradas_mes"], 1),
                      "Valor neto USD/mes": round(res["valor_neto_mes"], 0),
                      "IC95 valor": f"[{ic['valor'][0]:,.0f}, {ic['valor'][1]:,.0f}]",
                      "Costo operación USD/mes": round(res["costo_operacion_mes"], 2),
                      "Latencia CPU ms": round(float(x.latencia_cpu_ms.mean()), 1),
                      "Entrenamiento min": round(float(x.tiempo_train_min.mean()), 1)})
    tv = pd.DataFrame(filas)
    d10 = bootstrap_diferencia(sel["v0"], sel["v1"])
    d21 = bootstrap_diferencia(sel["v1"], sel["v2"])
    d20 = bootstrap_diferencia(sel["v0"], sel["v2"])
    tv.to_csv(TAB / "tabla_versiones.csv", index=False)
    (TAB / "tabla_versiones.md").write_text(tv.to_markdown(index=False))
    difs = {"v1_menos_v0": d10, "v2_menos_v1": d21, "v2_menos_v0": d20}
    print(tv.to_string()); print("IC95 diferencias de valor:", difs)

    # ---------------- figura: valor vs umbral
    fig, ax = plt.subplots(figsize=(7, 4))
    grid = np.linspace(0, 1, 101)
    for v, d in sel.items():
        curva = [VT(d, u)["valor_neto_mes"] for u in grid]
        ax.plot(grid, curva, color=COL[v], label=f"{v} ({d['it']})")
        ax.scatter([d["tau"]], [VT(d, d["tau"])["valor_neto_mes"]],
                   color=COL[v], zorder=5, s=40, edgecolor="white")
    ax.set_xlabel("Umbral de confianza (masa de probabilidad del top-3)")
    ax.set_ylabel("Valor neto (USD/mes, prueba)")
    ax.set_title("Valor según el umbral de confianza (punto = umbral elegido en validación)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "valor_vs_umbral.png"); plt.close(fig)

    # ---------------- figura: valor vs métrica técnica (todas las corridas)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    d_ = df.dropna(subset=["valor_neto_mes"])
    for _, r in d_.iterrows():
        it = r["iteracion"]
        c = COL["clasico"] if it.startswith("clasico") else COL["v2"] if "tl" in it else \
            COL["v1"] if "tf" in it else COL["v0"]
        mk = "s" if it.startswith("lc_") else "o"
        ax.scatter(r["test_top3"], r["valor_neto_mes"], color=c, marker=mk, s=28, alpha=0.8)
    # Curva teórica: misma distribución de errores, accuracy top-3 variable (modelo sin umbral).
    a = np.linspace(0, 1, 50); s = SUPUESTOS_BASE
    fb = s.t_manual + (1 - s.p_manual) * s.c_no_resuelta
    c_fallo = s.p_engano * s.c_error + (1 - s.p_engano) * (s.t_rev10 + 0.6 * fb)
    costo = s.t_grabar + s.t_rev3 + (s.p_vocab * (1 - a) + (1 - s.p_vocab)) * c_fallo
    teor = (costo_manual(s) - costo) * s.busquedas_mes / 3600 * s.valor_hora - s.costo_fijo_mes - s.costo_pred * s.busquedas_mes
    ax.plot(a, teor, "--", color="#999", lw=1, label="Aproximación analítica (sin umbral)")
    ax.axhline(0, color="#c0392b", lw=0.8)
    a_eq = a[np.argmin(np.abs(teor))]
    ax.annotate(f"punto de equilibrio ≈ top-3 {a_eq:.2f}", (a_eq, 0), xytext=(a_eq + 0.05, teor.min() * 0.5),
                arrowprops=dict(arrowstyle="->", color="#c0392b"), fontsize=8, color="#c0392b")
    for lab, c in [("Clásico (sin DL)", COL["clasico"]), ("LSTM", COL["v0"]), ("Transformer", COL["v1"]),
                   ("Transfer learning", COL["v2"])]:
        ax.scatter([], [], color=c, label=lab)
    ax.scatter([], [], color="#555", marker="s", label="curva de aprendizaje (menos datos)")
    ax.set_xlabel("Top-3 accuracy en prueba"); ax.set_ylabel("Valor neto (USD/mes)")
    ax.set_title("Vínculo métrica técnica → valor de negocio (cada punto es una corrida)")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig(FIG / "valor_vs_top3.png"); plt.close(fig)

    # ---------------- figura: curva de aprendizaje (participantes de entrenamiento)
    lc = []
    for fam, full in [("tf", "it2_tf_aug"), ("tl", "it5_tl_tf_finetune")]:
        for frac, it in [(0.25, f"lc_{fam}_f25"), (0.5, f"lc_{fam}_f50"), (0.75, f"lc_{fam}_f75"), (1.0, full)]:
            if it in g.groups:
                x = g.get_group(it)
                lc.append(dict(familia=fam, frac=frac, top3=x.test_top3.mean(), top3_sd=x.test_top3.std(),
                               valor=x.valor_neto_mes.mean(), valor_sd=x.valor_neto_mes.std()))
    if lc:
        lc = pd.DataFrame(lc); lc.to_csv(TAB / "curva_aprendizaje.csv", index=False)
        fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
        n_parts = 35
        for fam, c, lab in [("tf", COL["v1"], "Transformer desde cero"), ("tl", COL["v2"], "Transfer learning")]:
            x = lc[lc.familia == fam]
            axs[0].errorbar(x.frac * n_parts, x.top3, yerr=x.top3_sd, color=c, marker="o", capsize=3, label=lab)
            axs[1].errorbar(x.frac * n_parts, x.valor, yerr=x.valor_sd, color=c, marker="o", capsize=3, label=lab)
        axs[0].set_ylabel("Top-3 en prueba"); axs[1].set_ylabel("Valor neto (USD/mes)")
        for ax in axs:
            ax.set_xlabel("Participantes en entrenamiento")
        axs[0].legend(frameon=False, fontsize=8)
        fig.suptitle("¿Vale la pena conseguir más datos? Rendimientos decrecientes")
        fig.tight_layout(); fig.savefig(FIG / "curva_aprendizaje.png"); plt.close(fig)

    # ---------------- sensibilidad (modelo desplegado)
    dep = max(sel, key=lambda v: VV(sel[v], sel[v]["tau"])["valor_neto_mes"])
    d = sel[dep]
    base = VT(d, d["tau"])["valor_neto_mes"]
    campos = ["p_vocab", "t_manual", "valor_hora", "busquedas_mes", "p_manual", "c_error", "p_engano", "t_rev3",
              "t_rev10", "c_no_resuelta", "t_grabar", "costo_fijo_mes"]
    filas = []
    for c in campos:
        fila = {"supuesto": c, "valor_base": getattr(SUPUESTOS_BASE, c)}
        for f in [0.5, 0.8, 1.2, 1.5]:
            s2 = con_cambio(SUPUESTOS_BASE, c, f)
            # Se re-optimiza el umbral en validación con los supuestos cambiados.
            fila[f"x{f}"] = VT(d, TAU(d, s2), s2)["valor_neto_mes"]
            # Diferencia frente a v0 con esos supuestos (¿se mantiene la conclusión?)
            fila[f"delta_vs_v0_x{f}"] = fila[f"x{f}"] - VT(sel["v0"], TAU(sel["v0"], s2), s2)["valor_neto_mes"]
        filas.append(fila)
    sens = pd.DataFrame(filas)
    sens["rango_50"] = (sens["x1.5"] - sens["x0.5"]).abs()
    sens = sens.sort_values("rango_50")
    sens.to_csv(TAB / "sensibilidad.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    y = np.arange(len(sens))
    for (lo, hi, alpha, lab) in [("x0.5", "x1.5", 0.35, "±50 %"), ("x0.8", "x1.2", 0.9, "±20 %")]:
        izq = np.minimum(sens[lo], sens[hi]); der = np.maximum(sens[lo], sens[hi])
        ax.barh(y, der - izq, left=izq, color=COL["v1"], alpha=alpha, label=lab, height=0.6)
    ax.axvline(base, color="black", lw=1); ax.axvline(0, color="#c0392b", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(sens["supuesto"])
    ax.set_xlabel(f"Valor neto USD/mes ({dep}, umbral re-optimizado)")
    ax.set_title("Análisis de sensibilidad: ¿qué pasa si los supuestos están mal?")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(FIG / "sensibilidad_tornado.png"); plt.close(fig)

    # Escenario pesimista conjunto: todos los supuestos favorables un 50 % peor.
    pes = Supuestos(**{**supuestos_dict(), "t_manual": SUPUESTOS_BASE.t_manual * 0.5,
                       "valor_hora": SUPUESTOS_BASE.valor_hora * 0.5,
                       "p_manual": min(1.0, SUPUESTOS_BASE.p_manual * 1.3),
                       "p_engano": min(1.0, SUPUESTOS_BASE.p_engano * 1.5),
                       "c_error": SUPUESTOS_BASE.c_error * 1.5,
                       "t_rev3": SUPUESTOS_BASE.t_rev3 * 1.5, "t_rev10": SUPUESTOS_BASE.t_rev10 * 1.5,
                       "p_vocab": SUPUESTOS_BASE.p_vocab * 0.8})
    escenarios = {}
    for v, dd in sel.items():
        tp = TAU(dd, pes)
        escenarios[v] = dict(base=VT(dd, dd["tau"])["valor_neto_mes"],
                             pesimista=VT(dd, tp, pes)["valor_neto_mes"],
                             umbral_pesimista=tp)

    # ---------------- sesgos: desempeño por participante de prueba
    pt = d["pt"]
    top = np.argsort(-pt["probs"], 1)
    acierto3 = (top[:, :3] == pt["y"][:, None]).any(1)
    acierto1 = top[:, 0] == pt["y"]
    pp = pd.DataFrame({"p": pt["participantes"], "t1": acierto1, "t3": acierto3}).groupby("p").agg(
        n=("t1", "size"), top1=("t1", "mean"), top3=("t3", "mean")).sort_values("top3")
    pp.to_csv(TAB / "por_participante.csv")
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.bar(pp.index, pp.top3, color=COL["v2"], alpha=0.85, label="top-3")
    ax.bar(pp.index, pp.top1, color=COL["v1"], alpha=0.9, width=0.5, label="top-1")
    ax.set_ylim(0, 1); ax.set_ylabel("Accuracy"); ax.set_xlabel("Participante de prueba (no visto)")
    ax.set_title(f"Desempeño por persona ({dep}) — ¿funciona peor para alguien?")
    ax.legend(frameon=True, fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(FIG / "por_participante.png"); plt.close(fig)

    # ---------------- señas más difíciles y confusiones
    voc = json.loads((ROOT / "data" / "vocabulario.json").read_text())
    clases = voc["objetivo"]; es = voc["es"]
    pc = pd.DataFrame({"y": pt["y"], "t1": acierto1, "t3": acierto3}).groupby("y").agg(
        top1=("t1", "mean"), top3=("t3", "mean"), n=("t1", "size"))
    pc.index = [clases[i] for i in pc.index]
    pc = pc.sort_values(["top3", "top1"])
    pc.to_csv(TAB / "por_sena.csv")
    pred = top[:, 0]
    conf = pd.Series([f"{clases[a]} → {clases[b]}" for a, b in zip(pt["y"], pred) if a != b]).value_counts()
    conf.head(15).to_csv(TAB / "confusiones_top.csv", header=["veces"])
    fig, ax = plt.subplots(figsize=(7, 4))
    peor = pc.head(12)
    ax.barh([f"{g_} ({es.get(g_, '')})" for g_ in peor.index], peor.top3, color=COL["v2"], alpha=0.85, label="top-3")
    ax.barh([f"{g_} ({es.get(g_, '')})" for g_ in peor.index], peor.top1, color=COL["v1"], height=0.45, label="top-1")
    ax.set_xlim(0, 1); ax.set_title("Las 12 señas más difíciles (prueba)"); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "senas_dificiles.png"); plt.close(fig)

    # ---------------- calibración de la confianza
    conf3 = np.take_along_axis(pt["probs"], top[:, :3], 1).sum(1)
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    bins = np.linspace(0, 1, 11); mid = (bins[1:] + bins[:-1]) / 2
    idx_b = np.clip(np.digitize(conf3, bins) - 1, 0, 9)
    cnt = np.array([(idx_b == i).sum() for i in range(10)])
    obs = np.array([acierto3[idx_b == i].mean() if cnt[i] >= 10 else np.nan for i in range(10)])  # bins con ≥ 10 casos
    axs[0].plot([0, 1], [0, 1], "--", color="#999"); axs[0].plot(mid, obs, "o-", color=COL["v2"])
    for x_, o_, n_ in zip(mid, obs, cnt):
        if np.isfinite(o_):
            axs[0].annotate(f"n={n_}", (x_, o_), textcoords="offset points", xytext=(0, -12), ha="center", fontsize=7)
    axs[0].set_xlabel("Confianza top-3 del modelo"); axs[0].set_ylabel("Frecuencia real de acierto top-3")
    axs[0].set_title("Calibración (bins con ≥ 10 casos)")
    kw = dict(bins=np.linspace(0, 1, 26), alpha=0.6, density=True)
    axs[1].hist(conf3[acierto3], color=COL["v2"], label="en vocabulario, correcta en top-3", **kw)
    axs[1].hist(conf3[~acierto3], color="#c0392b", label="en vocabulario, no está", **kw)
    if d["ot"] is not None:
        conf_o = np.sort(d["ot"]["probs"], 1)[:, -3:].sum(1)
        axs[1].hist(conf_o, color="#8a8f98", label="seña FUERA del vocabulario", histtype="step", lw=2,
                    bins=kw["bins"], density=True)
    axs[1].axvline(d["tau"], color="black", lw=1); axs[1].text(d["tau"], axs[1].get_ylim()[1] * 0.9, " umbral", fontsize=8)
    axs[1].set_xlabel("Confianza top-3"); axs[1].legend(frameon=False, fontsize=7); axs[1].set_title("Separación aciertos / fallos")
    fig.tight_layout(); fig.savefig(FIG / "calibracion.png"); plt.close(fig)

    # ---------------- curvas de entrenamiento
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for it, c, lab in [("it0_lstm_base", COL["v0"], "v0 LSTM sin aumentos"),
                       ("it1_lstm_aug", "#555", "LSTM + aumentos"),
                       (versiones["v1"], COL["v1"], f"v1 {versiones['v1']}"),
                       (versiones["v2"], COL["v2"], f"v2 {versiones['v2']}")]:
        h = RUNS / f"{it}_s0_hist.csv"
        if not h.exists():
            continue
        h = pd.read_csv(h)
        axs[0].plot(h.epoch, h.train_loss, color=c, lw=1, ls="--")
        axs[0].plot(h.epoch, h.val_loss, color=c, lw=1.4, label=lab)
        axs[1].plot(h.epoch, h.val_top1, color=c, lw=1.4, label=lab)
    axs[0].set_title("Pérdida (– – train, —— validación)"); axs[0].set_xlabel("epoch"); axs[0].set_ylim(0, 6)
    axs[1].set_title("Top-1 en validación"); axs[1].set_xlabel("epoch")
    axs[1].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "curvas_entrenamiento.png"); plt.close(fig)

    # ---------------- despliegue
    nombres = {"v0": "v0 · LSTM desde cero", "v1": "v1 · Transformer desde cero",
               "v2": "v2 · Transformer con transfer learning"}
    if sel["v2"]["it"].startswith("it6"):
        nombres["v2"] = "v2 · LSTM con transfer learning"
    cfg = {"default": dep, "modelos": {}}
    for v, dd in sel.items():
        cfg["modelos"][v] = dict(archivo=f"{dd['it']}_s{dd['seed']}", desc=nombres[v], umbral=round(dd["tau"], 3),
                                 iteracion=dd["it"], seed=dd["seed"])
    (MODELS / "despliegue.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))

    resumen = dict(versiones={v: {k: dd[k] for k in ["it", "seed", "tau"]} for v, dd in sel.items()},
                   desplegado=dep, ic95_diferencias_valor=difs, escenarios=escenarios,
                   supuestos=supuestos_dict(), costo_manual_seg=costo_manual(SUPUESTOS_BASE),
                   preentreno_min=pre)
    (TAB / "resumen_valor.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False))
    print(json.dumps(resumen, indent=1, ensure_ascii=False)[:2500])


if __name__ == "__main__":
    main()
