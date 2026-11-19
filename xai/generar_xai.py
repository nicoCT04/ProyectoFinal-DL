"""
Laboratorio 10 — genera TODAS las figuras y tablas de explicabilidad del informe.

    .venv/bin/python xai/generar_xai.py          # ≈ 6 min en Apple M4 Pro (CPU, 1 hilo)

Salida en reports/xai/ (figuras .png, tablas .csv y resumen.json). Semilla fija (SEED = 23197),
todo en CPU y con un solo hilo para que los números se regeneren idénticos.
Requiere data/cache/objetivo.npz y data/cache/preentreno.npz (se generan con scripts 01–02;
no se versionan por la licencia de ASL Citizen).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import xai_asl as X_  # noqa: E402
from xai_asl import (GRUPOS_IG, GRUPOS_OCL, N_VENT, OUT, ROOT, SEED, agregar, cargar_modelo,  # noqa: E402
                     curva_borrado, integrated_gradients, linea_base, mano_dominante,
                     modelo_aleatorizado, oclusion, por_bloque, probabilidades)

sys.path.insert(0, str(ROOT))
from asl.data import cargar_grupo, vocabulario  # noqa: E402

N_STEPS = 64          # pasos de la integral de IG (error de completitud < 1e-3 con Gauss-Legendre)
UMBRAL_TOL = 0.05     # "cerca del umbral": |confianza - umbral| <= 0.05
plt.rcParams.update({"figure.dpi": 150, "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "font.family": "DejaVu Sans"})
NOMBRES = list(GRUPOS_IG)


def fijar_semillas():
    np.random.seed(SEED); torch.manual_seed(SEED)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)


def cuota(A):
    """Proporción de |atribución| que se lleva cada grupo, por instancia -> [N, 5]."""
    G = np.abs(agregar(A)).sum(2)
    return G / G.sum(1, keepdims=True)


def spearman(a, b):
    ra = a.argsort(1).argsort(1); rb = b.argsort(1).argsort(1)
    ra = ra - ra.mean(1, keepdims=True); rb = rb - rb.mean(1, keepdims=True)
    return (ra * rb).sum(1) / np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1))


def resumen_logreg(X):
    """Mismo vector fijo que el baseline clásico del proyecto (scripts/04_baseline_clasico.py)."""
    clave = X[:, [8, 24, 40, 56]].reshape(len(X), -1)
    return np.concatenate([X.mean(1), X.std(1), X.min(1), X.max(1), clave], 1)


def importancia_logreg(Xtr, ytr):
    """Baseline interpretable por diseño: regresión logística; importancia = |coef| estandarizado
    sumado sobre clases y estadísticos, agregado por grupo de variables."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
    lr.fit(resumen_logreg(Xtr), ytr)
    w = np.abs(lr[-1].coef_).sum(0).reshape(8, X_.N_FEAT).sum(0)  # por columna original
    imp = np.array([w[c].sum() for c in GRUPOS_IG.values()])
    return lr, imp / imp.sum()


def main():
    fijar_semillas()
    OUT.mkdir(parents=True, exist_ok=True)
    voc = vocabulario(); ES = voc["es"]
    m, clases, info = cargar_modelo("v2")
    umbral = info["umbral"]
    data, _ = cargar_grupo("objetivo")
    Xte, yte, pte, fte = data["test"]
    Xtr, ytr = data["train"][:2]
    media_train = Xtr.mean(0, keepdims=True)

    # ---------------------------------------------------------------- predicciones
    P = probabilidades(m, Xte); pred = P.argmax(1); conf = P.max(1)
    ok = pred == yte
    dom = mano_dominante(Xte)
    print(f"v2 top-1 prueba = {ok.mean():.3f}  ({(~ok).sum()} errores de {len(ok)})")

    # ---------------------------------------------------------------- IG con 3 líneas base
    bases = {t: linea_base(Xte, t, media_train) for t in ["reposo", "ceros", "media"]}
    IG, filas_base = {}, []
    for t, B in bases.items():
        t0 = time.time()
        A, d = integrated_gradients(m, Xte, pred, B, N_STEPS)
        IG[t] = A
        with torch.no_grad():
            lb = m(torch.from_numpy(B)).numpy()
        cu = cuota(A).mean(0)
        filas_base.append(dict(linea_base=t, segundos=round(time.time() - t0, 1),
                               err_completitud_max=float(np.abs(d).max()),
                               prob_max_en_base=float(torch.softmax(torch.from_numpy(lb), 1).max(1).values.mean()),
                               **{f"cuota {g}": round(float(c), 3) for g, c in zip(NOMBRES, cu)}))
    A = IG["reposo"]                                  # explicación principal
    Gs = agregar(A)                                   # [N, 5, 64] con signo

    # ---------------------------------------------------------------- oclusión (alternativa)
    t0 = time.time()
    OCL = oclusion(m, Xte, pred, bases["reposo"])
    t_ocl = time.time() - t0
    BIG = por_bloque(A)
    rho_ig_ocl = spearman(BIG, OCL)

    # ---------------------------------------------------------------- fidelidad (borrado)
    rng = np.random.default_rng(SEED)
    ordenes = {
        "IG (reposo)": np.argsort(-BIG, 1),
        "IG (ceros)": np.argsort(-por_bloque(IG["ceros"]), 1),
        "IG (media)": np.argsort(-por_bloque(IG["media"]), 1),
        "Oclusión": np.argsort(-OCL, 1),
        "Azar": np.stack([rng.permutation(BIG.shape[1]) for _ in range(len(Xte))]),
    }
    curvas = {k: curva_borrado(m, Xte, pred, bases["reposo"], o) for k, o in ordenes.items()}
    auc = {k: float(np.trapezoid(c, dx=1 / (len(c) - 1))) for k, c in curvas.items()}

    # ---------------------------------------------------------------- prueba de cordura
    mr = modelo_aleatorizado(m)
    Ar, _ = integrated_gradients(mr, Xte, pred, bases["reposo"], N_STEPS)
    rho_cordura = spearman(BIG, por_bloque(Ar))

    # ---------------------------------------------------------------- costo por predicción
    x1, b1, t1 = Xte[:1], bases["reposo"][:1], pred[:1]
    def cronometrar(f, rep=20):
        f(); t0 = time.perf_counter()
        for _ in range(rep):
            f()
        return (time.perf_counter() - t0) / rep * 1000
    costo = dict(
        inferencia_ms=cronometrar(lambda: probabilidades(m, x1)),
        ig_ms=cronometrar(lambda: integrated_gradients(m, x1, t1, b1, N_STEPS)),
        oclusion_ms=cronometrar(lambda: oclusion(m, x1, t1, b1)),
    )

    metodos = pd.DataFrame([
        dict(metodo=k, auc_borrado=round(v, 3)) for k, v in auc.items()])
    metodos["ms_por_instancia_cpu"] = metodos["metodo"].map(
        {"IG (reposo)": costo["ig_ms"], "IG (ceros)": costo["ig_ms"], "IG (media)": costo["ig_ms"],
         "Oclusión": costo["oclusion_ms"], "Azar": 0.0}).round(1)
    metodos.to_csv(OUT / "tabla_metodos.csv", index=False)
    pd.DataFrame(filas_base).to_csv(OUT / "tabla_lineas_base.csv", index=False)

    # ---------------------------------------------------------------- baseline interpretable
    lr, imp_lr = importancia_logreg(Xtr, ytr)
    acc_lr = float((lr.predict(resumen_logreg(Xte)) == yte).mean())

    # ---------------------------------------------------------------- atajos y subgrupos
    cu = cuota(A)
    absT = np.abs(Gs).sum(1)                                     # [N, 64]
    bordes = (absT[:, :8].sum(1) + absT[:, -8:].sum(1)) / absT.sum(1)
    # mano dominante vs no dominante (alinea izq./der. según quién firma)
    i_izq, i_der = NOMBRES.index("mano izq."), NOMBRES.index("mano der.")
    c_dom = np.where(dom == 1, cu[:, i_der], cu[:, i_izq])
    c_nodom = np.where(dom == 1, cu[:, i_izq], cu[:, i_der])
    filas = []
    for nombre, s in [("todas", np.ones_like(ok)), ("aciertos", ok), ("errores", ~ok),
                      ("mano der. dominante", dom == 1), ("mano izq. dominante", dom == 0)]:
        filas.append(dict(grupo=nombre, n=int(s.sum()), top1=round(float(ok[s].mean()), 3),
                          cuota_cara=round(float(cu[s, 0].mean()), 3),
                          cuota_brazos=round(float(cu[s, 1].mean()), 3),
                          cuota_mano_dominante=round(float(c_dom[s].mean()), 3),
                          cuota_mano_no_dominante=round(float(c_nodom[s].mean()), 3),
                          cuota_presencia=round(float(cu[s, 4].mean()), 3),
                          cuota_bordes_16_cuadros=round(float(bordes[s].mean()), 3)))
    atajos = pd.DataFrame(filas)
    atajos.to_csv(OUT / "tabla_atajos_subgrupos.csv", index=False)

    # ---------------------------------------------------------------- fuera de vocabulario
    oov, cl_oov = cargar_grupo("preentreno")
    Xo, yo, po, _ = oov["test"]
    Po = probabilidades(m, Xo)
    acepta = Po.max(1) >= umbral
    i_oov = int(np.argmax(np.where(acepta, Po.max(1), -1)))

    # ---------------------------------------------------------------- casos locales
    idx = np.arange(len(yte))
    casos = []
    def agregar_caso(tipo, i, x=None, objetivo=None, real=None, persona=None):
        casos.append(dict(tipo=tipo, i=i, x=x, objetivo=objetivo, real=real, persona=persona))
    agregar_caso("Acierto, alta confianza", int(idx[ok][np.argmax(conf[ok])]))
    agregar_caso("Error con más confianza", int(idx[~ok][np.argmax(conf[~ok])]))
    par = (yte == clases.index("GOOD")) & (pred == clases.index("THANKYOU"))
    agregar_caso("Error del par más confundido", int(idx[par][np.argmax(conf[par])]))
    cerca = np.abs(conf - umbral)
    agregar_caso(f"Cerca del umbral ({umbral})", int(np.argmin(cerca)))
    sub = ok & (dom == 0)
    # subgrupo: persona con mano izquierda dominante (zurda o video espejado), acierto típico
    agregar_caso("Subgrupo: mano izq. dominante", int(idx[sub][np.argsort(conf[sub])[len(conf[sub]) // 2]]))
    agregar_caso("Fuera del vocabulario (aceptada)", -1, x=Xo[i_oov], objetivo=int(Po[i_oov].argmax()),
                 real=cl_oov[yo[i_oov]], persona=po[i_oov])

    filas_c, mapas = [], []
    for c in casos:
        if c["i"] >= 0:
            i = c["i"]; x = Xte[i]; t = int(pred[i]); real = clases[yte[i]]; persona = pte[i]
            a = A[i]; pr = P[i]
        else:
            x = c["x"]; t = c["objetivo"]; real = c["real"]; persona = c["persona"]
            b = linea_base(x[None], "reposo")
            a = integrated_gradients(m, x[None], np.array([t]), b, N_STEPS)[0][0]
            pr = probabilidades(m, x[None])[0]
        g = agregar(a[None])[0]                                  # [5, 64]
        cu_i = np.abs(g).sum(1) / np.abs(g).sum()
        top3 = np.argsort(-pr)[:3]
        fila = dict(caso=c["tipo"], persona=persona, real=real, real_es=ES.get(real, ""),
                    pred=clases[t], pred_es=ES.get(clases[t], ""), confianza=round(float(pr[t]), 3),
                    top3=" · ".join(f"{clases[k]} {pr[k]:.2f}" for k in top3),
                    acepta_umbral=bool(pr[t] >= umbral),
                    **{f"cuota {n}": round(float(v), 2) for n, v in zip(NOMBRES, cu_i)},
                    cuadro_pico=int(np.abs(g).sum(0).argmax()))
        # Para los errores: ¿qué evidencia había para la clase correcta?
        if c["i"] >= 0 and t != yte[c["i"]]:
            b = linea_base(x[None], "reposo")
            a_real = integrated_gradients(m, x[None], np.array([yte[c["i"]]]), b, N_STEPS)[0][0]
            g_real = agregar(a_real[None])[0]
            fila["evidencia_neta_pred"] = round(float(g.sum()), 2)
            fila["evidencia_neta_real"] = round(float(g_real.sum()), 2)
            fila["parte_que_mas_separa"] = NOMBRES[int(np.argmax(g.sum(1) - g_real.sum(1)))]
        filas_c.append(fila); mapas.append((x, a, g, fila))
    pd.DataFrame(filas_c).to_csv(OUT / "tabla_casos_locales.csv", index=False)

    # ---------------------------------------------------------------- parámetros de diseño
    params = [
        ("Modelo", f"v2 = {info['archivo']} (despliegue.json), sin reentrenar; umbral {umbral}"),
        ("Datos", f"prueba: {len(yte)} videos de 11 personas no vistas; OOV: {len(yo)} videos de 450 señas"),
        ("Método principal", "Integrated Gradients (captum 0.9) sobre el logit de la clase predicha"),
        ("Línea base", "reposo: misma pose promediada en el tiempo y manos no detectadas (0 y bandera 0)"),
        ("Pasos de integración", f"{N_STEPS}, regla de Gauss-Legendre"),
        ("Agregación", "suma con signo por parte del cuerpo (5 grupos) y por cuadro (64)"),
        ("Alternativa", "oclusión de 4 partes × 8 ventanas de 8 cuadros, reemplazo por la línea base"),
        ("Fidelidad", "curva de borrado de los 32 bloques, área bajo la curva (menor = mejor)"),
        ("Semilla y entorno", f"SEED = {SEED}; CPU, 1 hilo, algoritmos deterministas de PyTorch"),
    ]
    pd.DataFrame(params, columns=["parametro", "valor"]).to_csv(OUT / "tabla_parametros.csv", index=False)

    # ================================================================= FIGURAS
    # 1) Casos locales: mapa parte × cuadro (con signo) + esqueleto en el cuadro pico.
    n = len(mapas)
    fig = plt.figure(figsize=(7.4, 1.25 * n + 0.3))
    gs = fig.add_gridspec(n, 2, width_ratios=[4.2, 1], hspace=0.7, wspace=0.08, right=0.88)
    for r, (x, a, g, fila) in enumerate(mapas):
        ax = fig.add_subplot(gs[r, 0])
        v = np.abs(g).max()
        im = ax.imshow(g, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
        ax.set_yticks(range(len(NOMBRES))); ax.set_yticklabels(NOMBRES, fontsize=6.5)
        ax.set_xticks([0, 16, 32, 48, 63]); ax.tick_params(axis="x", labelsize=6.5)
        if r == n - 1:
            ax.set_xlabel("cuadro de la seña (remuestreada a 64)")
        real = fila["real"] + (f" ({fila['real_es']})" if fila["real_es"] else "")
        ax.set_title(f"{r + 1}. {fila['caso']} — real: {real} · pred: {fila['pred']} ({fila['pred_es']}) "
                     f"{fila['confianza']:.2f}", fontsize=7.2, loc="left")
        # esqueleto
        ax2 = fig.add_subplot(gs[r, 1])
        tp = fila["cuadro_pico"]
        w = np.abs(a[max(tp - 2, 0): tp + 3, : X_.N_PTS * 3]).sum(0).reshape(X_.N_PTS, 3).sum(1)
        p = X_.puntos_xy(x, tp)
        # no dibujar una mano que MediaPipe no detectó en ese cuadro (queda en el origen)
        visible = np.ones(X_.N_PTS, bool)
        for k, sl in enumerate([X_.PUNTOS["mano izq."], X_.PUNTOS["mano der."]]):
            if x[tp, X_.N_PTS * 3 + k] < 0.5:
                visible[sl] = False
        for a_, b_ in X_.CONEX_POSE + X_.CONEX_MANOS:
            if visible[a_] and visible[b_]:
                ax2.plot(p[[a_, b_], 0], -p[[a_, b_], 1], color="#bbbbbb", lw=0.6, zorder=1)
        ax2.scatter(p[visible, 0], -p[visible, 1], s=4 + 60 * w[visible] / w.max(), c=w[visible],
                    cmap="Reds", vmin=0, vmax=w.max(), zorder=2, edgecolors="none")
        ax2.set_aspect("equal"); ax2.axis("off")
        ax2.set_title(f"cuadro {tp}", fontsize=6.5)
    cb = fig.colorbar(im, cax=fig.add_axes([0.91, 0.3, 0.012, 0.4]))
    cb.set_label("atribución IG (azul: en contra · rojo: a favor)", fontsize=6.5)
    cb.set_ticks([])
    fig.savefig(OUT / "fig_casos_locales.png", bbox_inches="tight"); plt.close(fig)

    # 2) Global: mapa promedio (manos alineadas a dominante) + cuota por parte vs. logística.
    Gd = np.abs(Gs).copy()
    izq = dom == 0
    Gd[izq, i_izq], Gd[izq, i_der] = Gd[izq, i_der], Gd[izq, i_izq].copy()
    filas_d = ["cara", "brazos", "mano no dominante", "mano dominante", "presencia de mano"]
    M = Gd.mean(0); M = M / M.sum()
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.3), gridspec_kw=dict(width_ratios=[2.2, 1]))
    im = axs[0].imshow(M, aspect="auto", cmap="magma_r", interpolation="nearest")
    axs[0].set_yticks(range(5)); axs[0].set_yticklabels(filas_d, fontsize=7)
    axs[0].set_xlabel("cuadro de la seña"); axs[0].set_title(
        f"|IG| promedio en prueba (n = {len(yte)}), manos alineadas a la dominante", fontsize=7.5, loc="left")
    fig.colorbar(im, ax=axs[0], fraction=0.03, pad=0.01).ax.tick_params(labelsize=6)
    cu_d = np.stack([cu[:, 0], cu[:, 1], c_nodom, c_dom, cu[:, 4]], 1).mean(0)
    # logística: izq./der. fijos -> alinear con la proporción de dominancia del set de prueba
    lr_d = np.array([imp_lr[0], imp_lr[1],
                     (dom == 1).mean() * imp_lr[i_izq] + (dom == 0).mean() * imp_lr[i_der],
                     (dom == 1).mean() * imp_lr[i_der] + (dom == 0).mean() * imp_lr[i_izq], imp_lr[4]])
    yy = np.arange(5)
    axs[1].barh(yy + 0.2, cu_d, 0.4, color="#2f5bd3", label="IG · v2 (Transformer)")
    axs[1].barh(yy - 0.2, lr_d, 0.4, color="#b4690e", label=f"|coef| · logística (top-1 {acc_lr:.2f})")
    axs[1].set_yticks(yy); axs[1].set_yticklabels([]); axs[1].invert_yaxis()
    for k in range(5):
        axs[1].text(cu_d[k] + 0.01, k + 0.2, f"{cu_d[k]:.2f}", va="center", fontsize=5.5)
        axs[1].text(lr_d[k] + 0.01, k - 0.2, f"{lr_d[k]:.2f}", va="center", fontsize=5.5)
    axs[1].set_xlim(0, 0.68)
    axs[1].set_xlabel("proporción de la importancia")
    axs[1].legend(fontsize=5.8, frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.32), ncol=1)
    axs[1].set_title("Cuota por parte del cuerpo", fontsize=7.5, loc="left")
    fig.tight_layout(); fig.savefig(OUT / "fig_global.png", bbox_inches="tight"); plt.close(fig)

    # 3) Fidelidad: curvas de borrado + cordura.
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.2), gridspec_kw=dict(width_ratios=[1.6, 1]))
    colores = {"IG (reposo)": "#2f5bd3", "IG (ceros)": "#7f9be0", "IG (media)": "#a9bbe8",
               "Oclusión": "#1f8a4c", "Azar": "#999999"}
    for k, c in curvas.items():
        axs[0].plot(np.linspace(0, 1, len(c)), c, color=colores[k], lw=1.6 if k == "IG (reposo)" else 1,
                    label=f"{k} · AUC {auc[k]:.3f}")
    axs[0].set_xlabel("fracción de bloques reemplazados por la línea base")
    axs[0].set_ylabel("prob. de la clase predicha"); axs[0].legend(fontsize=6, frameon=False)
    axs[0].set_title("Fidelidad: borrar primero lo más importante", fontsize=7.5, loc="left")
    axs[1].hist(rho_ig_ocl, bins=25, color="#1f8a4c", alpha=0.7, label=f"IG vs oclusión (mediana {np.median(rho_ig_ocl):.2f})")
    axs[1].hist(rho_cordura, bins=25, color="#c0392b", alpha=0.6, label=f"IG vs IG con pesos al azar ({np.median(rho_cordura):.2f})")
    axs[1].set_xlabel("ρ de Spearman entre rankings de bloques"); axs[1].legend(fontsize=5.8, frameon=False)
    axs[1].set_title("Acuerdo entre explicaciones", fontsize=7.5, loc="left")
    fig.tight_layout(); fig.savefig(OUT / "fig_fidelidad.png", bbox_inches="tight"); plt.close(fig)

    # ---------------------------------------------------------------- resumen numérico
    errores = pd.DataFrame(dict(real=[clases[k] for k in yte[~ok]], pred=[clases[k] for k in pred[~ok]]))
    pares = errores.value_counts().head(6)
    res = dict(
        top1=float(ok.mean()), n_test=int(len(ok)), n_errores=int((~ok).sum()), umbral=umbral,
        top1_logistica=acc_lr, costo_ms=costo, segundos_oclusion_total=t_ocl,
        auc_borrado=auc, spearman_ig_oclusion_mediana=float(np.median(rho_ig_ocl)),
        spearman_cordura_mediana=float(np.median(rho_cordura)),
        cuota_global=dict(zip(filas_d, map(float, cu_d))), cuota_logistica=dict(zip(filas_d, map(float, lr_d))),
        pct_mano_izq_dominante=float((dom == 0).mean()),
        oov_aceptadas=float(acepta.mean()), n_oov=int(len(yo)),
        pares_confusion={f"{a} → {b}": int(v) for (a, b), v in pares.items()},
        errores_con_conf_sobre_umbral=float((conf[~ok] >= umbral).mean()),
    )
    (OUT / "resumen.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=1, ensure_ascii=False))
    print(atajos.to_string()); print(pd.DataFrame(filas_c).to_string())


if __name__ == "__main__":
    main()
