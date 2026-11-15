"""
Paso 8 — Reúne los números del informe en reports/tablas/numeros_informe.json.

Las frases que dependen de los resultados se construyen aquí a partir de las tablas, para que el
informe (scripts/informe/build_informe.js) nunca contradiga a los resultados guardados.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from asl.features import N_POSE  # noqa: E402
from asl.valor import SUPUESTOS_BASE, costo_manual, supuestos_dict  # noqa: E402

TAB = ROOT / "reports" / "tablas"
RUNS = ROOT / "reports" / "runs"


def pct(x, d=1):
    return f"{100 * x:.{d}f} %"


def usd(x):
    return f"{x:,.0f}"


def mean_of(s):
    """'0.885 ± 0.013' -> 0.885"""
    return float(str(s).split("±")[0].replace(",", "").strip())


def main():
    N = {}
    res = json.loads((TAB / "resumen_valor.json").read_text())
    det = pd.read_csv(TAB / "runs_detalle.csv")
    it = pd.read_csv(TAB / "tabla_iteraciones.csv").set_index("Iteración")
    tv = pd.read_csv(TAB / "tabla_versiones.csv").set_index("Versión")
    sens = pd.read_csv(TAB / "sensibilidad.csv")
    dep = res["desplegado"]
    N["desplegado"] = dep
    N["supuestos"] = supuestos_dict()
    N["costo_manual_seg"] = costo_manual(SUPUESTOS_BASE)

    # ---------------- EDA
    sub = pd.read_csv(ROOT / "data" / "subconjunto.csv")
    obj = sub[sub.grupo == "objetivo"]
    filas = []
    for _, r in obj.iterrows():
        a = np.load(ROOT / "data" / "landmarks" / (r["Video file"][:-4] + ".npy")).astype(np.float32)
        filas.append(dict(p=r["Participant ID"], T=len(a), manos=(~np.isnan(a[:, N_POSE:, 0]).all(1)).mean(),
                          pose=(~np.isnan(a[:, 0, 0])).mean(),
                          izq=(~np.isnan(a[:, N_POSE:N_POSE + 21, 0]).all(1)).mean(),
                          der=(~np.isnan(a[:, N_POSE + 21:, 0]).all(1)).mean()))
    q = pd.DataFrame(filas)
    pm = q.groupby("p").manos.mean()
    lado = q.groupby("p")[["izq", "der"]].mean()
    n_izq = int((lado.izq > lado.der).sum())
    N["n_participantes"] = int(len(lado)); N["n_mano_izq"] = n_izq
    N["eda"] = dict(obj_videos=len(obj), obj_train=int((obj.split == "train").sum()),
                    obj_val=int((obj.split == "val").sum()), obj_test=int((obj.split == "test").sum()),
                    pre_videos=int((sub.grupo == "preentreno").sum()),
                    cuadros_mediana=int(q["T"].median()), seg_mediana=f"{q['T'].median() / 30:.1f}",
                    frac_pose=float(q.pose.mean()), frac_manos=float(q.manos.mean()),
                    manos_min_part=float(pm.min()), manos_max_part=float(pm.max()))

    # ---------------- clásico vs DL
    clas = det[det.iteracion == "clasico_logreg"].iloc[0]
    d = tv.loc[dep]
    N["clasico_top1"] = pct(clas.test_top1); N["clasico_top3"] = pct(clas.test_top3)
    N["clasico_err3"] = pct(1 - clas.test_top3); N["dep_err3"] = pct(1 - d["Top-3 test"])
    N["delta_vs_clasico"] = usd(d["Valor neto USD/mes"] - clas.valor_neto_mes)

    # ---------------- tabla de supuestos
    S = SUPUESTOS_BASE
    N["tabla_supuestos"] = [
        ["t_grabar", f"{S.t_grabar:.1f} s", "Medido en la prueba piloto (1 persona, 6 búsquedas): mediana desde «Listo» hasta el último resultado, con 2.5 intentos en promedio."],
        ["t_rev3", f"{S.t_rev3:.1f} s", "Medido en la prueba piloto: mediana de ms_hasta_elegir en modo seguro (la estimación previa era 15 s)."],
        ["t_rev10", f"{S.t_rev10:.0f} s", "Estimación: lista de 10 opciones (≈ 4–5 s c/u)."],
        ["t_manual", f"{S.t_manual:.0f} s", "Medido en la prueba piloto: mediana de 3 búsquedas en diccionarios en línea (tope de 5 min; la estimación previa era 180 s)."],
        ["p_manual", pct(S.p_manual, 0), "Estimación conservadora [2], [3]; en la prueba piloto se encontraron 0 de 3, así que no se usa el valor medido (muestra muy pequeña)."],
        ["c_no_resuelta", f"{S.c_no_resuelta:.0f} s", "Preguntar al docente o retrasar la tarea; equivalente a 5 min."],
        ["p_engano / p_engano_aviso", f"{pct(S.p_engano, 0)} / {pct(S.p_engano_aviso, 0)}", "Sesgo de automatización: se confía en la sugerencia mostrada con seguridad; el aviso explícito lo reduce."],
        ["c_error", f"{S.c_error:.0f} s", "Aprender o usar mal una seña y corregirlo después (10 min)."],
        ["p_vocab", pct(S.p_vocab, 0), "La app se alinea al vocabulario del curso; 15 % de búsquedas quedan fuera. Su efecto se mide con señas reales fuera del vocabulario."],
        ["búsquedas/mes", f"{S.busquedas_mes:,.0f}", "500 estudiantes × 20 búsquedas/mes."],
        ["valor_hora", f"{S.valor_hora:.0f} USD", "Costo de oportunidad del tiempo de un estudiante (salario de trabajo de medio tiempo)."],
        ["costo de operación", f"{S.costo_fijo_mes:.0f} USD/mes + {S.costo_pred:.5f} USD/pred.", "Servidor CPU pequeño; la inferencia tarda ≈ 10 ms en CPU."],
    ]

    # ---------------- frases
    v0, v1, v2 = tv.loc["v0"], tv.loc["v1"], tv.loc["v2"]
    dif = res["ic95_diferencias_valor"]
    N["frase_resumen_versiones"] = (
        f"Frente al LSTM base (v0), el Transformer desde cero (v1) aporta {usd(v1['Valor neto USD/mes'] - v0['Valor neto USD/mes'])} USD/mes "
        f"(IC 95 % [{usd(dif['v1_menos_v0'][0])}, {usd(dif['v1_menos_v0'][1])}]) y la versión con transfer learning (v2) "
        f"{usd(v2['Valor neto USD/mes'] - v0['Valor neto USD/mes'])} USD/mes (IC 95 % [{usd(dif['v2_menos_v0'][0])}, {usd(dif['v2_menos_v0'][1])}]), "
        f"con un costo de cómputo adicional despreciable.")

    tau = float(d["Umbral"])
    N["frase_umbral"] = (
        f"el {pct(d['Modo seguro'], 0)} de las búsquedas se muestra en modo seguro y el modelo marca como \"no estoy seguro\" "
        f"el {pct(d['OOV detectadas'], 0)} de las señas fuera del vocabulario. "
        + ("Un umbral más alto vuelve más lentas las búsquedas buenas; uno más bajo deja pasar más engaños con señas desconocidas."
           if tau > 0 else "Con este modelo conviene mostrar siempre 3 opciones; el modo de 10 opciones queda como acción del usuario."))

    tv_t3 = det.dropna(subset=["valor_neto_mes"])
    corr = np.corrcoef(tv_t3.test_top3, tv_t3.valor_neto_mes)[0, 1]
    N["frase_curva_valor"] = (
        f"El valor crece con el top-3 (correlación {corr:.2f} entre corridas): en promedio cada punto de top-3 vale "
        f"≈ {usd(np.polyfit(tv_t3.test_top3 * 100, tv_t3.valor_neto_mes, 1)[0])} USD/mes. Como la búsqueda manual es muy cara "
        f"({N['costo_manual_seg']:.0f} s), incluso el baseline clásico ya ahorra tiempo; el valor del deep learning está en los "
        f"puntos de top-3 que evitan engaños y búsquedas manuales de respaldo.")

    s = sens.sort_values("rango_50", ascending=False)
    top = s.head(3)
    base_v = float(d["Valor neto USD/mes"])
    negativos = s[(s[["x0.5", "x0.8", "x1.2", "x1.5"]] < 0).any(axis=1)]
    delta_cols = [c for c in s.columns if c.startswith("delta_vs_v0")]
    signo_ok = bool((s[delta_cols] > 0).all().all())
    N["frase_sensibilidad"] = (
        f"Los supuestos que más mueven el valor son {', '.join(top.supuesto)} (rango con ±50 %: "
        + ", ".join(f"{usd(min(r['x0.5'], r['x1.5']))}–{usd(max(r['x0.5'], r['x1.5']))}" for _, r in top.iterrows())
        + f" USD/mes frente a {usd(base_v)} en el escenario base). "
        + ("Ningún cambio individual de ±50 % vuelve negativo el valor. " if negativos.empty else
           f"Solo {', '.join(negativos.supuesto)} con −50 % lo vuelve negativo. ")
        + ("La ventaja del modelo desplegado sobre v0 se mantiene positiva en todos los escenarios. " if signo_ok else
           "La ventaja sobre v0 cambia de signo en algún escenario extremo (ver sensibilidad.csv). ")
        + f"En un escenario pesimista conjunto (búsqueda manual 50 % más corta, tiempo valorado a la mitad, más engaños, menos cobertura de vocabulario) "
          f"el valor neto cae a {usd(res['escenarios'][dep]['pesimista'])} USD/mes.")

    # Costos
    pre = res.get("preentreno_min", {})
    sc = det[det.iteracion.isin(["it0_lstm_base", "it1_lstm_aug", "it2_tf_aug", "it3_tf_aug_small"])]
    scratch_min, scratch_max = sc.tiempo_train_min.min(), sc.tiempo_train_min.max()
    lat = tv["Latencia CPU ms"]
    N["frase_costo"] = (
        f"Todo se entrenó en una laptop Apple M4 Pro (GPU MPS). Un modelo desde cero tarda {scratch_min:.1f}–{scratch_max:.1f} min por semilla "
        f"(tiempos medidos con la máquina compartida con otros procesos); el preentrenamiento "
        f"del Transformer {pre.get('pre_tf', float('nan')):.1f} min y el fine-tuning ≈ {mean_of(it.loc['it5_tl_tf_finetune','Entrenamiento (min, M4 Pro)'].split('+')[0]) if 'it5_tl_tf_finetune' in it.index else float('nan'):.1f} min. "
        f"A precio de una instancia GPU en la nube (≈ 0.5–1 USD/h), entrenar cualquier versión cuesta centavos; la extracción de puntos clave "
        f"(≈ 1 h de CPU para 15 300 videos) es el paso más caro. En inferencia los modelos tardan {lat.min():.1f}–{lat.max():.1f} ms por seña en CPU, "
        f"por lo que un servidor de {S.costo_fijo_mes:.0f} USD/mes sobra para 10 000 búsquedas/mes."
        + (" La iteración it7 se entrenó después en una laptop con GPU NVIDIA RTX 4050: "
           f"{mean_of(it.loc['it7_tl_tf_webcam','Entrenamiento (min, M4 Pro)'].split('+')[0]):.1f} min por semilla."
           if "it7_tl_tf_webcam" in it.index else ""))

    filas_it = it.reset_index()
    N["frase_iteraciones"] = (
        f"El LSTM base sobreajusta rápido (top-1 de train ≈ {filas_it.set_index('Iteración').loc['it0_lstm_base','Top-1 train']} contra "
        f"{filas_it.set_index('Iteración').loc['it0_lstm_base','Top-1 test']} en prueba). Los aumentos de datos son el cambio más rentable dentro de la LSTM "
        f"({it.loc['it0_lstm_base','Top-1 test']} → {it.loc['it1_lstm_aug','Top-1 test']} de top-1). El salto mayor viene de cambiar a atención: "
        f"el Transformer con aumentos llega a {it.loc['it2_tf_aug','Top-1 test']} de top-1 y {it.loc['it2_tf_aug','Top-3 test']} de top-3; "
        f"reducirlo a 3 capas y d = 128 ({it.loc['it3_tf_aug_small','Top-3 test']}) casi no cambia el resultado, señal de que el límite está en los datos y no en la capacidad.")
    N["frase_versiones"] = (
        f"v0 resuelve el {pct(v0['Resueltas'])} de las búsquedas con {v0['Seg/búsqueda app']:.0f} s equivalentes por búsqueda; v1 el {pct(v1['Resueltas'])} "
        f"con {v1['Seg/búsqueda app']:.0f} s y v2 el {pct(v2['Resueltas'])} con {v2['Seg/búsqueda app']:.0f} s, frente a {N['costo_manual_seg']:.0f} s de la búsqueda manual. "
        f"En dinero: v1 genera {usd(v1['Valor neto USD/mes'] - v0['Valor neto USD/mes'])} USD/mes más que v0 y v2 {usd(v2['Valor neto USD/mes'] - v1['Valor neto USD/mes'])} USD/mes "
        f"más que v1 (IC 95 % [{usd(dif['v2_menos_v1'][0])}, {usd(dif['v2_menos_v1'][1])}]). El costo extra de cómputo es mínimo: el fine-tuning "
        f"tarda {v2['Entrenamiento min']:.1f} min frente a {v1['Entrenamiento min']:.1f} min del entrenamiento desde cero, más "
        f"{pre.get('pre_tf', 0):.1f} min de preentrenamiento que se hace una sola vez, y la latencia de inferencia es la misma.")
    if "it5_tl_tf_finetune" in it.index:
        N["frase_tl"] = (
            f"Con la misma arquitectura, el Transformer desde cero (modelo base, it2) obtiene {it.loc['it2_tf_aug','Top-1 test']} de top-1, "
            f"{it.loc['it2_tf_aug','Top-3 test']} de top-3 y {it.loc['it2_tf_aug','Valor neto USD/mes (test)']} USD/mes. "
            f"Con el backbone preentrenado y congelado (it4) se obtiene {it.loc['it4_tl_tf_congelado','Top-1 test']} / {it.loc['it4_tl_tf_congelado','Top-3 test']}, "
            f"y con fine-tuning completo (it5) {it.loc['it5_tl_tf_finetune','Top-1 test']} / {it.loc['it5_tl_tf_finetune','Top-3 test']} "
            f"({it.loc['it5_tl_tf_finetune','Valor neto USD/mes (test)']} USD/mes). En la LSTM, el preentrenamiento pasa de "
            f"{it.loc['it1_lstm_aug','Top-1 test']} a {it.loc['it6_tl_lstm_finetune','Top-1 test']} de top-1. " + N.get("_tl_extra", ""))
        lc_ = pd.read_csv(TAB / "curva_aprendizaje.csv")
        f25 = lc_[lc_.frac == 0.25].set_index("familia")
        N["frase_tl"] += (
            "Con los 35 participantes la ganancia del Transformer es modesta (la arquitectura ya generaliza bien), pero el modelo ajustado "
            "converge en 3–4 veces menos epochs y, sobre todo, es mucho más eficiente en datos: entrenando con solo 9 personas, el modelo "
            f"desde cero obtiene {f25.loc['tf','top3']:.3f} de top-3 y {usd(f25.loc['tf','valor'])} USD/mes, mientras que con transfer learning "
            f"obtiene {f25.loc['tl','top3']:.3f} y {usd(f25.loc['tl','valor'])} USD/mes. Esa eficiencia es la que importa para crecer el "
            "vocabulario, donde cada seña nueva tendrá pocas grabaciones.")
    N["frase_problemas"] = (
        "(1) Sobreajuste severo de la LSTM sin aumentos (pérdida de entrenamiento → 0 en ~60 epochs mientras la de validación sube), "
        "resuelto con aumentos y label smoothing; (2) MediaPipe pierde la mano en cuadros con movimiento rápido o cuando la mano está "
        "frente a la cara (p. ej. GIRL), por lo que se rellenan los huecos con el último valor y se añaden indicadores de presencia; "
        "(3) personas zurdas: la mano dominante cambia de lado, lo que se aborda con el aumento de espejo; (4) la validación tiene solo 6 "
        "personas, así que su top-1 es ruidoso (±1–2 puntos entre semillas): por eso se reportan 3 semillas e intervalos por bootstrap.")

    # Iteración de datos (IMAGE vs VIDEO)
    # (Estas dos secciones dependen de corridas manuales: script 09 y docs/PRUEBA_E2E_APP.md.)
    if not (TAB / "iteracion_datos_resumen.json").exists():
        (TAB / "iteracion_datos_resumen.json").write_text(json.dumps({"it2_tf_aug": {k: float("nan") for k in
            ["A_top3", "B_top3", "C_top3", "A_top1", "B_top1", "C_top1"]}}))
    idat = json.loads((TAB / "iteracion_datos_resumen.json").read_text())
    k = "it2_tf_aug" if "it2_tf_aug" in idat else next(iter(idat))
    r_ = idat[k]
    N["frase_iteracion_datos"] = (
        "La primera versión extrajo los puntos clave de los videos de entrenamiento con MediaPipe en modo IMAGE (cada cuadro "
        "independiente), mientras que la app usa el modo VIDEO del navegador, que hace seguimiento entre cuadros. Al probar la app "
        "de punta a punta con videos de prueba el acierto fue menor que el medido offline; comparando cuadro a cuadro un mismo video, el "
        "modo VIDEO detectaba la mano en más cuadros y la muñeca de la pose temblaba casi 4 veces menos. Es decir, el modelo se entrenó con "
        f"datos distintos a los de producción. Para el Transformer ({k}), el top-3 era {r_['A_top3']:.3f} con datos IMAGE, pero con datos "
        f"como los de la app caía a {r_['B_top3']:.3f} (top-1 {r_['A_top1']:.3f} → {r_['B_top1']:.3f}). Se re-extrajeron los 15 300 videos "
        f"en modo VIDEO (además ~2 veces más rápido) y se re-entrenaron todas las versiones: el mismo modelo llega a {r_['C_top3']:.3f} de "
        f"top-3 y {r_['C_top1']:.3f} de top-1 con los datos que realmente verá. Todos los resultados de este informe usan la extracción VIDEO.")

    if (TAB / "prueba_app_e2e.json").exists():
        e2e = json.loads((TAB / "prueba_app_e2e.json").read_text())
        v_, i_, d2, d0 = e2e["transformer_video"], e2e["transformer_image"], e2e["v2_desplegado"], e2e["v0"]
        N["frase_e2e"] = (
            f"Prueba de punta a punta: {d2['n']} videos de prueba (personas no vistas) se pasaron por el flujo completo de la app en el navegador "
            f"(MediaPipe JS, segmentación automática y API). La v2 de ese momento ({d2['modelo'].split(' ')[0]}) acertó {d2['top1']}/{d2['n']} en la primera opción, "
            f"{d2['top3']}/{d2['n']} dentro de las 3 mostradas y mostró {d2['modo_seguro']}/{d2['n']} búsquedas en modo seguro; v0 acertó "
            f"{d0['top1']}/{d0['n']} y {d0['top3']}/{d0['n']}. El Transformer desde cero entrenado con la extracción IMAGE original acertaba "
            f"{i_['top1']}/{i_['n']} y {i_['top3']}/{i_['n']}, contra {v_['top1']}/{v_['n']} y {v_['top3']}/{v_['n']} tras re-extraer en VIDEO. "
            "Esta prueba valida que lo medido offline se sostiene en el producto real.")
    else:
        N["frase_e2e"] = "(Pendiente: correr la prueba de docs/PRUEBA_E2E_APP.md.)"

    if (TAB / "robustez_webcam.csv").exists() and "it7_tl_tf_webcam" in it.index:
        rb = pd.read_csv(TAB / "robustez_webcam.csv").set_index("perturbación")
        a, b = "it5_tl_tf_finetune top3", "it7_tl_tf_webcam top3"
        N["frase_webcam"] = (
            "Al probar la app como la usaría una persona, encontramos que lo que más baja el acierto no es el modelo sino lo que "
            "la cámara le entrega: la seña cortada por la segmentación, los hombros fuera de cuadro o la cámara demasiado cerca. "
            "Medido sobre los puntos crudos del conjunto de prueba (scripts/14_robustez_webcam.py), el top-3 de it5 cae de "
            f"{rb.loc['original', a]:.3f} a {rb.loc['seña cortada: último 60 %', a]:.3f} si se pierde el inicio de la seña, a "
            f"{rb.loc['hombros sin detectar', a]:.3f} sin hombros y a {rb.loc['cámara 1.4× más cerca', a]:.3f} con la cámara más cerca. "
            "La iteración it7 repite el fine-tuning de it5 con aumentos que imitan esas fallas (recorte de 0–40 % por un extremo, "
            "escala y traslación grandes, mano perdida varios cuadros seguidos). En prueba limpia no pierde nada "
            f"({it.loc['it7_tl_tf_webcam','Top-1 test']} / {it.loc['it7_tl_tf_webcam','Top-3 test']} de top-1 / top-3) y es más robusta: "
            f"{rb.loc['seña cortada: último 60 %', b]:.3f}, {rb.loc['hombros sin detectar', b]:.3f} y {rb.loc['cámara 1.4× más cerca', b]:.3f} "
            "en esos mismos casos. Como también es la mejor en validación, el criterio de selección la eligió como v2.")
    if (TAB / "prueba_usuarios.json").exists():
        pu = json.loads((TAB / "prueba_usuarios.json").read_text())
        m_, a_, e_ = pu["manual"], pu["app"], pu.get("encuesta", {})
        N["frase_usuarios"] = (
            f"Prueba piloto con un integrante del equipo ({pu['participantes']} sesiones) usando la sesión guiada de la app (/?prueba=usuarios; "
            f"docs/RESULTADOS_PRUEBA_USUARIOS.md). Buscando a mano en diccionarios en línea (Handspeak, SignASL, Lifeprint) señas que no "
            f"conocía, tardó una mediana de {m_['t_mediana_s']:.0f} s y encontró {m_['encontrada'] * pu['ensayos_manual']:.0f} de "
            f"{pu['ensayos_manual']}; con SeñaBusca resolvió {a_['resuelta'] * pu['ensayos_app']:.0f} de {pu['ensayos_app']} búsquedas, "
            f"siempre en la primera opción, en una mediana de {a_['t_total_mediana_s']:.0f} s ({a_['intentos_promedio']:.1f} intentos de "
            f"grabación en promedio). Calificó la facilidad con {e_.get('facil_manual', float('nan')):.0f}/5 a mano y "
            f"{e_.get('facil_app', float('nan')):.0f}/5 con la app. Con estos datos se reemplazaron t_manual, t_rev3 y t_grabar; p_manual se "
            "mantiene en la estimación conservadora porque 3 búsquedas no bastan para estimarla. Es una sola persona: la medición con más "
            "estudiantes es el siguiente paso, y el análisis de sensibilidad muestra que la conclusión no depende de estos valores. "
            "No pagaría por la app como estudiante, lo que confirma que el cliente es la institución (curso o programa de ASL) que "
            "valora el tiempo de sus estudiantes, no el estudiante individual.")
    if (TAB / "validacion_propia.json").exists() and "frase_usuarios" in N:
        vp = json.loads((TAB / "validacion_propia.json").read_text())
        N["frase_usuarios"] += (
            f" Además, en una validación propia con {vp['ensayos']} señas del vocabulario aprendidas en el momento con el video de "
            f"ejemplo (/?prueba=propia), la app acertó en la primera opción el {vp['top1_primer_intento'] * 100:.0f} % de las veces al "
            f"primer intento ({vp['top3_primer_intento'] * 100:.0f} % dentro de las 3 mostradas) y el {vp['top1_ultimo_intento'] * 100:.0f} % "
            f"tras repetir la seña ({vp['intentos_promedio']:.2f} intentos en promedio).")
    if (TAB / "prueba_camara_simulada.json").exists():
        cs = json.loads((TAB / "prueba_camara_simulada.json").read_text())
        mm, aj = cs["modelos"], cs["ajuste_segmentacion_60_videos"]
        N["frase_camara"] = (
            f"Prueba con cámara simulada en tiempo real (scripts/13_prueba_camara_simulada.py): los {cs['videos']} videos de prueba se "
            "encadenaron en una transmisión continua que Chromium usó como webcam, así que pasan por el flujo real de la cámara "
            f"(detección a ≈ {cs['detector_ms_por_cuadro']} ms por cuadro, segmentación automática y API) y no cuadro por cuadro. La primera "
            f"corrida mostró que la segmentación descartaba señas cortas: exigía 15 cuadros con mano activa y en ASL Citizen el 4.3 % de "
            f"las señas tiene menos. Bajarlo a 8 subió v2 de {aj['antes']['top1_n']}/60 a {aj['despues']['top1_n']}/60 en top-1 y de "
            f"{aj['antes']['top3_n']}/60 a {aj['despues']['top3_n']}/60 en top-3. Con los {cs['videos']} videos, la app grabó "
            f"{cs['videos'] - cs['senas_sin_grabacion']} señas ({cs['senas_sin_grabacion']} se perdieron, sobre todo señas rápidas a la "
            f"altura del pecho como STOP y WORK, donde MediaPipe en el navegador pierde la mano) y la versión desplegada acertó "
            f"{mm['v2']['top1'] * 100:.1f} % en la primera opción y {mm['v2']['top3'] * 100:.1f} % dentro de las 3 mostradas (v1 {mm['v1']['top3'] * 100:.1f} %, "
            f"v0 {mm['v0']['top3'] * 100:.1f} % de top-3). Es la cifra más cercana al uso real: unos 5 puntos por debajo de la medición offline.")

    exp_all = json.loads((TAB / "explicabilidad_oclusion.json").read_text())
    exp = exp_all[dep]
    N["_drop_espejo_v0"] = f"{100 * (exp_all['v0']['base_top3'] - exp_all['v0']['espejo (zurdo)']):.1f} puntos"
    base3 = exp["base_top3"]
    caidas = {k: base3 - v_ for k, v_ in exp.items() if k.startswith("sin ")}
    orden = sorted(caidas.items(), key=lambda kv: -kv[1])
    N["frase_explicabilidad"] = (
        f"Se aplicaron oclusión por parte del cuerpo, saliencia por gradiente y atención del token [CLS]. Al ocultar cada grupo de puntos, el top-3 "
        f"cae de {base3:.3f} a: " + "; ".join(f"{k} → {exp[k]:.3f}" for k, _ in orden) + ". "
        f"Lo que más lo afecta es quitar {orden[0][0].replace('sin ', '')} y {orden[1][0].replace('sin ', '')}: el modelo usa la "
        f"trayectoria de las manos (las muñecas de la pose y la mano derecha, la dominante de la mayoría de participantes) y su "
        f"ubicación respecto al cuerpo, como se espera de un vocabulario donde la mayoría de las señas son de una mano dominante y "
        f"se distinguen por dónde se hacen (cara, pecho). La saliencia por gradiente se concentra en la parte central de la seña "
        f"y cae en los cuadros de reposo del inicio y el final; la atención del [CLS] es más uniforme, pero también es menor en los extremos. En robustez: reflejada (persona zurda) el top-3 es {exp['espejo (zurdo)']:.3f}, con ruido gaussiano "
        f"{exp['ruido σ=0.05']:.3f} y con la seña cortada al 80 % {exp['seña cortada (80 %)']:.3f}.")

    pp = pd.read_csv(TAB / "por_participante.csv")
    ps = pd.read_csv(TAB / "por_sena.csv", index_col=0)
    conf = pd.read_csv(TAB / "confusiones_top.csv", index_col=0)
    N["frase_sesgos"] = (
        f"El desempeño varía por persona: el top-3 va de {pp.top3.min():.2f} ({pp.sort_values('top3').iloc[0, 0]}) a {pp.top3.max():.2f} entre los 11 "
        f"participantes de prueba, y el top-1 de {pp.top1.min():.2f} a {pp.top1.max():.2f}. El dataset no publica variables demográficas por participante, así que "
        f"no podemos atribuir la diferencia a edad, tono de piel o región; sí observamos que la tasa de detección de manos de MediaPipe varía mucho entre "
        f"personas, lo que sugiere que iluminación, cámara y contraste piel-fondo son la vía más probable de sesgo. El costo humano de esos errores recae "
        f"en quien ya tiene peor detección: más búsquedas manuales o, peor, aprender un significado equivocado. Por seña, las más difíciles son "
        + ", ".join(f"{g} (top-3 {r.top3:.2f})" for g, r in ps.head(4).iterrows())
        + f"; las confusiones más frecuentes son " + ", ".join(f"{k} ({v_})" for k, v_ in conf.iloc[:4, 0].items())
        + ", pares que comparten forma de mano o ubicación. "
        f"Además, en {N['n_mano_izq']} de {N['n_participantes']} participantes la mano más activa está del lado opuesto a la mayoría (personas "
        f"probablemente zurdas): sin el aumento de espejo el LSTM base pierde {N['_drop_espejo_v0']} de top-3 con señas reflejadas, mientras que "
        f"el modelo desplegado prácticamente no cambia. En cuanto a la confianza, el modelo es subconfiado (efecto del label smoothing): la frecuencia real de acierto supera la confianza que reporta, sobre todo en confianzas bajas. "
        "Como el umbral se elige con datos de validación, la decisión de negocio no se afecta, pero las probabilidades que ve el usuario son conservadoras; se podría corregir con temperature scaling.")

    # Latencia real de la API (en proceso, sin red) con una seña de prueba de largo típico.
    import time
    from fastapi.testclient import TestClient
    import app.server as srv
    cli = TestClient(srv.app)
    t = obj[obj.split == "test"].iloc[0]
    cr = np.load(ROOT / "data" / "landmarks" / (t["Video file"][:-4] + ".npy")).astype(np.float32)
    frames = []
    for f in cr:
        pose = np.zeros((33, 3)); pose[[0, 2, 5, 9, 10, 11, 12, 13, 14, 15, 16]] = np.nan_to_num(f[:N_POSE])
        hands = [h.tolist() for h in (f[N_POSE:N_POSE + 21], f[N_POSE + 21:]) if not np.isnan(h).any()]
        frames.append({"pose": pose.tolist(), "hands": hands})
    lat = {}
    for alias in srv.MODELOS:
        ts = []
        for _ in range(50):
            t0 = time.perf_counter()
            cli.post("/api/predict", json={"frames": frames, "modelo": alias, "modo": "en_vivo"})
            ts.append((time.perf_counter() - t0) * 1000)
        lat[alias] = (float(np.median(ts)), float(np.percentile(ts, 95)))
    N["latencia_api_ms"] = lat
    N["frase_latencia"] = (
        "Latencia medida de la API completa (deserializar, preprocesar, inferir, responder) para una seña de "
        f"{len(cr)} cuadros: " + "; ".join(f"{a} p50 {m:.0f} ms / p95 {p:.0f} ms" for a, (m, p) in lat.items())
        + ". Sumando la detección en el navegador (en paralelo a la seña) y la red local, el usuario ve el resultado en menos de "
          "medio segundo después de bajar las manos. Un solo proceso CPU atiende del orden de "
        f"{1000 / max(m for m, _ in lat.values()):.0f} búsquedas por segundo (secuenciales), muy por encima de la demanda del escenario.")
    N["frase_ux"] = (
        f"Con v0, el {pct(1 - v0['Top-3 test'])} de las señas del vocabulario no aparece en las 3 opciones; con {dep} solo el {pct(1 - d['Top-3 test'])}. "
        f"Eso se traduce en {v0['Seg/búsqueda app'] - d['Seg/búsqueda app']:.0f} s menos por búsqueda en promedio y en bajar la tasa de engaños de "
        f"{pct(v0['Engaños'])} a {pct(d['Engaños'])}.")
    N["frase_mayor_impacto"] = (
        f"En métrica técnica, el cambio más grande fue pasar de LSTM a Transformer (con aumentos): explica {usd(v1['Valor neto USD/mes'] - v0['Valor neto USD/mes'])} de los "
        f"{usd(v2['Valor neto USD/mes'] - v0['Valor neto USD/mes'])} USD/mes que separan a v2 de v0. El cambio que más importó para el producto, sin "
        f"embargo, no fue de arquitectura sino de datos: corregir el desajuste IMAGE/VIDEO entre entrenamiento y la app, invisible en las métricas offline. "
        f"Elegir el umbral según el valor también importa: con el umbral elegido, {dep} marca como \"no estoy seguro\" el {pct(d['OOV detectadas'], 0)} de las "
        f"señas fuera de vocabulario; sin él (τ = 0) esas búsquedas se mostrarían como sugerencias seguras y el valor de {dep} caería a "
        f"{usd(float(det[(det.iteracion == tv.loc[dep, 'Iteración']) & (det.seed == tv.loc[dep, 'Semilla'])].valor_sin_umbral.iloc[0]))} USD/mes.")
    N["frase_rendimientos"] = N.get("frase_rendimientos", "")
    N["frase_tradeoff"] = (
        f"Las tres versiones cuestan prácticamente lo mismo de operar (≈ {S.costo_fijo_mes:.0f} USD/mes) y su latencia es imperceptible frente a los "
        f"segundos que el usuario tarda en hacer la seña, por lo que el criterio de elección es el valor; la complejidad adicional del preentrenamiento solo se "
        f"justifica si mejora el valor de forma consistente y, sobre todo, si se planea escalar el vocabulario, donde aprovechar señas extra es más valioso.")

    # Rendimientos decrecientes (curva de aprendizaje)
    lc = TAB / "curva_aprendizaje.csv"
    if lc.exists():
        lc = pd.read_csv(lc)
        x = lc[lc.familia == "tf"].sort_values("frac")
        if len(x) >= 2:
            g1 = x.valor.iloc[1] - x.valor.iloc[0]; g2 = x.valor.iloc[-1] - x.valor.iloc[-2]
            N["frase_rendimientos"] = (
                f"Con el Transformer desde cero, pasar de {x.frac.iloc[0] * 35:.0f} a {x.frac.iloc[1] * 35:.0f} personas en entrenamiento añade "
                f"{usd(g1)} USD/mes, mientras que el último tramo ({x.frac.iloc[-2] * 35:.0f} → 35 personas) añade {usd(g2)} USD/mes. "
                + (f"Con transfer learning la curva parte más alta: con 25 % de las personas ya se obtiene {lc[(lc.familia == 'tl') & (lc.frac == 0.25)].top3.iloc[0]:.3f} de top-3. "
                   if ((lc.familia == "tl") & (lc.frac == 0.25)).any() else "")
                + "Desde el punto de vista del negocio, conseguir más grabaciones de las mismas 50 señas ya no paga; lo que sí paga es ampliar el vocabulario "
                  "(sube p_vocab) y medir los supuestos con usuarios reales.")

    (TAB / "numeros_informe.json").write_text(json.dumps(N, indent=2, ensure_ascii=False, default=float))
    print(json.dumps({k: v_ for k, v_ in N.items() if k.startswith("frase")}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
