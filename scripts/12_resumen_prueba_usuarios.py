"""
Resume la sesión de prueba de la app (logs/prueba_usuarios.jsonl) en los supuestos MEDIDOS del
modelo de valor y en tablas anónimas (sin nombres, solo P1, P2…).

  modo usuarios -> reports/tablas/prueba_usuarios.json + docs/RESULTADOS_PRUEBA_USUARIOS.md
  modo propia   -> reports/tablas/validacion_propia.json

Uso:  python scripts/12_resumen_prueba_usuarios.py [--excluir P3 P5] [--excluir-ensayos P1:manual]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs" / "prueba_usuarios.jsonl"
TABLES = ROOT / "reports" / "tablas"
LIMIT_S = 300


def median(x):
    x = [v for v in x if v is not None and not pd.isna(v)]
    return float(np.median(x)) if x else None


def iqr(x):
    x = [v for v in x if v is not None and not pd.isna(v)]
    return [float(np.percentile(x, 25)), float(np.percentile(x, 75))] if x else None


def users_summary(ev: pd.DataFrame) -> dict:
    tr = ev[ev.tipo == "ensayo"]
    man, app = tr[tr.tarea == "manual"], tr[tr.tarea == "app"]
    sv = ev[ev.tipo == "encuesta"]
    chosen = app[app.rank_elegido.notna()]
    safe = chosen[chosen.modo_app == "seguro"]
    wrong = app[app.glosa_elegida.notna() & (app.glosa_elegida != app.glosa)]
    out = dict(
        participantes=int(tr.participante.nunique()),
        ensayos_manual=len(man), ensayos_app=len(app),
        manual=dict(t_mediana_s=median(man.t_s.clip(upper=LIMIT_S)), t_iqr_s=iqr(man.t_s.clip(upper=LIMIT_S)),
                    encontrada=float(man.encontrada.mean()) if len(man) else None),
        app=dict(t_total_mediana_s=median(app.t_s), t_total_iqr_s=iqr(app.t_s),
                 resuelta=float(app.correcta.mean()) if len(app) else None,
                 intentos_promedio=float(app.intentos.mean()) if len(app) else None,
                 t_primer_resultado_mediana_s=median(app.t_primer_resultado_s),
                 t_eleccion_mediana_s=median(chosen.ms_hasta_elegir / 1000),
                 t_eleccion_modo_seguro_mediana_s=median(safe.ms_hasta_elegir / 1000),
                 rank_elegido=chosen.rank_elegido.value_counts().sort_index().to_dict(),
                 eligio_opcion_equivocada=len(wrong)),
        supuestos_medidos=dict(
            t_manual=median(man.t_s.clip(upper=LIMIT_S)),
            p_manual=float(man.encontrada.mean()) if len(man) else None,
            t_rev3=median(safe.ms_hasta_elegir / 1000),
            t_grabar=median(app.t_ultimo_resultado_s)),
    )
    if len(sv):
        scales = ["facil_manual", "facil_app", "confianza", "usaria"]
        pay = sv.pagaria_usd_mes.dropna() if "pagaria_usd_mes" in sv else pd.Series(dtype=float)
        out["encuesta"] = dict(
            n=len(sv), **{k: float(sv[k].mean()) for k in scales if k in sv and sv[k].notna().any()},
            pagaria_mediana_usd=median(pay), pagaria_fraccion=float((pay > 0).mean()) if len(pay) else None)
        out["comentarios"] = [dict(participante=p, texto=c) for p, c in zip(sv.participante, sv.get("comentario", []))
                              if isinstance(c, str) and c]
    return out


def first_attempts(ev: pd.DataFrame) -> list:
    """Top-3 del primer intento de cada ensayo, a partir de logs/segmentacion.jsonl (requiere ?debug=1)."""
    seg_path = ROOT / "logs" / "segmentacion.jsonl"
    if not seg_path.exists():
        return []
    sent = [e for e in (json.loads(l) for l in seg_path.read_text().splitlines() if l.strip()) if e["ev"] == "enviada"]
    out, prev = [], ev[ev.tipo == "inicio"].ts.min()
    for _, t in ev[ev.tipo == "ensayo"].sort_values("ts").iterrows():
        att = [s for s in sent if prev < s["ts"] <= t.ts]
        prev = t.ts
        out.append((t.glosa, att[0]["top3"] if att else []))
    return out


def own_summary(ev: pd.DataFrame) -> dict:
    app = ev[(ev.tipo == "ensayo") & (ev.tarea == "app")]
    first = first_attempts(ev)
    shown = app.top3.apply(lambda t: t if isinstance(t, list) else [])
    top1 = [bool(t) and t[0] == g for t, g in zip(shown, app.glosa)]
    top3 = [g in t for t, g in zip(shown, app.glosa)]
    return dict(ensayos=len(app), top1_ultimo_intento=float(np.mean(top1)) if len(app) else None,
                top3_ultimo_intento=float(np.mean(top3)) if len(app) else None,
                eligio_correcta=float(app.correcta.mean()) if len(app) else None,
                intentos_promedio=float(app.intentos.mean()) if len(app) else None,
                t_total_mediana_s=median(app.t_s),
                top1_primer_intento=float(np.mean([bool(t) and t[0] == g for g, t in first])) if first else None,
                top3_primer_intento=float(np.mean([g in t for g, t in first])) if first else None,
                fallos=[dict(glosa=g, top3=t) for g, t, ok in zip(app.glosa, shown, top3) if not ok])


def fmt(v, d=1):
    return "–" if v is None else (f"{v:.{d}f}" if isinstance(v, float) else str(v))


def write_markdown(s: dict, path: Path):
    m, a, sm = s["manual"], s["app"], s["supuestos_medidos"]
    lines = [
        "# Resultados de la prueba con usuarios",
        "",
        f"Generado por `scripts/12_resumen_prueba_usuarios.py` a partir de la sesión guiada de la app "
        f"(`/?prueba=usuarios`). {s['participantes']} sesiones (P1, P2…), {s['ensayos_manual']} búsquedas manuales "
        f"y {s['ensayos_app']} con SeñaBusca. Datos anónimos (P1, P2…); los registros crudos no se publican.",
        "",
        "| Métrica | Búsqueda manual (tarea A) | SeñaBusca (tarea B) |",
        "|---|---|---|",
        f"| Tiempo por búsqueda, mediana [RIC] | {fmt(m['t_mediana_s'], 0)} s {m['t_iqr_s'] and [round(x) for x in m['t_iqr_s']]} "
        f"| {fmt(a['t_total_mediana_s'], 0)} s {a['t_total_iqr_s'] and [round(x) for x in a['t_total_iqr_s']]} |",
        f"| Búsquedas resueltas | {fmt(m['encontrada'] and m['encontrada'] * 100, 0)} % | {fmt(a['resuelta'] and a['resuelta'] * 100, 0)} % |",
        f"| Intentos de grabación por búsqueda | – | {fmt(a['intentos_promedio'], 2)} |",
        f"| Eligió una opción equivocada | – | {a['eligio_opcion_equivocada']} |",
        "",
        "## Supuestos del modelo de valor medidos",
        "",
        "| Supuesto | Medido | Cómo |",
        "|---|---|---|",
        f"| `t_manual` | {fmt(sm['t_manual'], 0)} s | mediana del tiempo de la tarea A (tope {LIMIT_S} s) |",
        f"| `p_manual` | {fmt(sm['p_manual'], 2)} | fracción de la tarea A con la respuesta correcta |",
        f"| `t_rev3` | {fmt(sm['t_rev3'], 1)} s | mediana de `ms_hasta_elegir` en modo seguro |",
        f"| `t_grabar` | {fmt(sm['t_grabar'], 1)} s | mediana desde «Listo» hasta el último resultado (incluye reintentos) |",
    ]
    if "encuesta" in s:
        e = s["encuesta"]
        lines += ["", "## Encuesta (1–5)", "", "| Pregunta | Promedio |", "|---|---|"]
        for k, txt in [("facil_manual", "Facilidad buscando a mano"), ("facil_app", "Facilidad con SeñaBusca"),
                       ("confianza", "Confiaría sin revisar el ejemplo"), ("usaria", "La usaría en un curso")]:
            lines.append(f"| {txt} | {fmt(e.get(k), 1)} |")
        lines += ["", f"Pagaría algo al mes: {fmt(e['pagaria_fraccion'] and e['pagaria_fraccion'] * 100, 0)} % "
                      f"(mediana {fmt(e['pagaria_mediana_usd'], 1)} USD)."]
        if s.get("comentarios"):
            lines += ["", "## Comentarios", ""] + [f"* {c['participante']}: «{c['texto']}»" for c in s["comentarios"]]
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excluir", nargs="*", default=[])
    ap.add_argument("--excluir-ensayos", nargs="*", default=[], help="participante:tarea o participante:encuesta, p. ej. P1:manual")
    args = ap.parse_args()
    if not LOG.exists():
        raise SystemExit(f"No existe {LOG}: corre la sesión en /?prueba=usuarios o /?prueba=propia")
    ev = pd.DataFrame([json.loads(l) for l in LOG.read_text().splitlines() if l.strip()])
    ev = ev[~ev.participante.isin(args.excluir)]
    for col in ["tarea", "rank_elegido", "modo_app", "glosa_elegida", "top3", "comentario"]:
        if col not in ev:
            ev[col] = None
    for item in args.excluir_ensayos:
        who, task = item.split(":")
        ev = ev[~((ev.participante == who) & ((ev.tarea == task) | (ev.tipo == task)))]
    TABLES.mkdir(parents=True, exist_ok=True)
    users = ev[ev.modo == "usuarios"]
    if (users.tipo == "ensayo").any():
        s = users_summary(users)
        (TABLES / "prueba_usuarios.json").write_text(json.dumps(s, indent=2, ensure_ascii=False))
        write_markdown(s, ROOT / "docs" / "RESULTADOS_PRUEBA_USUARIOS.md")
        print(json.dumps(s["supuestos_medidos"], indent=2))
    own = ev[ev.modo == "propia"]
    if (own.tipo == "ensayo").any():
        s = own_summary(own)
        (TABLES / "validacion_propia.json").write_text(json.dumps(s, indent=2, ensure_ascii=False))
        print(json.dumps({k: v for k, v in s.items() if k != "fallos"}, indent=2))


if __name__ == "__main__":
    main()
