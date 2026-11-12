"""Diagrama de la arquitectura desplegada (reports/figuras/arquitectura.png)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
fig, ax = plt.subplots(figsize=(11, 4.4), dpi=150)
ax.set_xlim(0, 110); ax.set_ylim(0, 44); ax.axis("off")


def caja(x, y, w, h, titulo, texto, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5",
                                fc=color, ec="#333", lw=0.8))
    ax.text(x + w / 2, y + h - 2.2, titulo, ha="center", va="top", fontsize=9, weight="bold")
    ax.text(x + w / 2, y + h - 6.0, texto, ha="center", va="top", fontsize=6.3, linespacing=1.35)


def flecha(x1, y1, x2, y2, txt=""):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", color="#333", lw=1))
    if txt:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 2.0, txt, ha="center", fontsize=6.8, color="#444")


# Navegador
ax.add_patch(FancyBboxPatch((1, 17), 47, 26, boxstyle="round,pad=0.3", fc="none", ec="#2f5bd3", ls="--"))
ax.text(2.5, 41.5, "NAVEGADOR DEL USUARIO (el video nunca sale del equipo)", fontsize=7.5, color="#2f5bd3", weight="bold")
caja(3, 20, 12, 17, "Cámara", "getUserMedia\n640×480, ~30 fps\n\no video subido\n(respaldo)", "#eef2fb")
caja(18, 20, 13, 17, "MediaPipe Tasks", "HandLandmarker ×2\nPoseLandmarker lite\n(WASM/GPU)\n≈15 ms/cuadro", "#eef2fb")
caja(34, 20, 12, 17, "Segmentación", "inicia: manos\nvisibles 4 cuadros\ntermina: sin\nmanos 12 cuadros", "#eef2fb")
flecha(15.5, 28.5, 17.5, 28.5); flecha(31.5, 28.5, 33.5, 28.5)

# Servidor
ax.add_patch(FancyBboxPatch((51, 17), 57, 26, boxstyle="round,pad=0.3", fc="none", ec="#1f8a4c", ls="--"))
ax.text(52.5, 41.5, "SERVIDOR FastAPI (CPU, 1 proceso)", fontsize=7.5, color="#1f8a4c", weight="bold")
caja(53, 20, 12, 17, "Preproceso", "asignar manos\nrecortar reposo\nnormalizar\n(hombros)\n→ 64 × 161", "#eaf6ef")
caja(68, 20, 13, 17, "Modelo", "LSTM o\nTransformer\n(v0 / v1 / v2,\nseleccionable)", "#eaf6ef")
caja(84, 20, 11, 17, "Decisión", "conf = Σ top-3\n≥ τ: 3 opciones\n< τ: 10 opciones\n\"no estoy\nseguro\"", "#eaf6ef")
caja(97, 20, 9.5, 17, "Registro", "app_eventos\n.jsonl\n(monitoreo y\nfeedback)", "#eaf6ef")
flecha(46.5, 28.5, 52.5, 28.5); ax.text(49.5, 14.5, "JSON de\nlandmarks", ha="center", fontsize=6.5, color="#444")
flecha(65.5, 28.5, 67.5, 28.5); flecha(81.5, 28.5, 83.5, 28.5); flecha(95.5, 28.5, 96.5, 28.5)

# Respuesta
caja(30, 2, 50, 10, "Interfaz de resultados",
     "top-3 (o top-10) con significado en español, barra de probabilidad, esqueleto animado de ejemplo,\n"
     "botones \"Es esta\" / \"Ninguna es\" → feedback etiquetado para re-entrenar y monitorear", "#fdf3e4")
flecha(89, 19.5, 75, 12.5, "top-10 + confianza (≈10 ms)")

# Offline
ax.text(1, 11, "Fuera de línea:", fontsize=7.5, weight="bold")
ax.text(1, 1.5, "ASL Citizen → MediaPipe →\nentrenamiento (MPS) →\nmodelo de valor → τ",
        fontsize=6.8, color="#444", linespacing=1.3)
fig.savefig(ROOT / "reports" / "figuras" / "arquitectura.png", bbox_inches="tight")
print("ok")
