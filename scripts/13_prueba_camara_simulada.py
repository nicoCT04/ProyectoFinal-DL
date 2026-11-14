"""
Prueba de la app con CÁMARA SIMULADA en tiempo real (complementa docs/PRUEBA_E2E_APP.md).

Los videos de prueba (personas no vistas) se encadenan en una transmisión continua que Chromium usa
como webcam falsa: pasan por el flujo real de la cámara (MediaPipe JS en tiempo real, segmentación
automática y /api/predict) y no cuadro por cuadro. Después, cada segmento que la app grabó
(logs/capturas/) se evalúa con todos los modelos, así la comparación usa exactamente las mismas entradas.

Requisitos: videos de prueba (scripts/11), ffmpeg, Chromium y `pip install playwright`
(se usa el Chromium del sistema, no hace falta `playwright install`). La app debe estar corriendo.

Uso:  python scripts/13_prueba_camara_simulada.py [--n 616] [--url http://localhost:8000]
                                                  [--chromium /usr/bin/chromium] [--headless]
Salida: reports/tablas/prueba_camara_simulada.json
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.server import Frame, frames_a_array  # noqa: E402
from asl.features import preparar  # noqa: E402
from asl.models import construir  # noqa: E402

VIDEOS = ROOT / "data" / "raw" / "ASL_Citizen" / "videos"
PAD_S = 1.2
LAG_S = 0.8


def build_stream(n: int, seed: int, out: Path) -> list[dict]:
    """Encadena n videos de prueba (640x480, 30 fps) con 1.2 s de reposo antes y después de cada seña."""
    out.mkdir(parents=True, exist_ok=True)
    sub = pd.read_csv(ROOT / "data" / "subconjunto.csv")
    test = sub[(sub.grupo == "objetivo") & (sub.split == "test")]
    test = test.sample(min(n, len(test)), random_state=seed)
    meta, parts, t0 = [], [], 0.0
    for i, (f, g) in enumerate(zip(test["Video file"], test.Gloss)):
        clip = out / f"c{i:03d}.mp4"
        vf = ("fps=30,scale=640:480:force_original_aspect_ratio=decrease,pad=640:480:(ow-iw)/2:(oh-ih)/2,"
              f"tpad=start_mode=clone:start_duration={PAD_S}:stop_mode=clone:stop_duration={PAD_S}")
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(VIDEOS / f), "-vf", vf, "-an",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", str(clip)], check=True)
        d = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                           "-of", "csv=p=0", str(clip)]).decode())
        meta.append(dict(i=i, file=f, gloss=g, start=t0, end=t0 + d))
        parts.append(f"file '{clip}'")
        t0 += d
    (out / "list.txt").write_text("\n".join(parts))
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(out / "list.txt"),
                    "-c:v", "mjpeg", "-q:v", "7", "-pix_fmt", "yuvj420p", "-r", "30", str(out / "stream.mjpeg")],
                   check=True)
    for c in out.glob("c*.mp4"):
        c.unlink()
    return meta


def run_browser(stream: Path, duration: float, url: str, chromium: str, headless: bool) -> list[dict]:
    from playwright.sync_api import sync_playwright
    preds = []
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=chromium, headless=headless, args=[
            "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
            f"--use-file-for-fake-video-capture={stream}", "--ignore-gpu-blocklist", "--use-angle=gl"])
        pg = b.new_page()
        t_cam = [0.0]

        def on_response(r):
            if not (r.url.endswith("/api/predict") and r.request.method == "POST"):
                return
            body = json.loads(r.request.post_data or "{}")
            if body.get("modo", "grabacion") != "grabacion":
                return
            j = r.json()
            preds.append(dict(t=time.time() - t_cam[0], n=len(body["frames"]), ok=j.get("ok"),
                              id=j.get("id_prediccion")))

        pg.on("response", on_response)
        pg.goto(url)
        pg.wait_for_function("document.getElementById('estado').textContent.includes('Enciende')", timeout=120000)
        t_cam[0] = time.time()
        pg.click("#btnCamara")
        pg.wait_for_timeout(int((duration + 3) * 1000))
        print(pg.inner_text("#metricas"))
        b.close()
    return preds


def score(meta: list[dict], preds: list[dict]) -> dict:
    """Asigna cada grabación al video con el que más se solapa y evalúa todos los modelos desplegados."""
    per_clip = {c["i"]: [] for c in meta}
    for p in preds:
        b = p["t"] - LAG_S
        a = b - p["n"] / 30
        overlap, i = max((min(b, c["end"]) - max(a, c["start"]), c["i"]) for c in meta)
        if overlap > 0:
            per_clip[i].append(p)
    cfg = json.loads((ROOT / "models" / "despliegue.json").read_text())
    models = {}
    for alias, info in cfg["modelos"].items():
        ck = torch.load(ROOT / "models" / f"{info['archivo']}.pt", map_location="cpu")
        m = construir(ck["arq"], len(ck["clases"]), **ck["kw"])
        m.load_state_dict(ck["state_dict"]); m.eval()
        models[alias] = (m, ck["clases"], info["archivo"])
    hits = {a: [0, 0] for a in models}
    for c in meta:
        ok = [p for p in per_clip[c["i"]] if p["ok"] and p["id"]]
        if not ok:
            continue
        p = max(ok, key=lambda q: q["n"])
        frames = [Frame(**f) for f in json.loads((ROOT / "logs" / "capturas" / f"{p['id']}.json").read_text())]
        x = torch.from_numpy(preparar(frames_a_array(frames)))[None]
        for a, (m, classes, _) in models.items():
            with torch.no_grad():
                order = np.argsort(-F.softmax(m(x), 1)[0].numpy())
            top = [classes[i] for i in order[:3]]
            hits[a][0] += top[0] == c["gloss"]
            hits[a][1] += c["gloss"] in top
    n = len(meta)
    return dict(
        videos=n, grabaciones=len(preds),
        senas_sin_grabacion=sum(1 for v in per_clip.values() if not v),
        senas_partidas=sum(1 for v in per_clip.values() if len(v) > 1),
        modelos={a: dict(archivo=models[a][2], top1=h[0] / n, top3=h[1] / n, top1_n=h[0], top3_n=h[1])
                 for a, h in hits.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=616)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--url", default="http://localhost:8000/?debug=1")
    ap.add_argument("--chromium", default="/usr/bin/chromium")
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()
    out = ROOT / "data" / "raw" / f"stream{args.n}"
    meta = build_stream(args.n, args.seed, out)
    print(f"{len(meta)} videos · {meta[-1]['end'] / 60:.1f} min de transmisión")
    preds = run_browser(out / "stream.mjpeg", meta[-1]["end"], args.url, args.chromium, args.headless)
    res = score(meta, preds)
    res["fecha"] = time.strftime("%Y-%m-%d")
    print(json.dumps(res, indent=2))
    (ROOT / "reports" / "tablas" / "prueba_camara_simulada.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
