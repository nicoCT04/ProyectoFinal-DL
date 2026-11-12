"""
API + interfaz web del diccionario inverso de ASL.

  GET  /                    interfaz web (cámara, grabación automática, resultados)
  GET  /api/info            modelos disponibles, umbral de operación, supuestos de valor
  POST /api/predict         landmarks crudos de MediaPipe (desde el navegador) -> top-10
  POST /api/predict_video   video (respaldo) -> extracción en el servidor -> top-10
  POST /api/feedback        el usuario marca cuál era la seña (monitoreo + prueba de usabilidad)
  GET  /api/ejemplo/{glosa} esqueleto promedio de la seña (para mostrar cómo se hace)

Ejecutar:  .venv/bin/uvicorn app.server:app --port 8000   (desde la raíz del proyecto)
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from asl.features import N_PTS, T_FIJO, asignar_manos, POSE_IDX, preparar  # noqa: E402
from asl.models import construir  # noqa: E402
from asl.valor import SUPUESTOS_BASE, supuestos_dict  # noqa: E402

MODELS = ROOT / "models"
LOGS = ROOT / "logs"
LOGS.mkdir(exist_ok=True)
CONFIG = json.loads((MODELS / "despliegue.json").read_text())   # lo genera scripts/05_modelo_valor.py
VOC = json.loads((ROOT / "data" / "vocabulario.json").read_text())
ES = VOC["es"]
torch.set_num_threads(2)

app = FastAPI(title="Diccionario inverso de ASL")


def _cargar(nombre: str):
    ck = torch.load(MODELS / f"{nombre}.pt", map_location="cpu")
    m = construir(ck["arq"], len(ck["clases"]), **ck["kw"])
    m.load_state_dict(ck["state_dict"]); m.eval()
    return m, ck["clases"]


MODELOS = {}
for alias, info in CONFIG["modelos"].items():
    MODELOS[alias] = (*_cargar(info["archivo"]), info)
DEFAULT = CONFIG["default"]
PLANTILLAS = None
if (MODELS / "plantillas.npz").exists():
    PLANTILLAS = dict(np.load(MODELS / "plantillas.npz"))


# ----------------------------------------------------------------------------- esquemas
class Frame(BaseModel):
    pose: Optional[list[list[float]]] = None        # 33 x [x, y, z]
    hands: list[list[list[float]]] = []             # n_manos x 21 x [x, y, z]


class PeticionPredict(BaseModel):
    frames: list[Frame]
    modelo: Optional[str] = None
    sesion: Optional[str] = None
    modo: str = "grabacion"                          # grabacion | en_vivo


class Feedback(BaseModel):
    id_prediccion: str
    glosa_correcta: Optional[str] = None             # None = "ninguna era"
    rank_elegido: Optional[int] = None
    ms_hasta_elegir: Optional[float] = None
    sesion: Optional[str] = None


# ----------------------------------------------------------------------------- lógica
def frames_a_array(frames: list[Frame]) -> np.ndarray:
    out = np.full((len(frames), N_PTS, 3), np.nan, np.float32)
    for t, fr in enumerate(frames):
        if fr.pose is not None and len(fr.pose) >= 33:
            p = np.asarray(fr.pose, np.float32)
            out[t, : len(POSE_IDX)] = p[POSE_IDX]
        manos = [np.asarray(h, np.float32) for h in fr.hands if len(h) == 21]
        asignar_manos(out[t], manos)
    return out


def inferir(seq_cruda: np.ndarray, alias: str) -> dict:
    if alias not in MODELOS:
        raise HTTPException(400, f"Modelo desconocido: {alias}")
    model, clases, info = MODELOS[alias]
    hay_mano = ~np.isnan(seq_cruda[:, len(POSE_IDX):, 0]).all(1)
    if hay_mano.sum() < 5:
        return dict(ok=False, motivo="No se detectaron manos suficientes. Acércate a la cámara y "
                                     "asegúrate de que tus manos se vean.")
    t0 = time.perf_counter()
    x = preparar(seq_cruda)
    t1 = time.perf_counter()
    with torch.no_grad():
        logits = model(torch.from_numpy(x)[None])
        if info.get("tta_espejo"):
            from asl.data import reflejar
            logits = (logits + model(torch.from_numpy(reflejar(x))[None])) / 2
        p = F.softmax(logits, 1)[0].numpy()
    t2 = time.perf_counter()
    orden = np.argsort(-p)[:10]
    conf3 = float(p[orden[:3]].sum())
    umbral = info["umbral"]
    return dict(
        ok=True,
        id_prediccion=uuid.uuid4().hex[:12],
        modelo=alias,
        modelo_desc=info["desc"],
        confianza_top3=conf3,
        umbral=umbral,
        modo="seguro" if conf3 >= umbral else "no_seguro",
        candidatos=[dict(glosa=clases[i], es=ES.get(clases[i], ""), prob=float(p[i]), rank=r)
                    for r, i in enumerate(orden)],
        cuadros=int(len(seq_cruda)),
        cuadros_con_manos=int(hay_mano.sum()),
        ms_preproceso=(t1 - t0) * 1000,
        ms_modelo=(t2 - t1) * 1000,
    )


def log(evento: dict):
    evento["ts"] = time.time()
    with open(LOGS / "app_eventos.jsonl", "a") as f:
        f.write(json.dumps(evento, ensure_ascii=False) + "\n")


# ----------------------------------------------------------------------------- rutas
@app.get("/api/info")
def info():
    return dict(
        default=DEFAULT,
        modelos={a: {k: v for k, v in m[2].items() if k != "archivo"} for a, m in MODELOS.items()},
        vocabulario=[dict(glosa=g, es=ES.get(g, "")) for g in VOC["objetivo"]],
        supuestos=supuestos_dict(SUPUESTOS_BASE),
        t_fijo=T_FIJO,
    )


@app.post("/api/predict")
def predict(req: PeticionPredict):
    if not req.frames:
        raise HTTPException(400, "Sin cuadros")
    t0 = time.perf_counter()
    seq = frames_a_array(req.frames[-240:])
    res = inferir(seq, req.modelo or DEFAULT)
    res["ms_servidor"] = (time.perf_counter() - t0) * 1000
    if req.modo == "grabacion" and res.get("id_prediccion"):
        # Diagnóstico local: se guardan los puntos clave (nunca video) de cada grabación.
        cap = LOGS / "capturas"; cap.mkdir(exist_ok=True)
        (cap / f"{res['id_prediccion']}.json").write_text(json.dumps([f.model_dump() for f in req.frames]))
    if req.modo == "grabacion":
        log(dict(tipo="prediccion", sesion=req.sesion, **{k: res.get(k) for k in [
            "id_prediccion", "modelo", "confianza_top3", "modo", "cuadros", "ms_servidor"]},
            top3=[c["glosa"] for c in res.get("candidatos", [])[:3]], ok=res["ok"]))
    return res


@app.post("/api/predict_video")
async def predict_video(file: UploadFile = File(...), modelo: Optional[str] = None):
    from asl.extraccion import extraer  # import diferido: MediaPipe tarda en cargar
    t0 = time.perf_counter()
    with tempfile.NamedTemporaryFile(suffix=Path(file.filename or "v.mp4").suffix or ".mp4") as tmp:
        tmp.write(await file.read()); tmp.flush()
        seq = extraer(tmp.name)
    t1 = time.perf_counter()
    res = inferir(seq, modelo or DEFAULT)
    res["ms_extraccion"] = (t1 - t0) * 1000
    res["ms_servidor"] = (time.perf_counter() - t0) * 1000
    log(dict(tipo="prediccion_video", **{k: res.get(k) for k in ["id_prediccion", "modelo", "modo", "ms_servidor"]}))
    return res


@app.post("/api/feedback")
def feedback(fb: Feedback):
    log(dict(tipo="feedback", **fb.model_dump()))
    return dict(ok=True)


@app.get("/api/ejemplo/{glosa}")
def ejemplo(glosa: str):
    if PLANTILLAS is None or glosa not in PLANTILLAS:
        raise HTTPException(404, "Sin plantilla")
    return dict(glosa=glosa, frames=np.round(PLANTILLAS[glosa], 4).tolist())


@app.get("/dev/video/{nombre}")
def video_local(nombre: str, request: Request):
    """Solo para pruebas locales: sirve un video de ASL Citizen descargado (nunca se publica)."""
    if request.client is None or request.client.host not in ("127.0.0.1", "::1", "testclient"):
        raise HTTPException(403)
    ruta = (ROOT / "data" / "raw" / "ASL_Citizen" / "videos" / Path(nombre).name)
    if not ruta.exists():
        raise HTTPException(404)
    return FileResponse(ruta, media_type="video/mp4")


app.mount("/mediapipe", StaticFiles(directory=MODELS / "mediapipe"), name="mediapipe")
app.mount("/static", StaticFiles(directory=ROOT / "app" / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(ROOT / "app" / "static" / "index.html")
