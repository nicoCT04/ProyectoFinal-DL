"""
Extracción de landmarks de un archivo de video con MediaPipe Tasks.

Se usa el modo VIDEO (con seguimiento entre cuadros), igual que la app en el navegador
(@mediapipe/tasks-vision con runningMode "VIDEO"). Esto importa: en una primera versión se usó
el modo IMAGE (cada cuadro independiente) y el navegador veía datos distintos a los de
entrenamiento — más manos detectadas y muñecas mucho más estables —, lo que bajaba el acierto
en la app. Ver "iteración de datos" en el informe.

Los detectores se crean de nuevo para cada video para que el seguimiento de un video no
contamine al siguiente.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .features import frame_desde_resultados

MP_DIR = Path(__file__).resolve().parents[1] / "models" / "mediapipe"
FPS = 30.0


def _crear(modo: str):
    os.environ.setdefault("GLOG_minloglevel", "2")
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    rm = vision.RunningMode.VIDEO if modo == "video" else vision.RunningMode.IMAGE
    hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MP_DIR / "hand_landmarker.task")),
        running_mode=rm, num_hands=2,
        min_hand_detection_confidence=0.4, min_hand_presence_confidence=0.4))
    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MP_DIR / "pose_landmarker_lite.task")),
        running_mode=rm, num_poses=1))
    return mp, hand, pose


def extraer(ruta: str | Path, max_lado: int = 640, modo: str = "video") -> np.ndarray:
    """Video -> (T, 53, 3) con NaN en puntos no detectados. modo: 'video' (por defecto) o 'image'."""
    import cv2

    mp, hand, pose = _crear(modo)
    cap = cv2.VideoCapture(str(ruta))
    frames, t = [], 0.0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            h, w = bgr.shape[:2]
            if max(h, w) > max_lado:
                esc = max_lado / max(h, w)
                bgr = cv2.resize(bgr, (int(w * esc), int(h * esc)))
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            if modo == "video":
                ts = int(round(t))
                rp, rh = pose.detect_for_video(img, ts), hand.detect_for_video(img, ts)
                t += 1000.0 / FPS
            else:
                rp, rh = pose.detect(img), hand.detect(img)
            frames.append(frame_desde_resultados(rp.pose_landmarks[0] if rp.pose_landmarks else None,
                                                 rh.hand_landmarks))
    finally:
        cap.release(); hand.close(); pose.close()
    if not frames:
        return np.full((1, 53, 3), np.nan, np.float32)
    return np.stack(frames)
