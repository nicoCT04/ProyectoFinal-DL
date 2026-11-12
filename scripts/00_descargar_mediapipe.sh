#!/usr/bin/env bash
# Descarga los modelos de MediaPipe Tasks usados por la extracción y por la app.
set -euo pipefail
mkdir -p "$(dirname "$0")/../models/mediapipe" && cd "$(dirname "$0")/../models/mediapipe"
curl -sSLO https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
curl -sSLO https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
ls -la
