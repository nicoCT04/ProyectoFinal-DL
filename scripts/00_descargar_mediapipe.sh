#!/usr/bin/env bash
# Descarga los modelos de MediaPipe Tasks usados por la extracción y por la app.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/models/mediapipe" && cd "$ROOT/models/mediapipe"
curl -sSLO https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
curl -sSLO https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
ls -la

# Copia local de MediaPipe Tasks para el navegador: la app funciona sin internet (con respaldo al CDN).
CDN=https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14
VENDOR="$ROOT/app/static/vendor/tasks-vision"
mkdir -p "$VENDOR/wasm"
curl -sSL "$CDN/vision_bundle.mjs" -o "$VENDOR/vision_bundle.mjs"
for f in vision_wasm_internal.js vision_wasm_internal.wasm vision_wasm_nosimd_internal.js vision_wasm_nosimd_internal.wasm; do
  curl -sSL "$CDN/wasm/$f" -o "$VENDOR/wasm/$f"
done
ls -la "$VENDOR" "$VENDOR/wasm"
