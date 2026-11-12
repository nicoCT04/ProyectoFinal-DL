"""
Paso 10 — Re-mide la latencia de inferencia en CPU de todas las corridas con la máquina en reposo.

Durante el entrenamiento la latencia se midió con la CPU ocupada por la extracción de MediaPipe y
otros entrenamientos, lo que la distorsiona. Aquí se mide de forma secuencial (batch 1, 2 hilos,
igual que el servidor) y se actualiza 'latencia_cpu_ms' en reports/runs/*.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from asl.features import N_FEAT, T_FIJO  # noqa: E402
from asl.models import construir  # noqa: E402

torch.set_num_threads(2)
x = torch.randn(1, T_FIJO, N_FEAT)
for js in sorted((ROOT / "reports" / "runs").glob("*.json")):
    r = json.loads(js.read_text())
    pt = ROOT / "models" / f"{r['iteracion']}_s{r['seed']}.pt"
    if not pt.exists():
        continue
    ck = torch.load(pt, map_location="cpu")
    m = construir(ck["arq"], len(ck["clases"]), **ck["kw"]); m.load_state_dict(ck["state_dict"]); m.eval()
    with torch.no_grad():
        for _ in range(30):
            m(x)
        ts = []
        for _ in range(300):
            t0 = time.perf_counter(); m(x); ts.append((time.perf_counter() - t0) * 1000)
    r["latencia_cpu_ms"] = float(np.median(ts)); r["latencia_cpu_p95_ms"] = float(np.percentile(ts, 95))
    js.write_text(json.dumps(r, indent=2))
    print(f"{r['iteracion']} s{r['seed']}: p50 {r['latencia_cpu_ms']:.2f} ms  p95 {r['latencia_cpu_p95_ms']:.2f} ms")
