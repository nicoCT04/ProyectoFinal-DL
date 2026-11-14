"""
Descarga solo los videos de PRUEBA de las 50 señas objetivo (616 videos, ~350 MB) a data/raw/.

Los usan la sesión de prueba de la app (/?prueba=usuarios muestra la seña hecha por otra persona)
y docs/PRUEBA_E2E_APP.md. Requiere data/subconjunto.csv (script 01). No se versionan (licencia).

Uso:  python scripts/11_descargar_videos_prueba.py [--split test] [--hilos 12]
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from remotezip import RemoteZip

URL = "https://download.microsoft.com/download/b/8/8/b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip"
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--hilos", type=int, default=12)
    args = ap.parse_args()
    sub = pd.read_csv(ROOT / "data" / "subconjunto.csv")
    files = sub[(sub["grupo"] == "objetivo") & (sub["split"] == args.split)]["Video file"].tolist()
    dest = RAW / "ASL_Citizen" / "videos"
    dest.mkdir(parents=True, exist_ok=True)
    pending = [f for f in files if not (dest / f).exists()]
    print(f"Videos: {len(files)} · faltan {len(pending)}")

    def fetch(batch):
        with RemoteZip(URL) as z:
            for f in batch:
                z.extract(f"ASL_Citizen/videos/{f}", RAW)
        return len(batch)

    batches = [b for b in (pending[i::args.hilos * 3] for i in range(args.hilos * 3)) if b]
    done = 0
    with ThreadPoolExecutor(args.hilos) as ex:
        for n in ex.map(fetch, batches):
            done += n
            print(f"  {done}/{len(pending)}", flush=True)


if __name__ == "__main__":
    main()
