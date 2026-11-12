"""
Paso 1 — Selección del vocabulario y descarga parcial de ASL Citizen.

El zip oficial pesa ~46 GB. El servidor de Microsoft acepta peticiones HTTP por rango,
así que con `remotezip` leemos el índice del zip y extraemos SOLO los videos que usamos:
  * 50 señas objetivo (vocabulario cotidiano de un estudiante de ASL).
  * 450 señas adicionales (aleatorias, semilla fija) para el preentrenamiento del
    modelo v2 (transfer learning).

Los videos quedan en data/raw/ (no se versionan: la licencia prohíbe redistribuirlos).
Uso:  python scripts/01_seleccionar_y_descargar.py [--solo-objetivo]
"""
import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from remotezip import RemoteZip
from tqdm import tqdm

URL = "https://download.microsoft.com/download/b/8/8/b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip"
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SPLITS = RAW / "ASL_Citizen" / "splits"
SEED = 23197
N_PRE = 450

# Vocabulario objetivo: palabras de uso diario para quien empieza a aprender ASL
# (saludos, familia, escuela, preguntas, emociones, necesidades básicas, tiempo).
OBJETIVO = [
    "HELLO", "THANKYOU", "PLEASE", "SORRY", "YES", "NO", "HELP", "WATER",
    "MOTHER", "FATHER", "FAMILY", "FRIEND", "BROTHER", "SISTER", "BOY", "GIRL",
    "SCHOOL", "TEACHER", "STUDENT", "LEARN", "BOOK", "WORK", "NAME",
    "WHERE", "WHO", "WHY", "WHEN",
    "GOOD", "BAD", "HAPPY", "SAD", "LOVE", "LIKE",
    "NEED", "MORE", "FINISH", "UNDERSTAND", "KNOW", "STOP",
    "BATHROOM", "SICK", "HOUSE", "PHONE", "APPLE", "COFFEE",
    "TODAY", "TOMORROW", "TIME", "HOT", "COLD",
]

# Traducción al español para la interfaz.
ES = {
    "HELLO": "hola", "THANKYOU": "gracias", "PLEASE": "por favor", "SORRY": "perdón / lo siento",
    "YES": "sí", "NO": "no", "HELP": "ayuda", "WATER": "agua", "MOTHER": "mamá", "FATHER": "papá",
    "FAMILY": "familia", "FRIEND": "amigo/a", "BROTHER": "hermano", "SISTER": "hermana",
    "BOY": "niño", "GIRL": "niña", "SCHOOL": "escuela", "TEACHER": "maestro/a",
    "STUDENT": "estudiante", "LEARN": "aprender", "BOOK": "libro", "WORK": "trabajo",
    "NAME": "nombre", "WHERE": "¿dónde?", "WHO": "¿quién?", "WHY": "¿por qué?",
    "WHEN": "¿cuándo?", "GOOD": "bueno", "BAD": "malo", "HAPPY": "feliz", "SAD": "triste",
    "LOVE": "amor / amar", "LIKE": "gustar", "NEED": "necesitar", "MORE": "más",
    "FINISH": "terminar / ya", "UNDERSTAND": "entender", "KNOW": "saber / conocer",
    "STOP": "alto / parar", "BATHROOM": "baño", "SICK": "enfermo", "HOUSE": "casa",
    "PHONE": "teléfono", "APPLE": "manzana", "COFFEE": "café", "TODAY": "hoy",
    "TOMORROW": "mañana", "TIME": "tiempo / hora", "HOT": "caliente", "COLD": "frío",
}


def cargar_splits():
    return pd.concat(
        [pd.read_csv(SPLITS / f"{s}.csv").assign(split=s) for s in ["train", "val", "test"]],
        ignore_index=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo-objetivo", action="store_true")
    ap.add_argument("--hilos", type=int, default=16)
    args = ap.parse_args()

    df = cargar_splits()
    glosas = set(df["Gloss"])
    faltan = [g for g in OBJETIVO if g not in glosas]
    assert not faltan, f"Glosas no encontradas: {faltan}"
    assert len(OBJETIVO) == 50 and len(set(OBJETIVO)) == 50

    rng = random.Random(SEED)
    resto = sorted(glosas - set(OBJETIVO))
    pre = sorted(rng.sample(resto, N_PRE))

    meta = {"objetivo": OBJETIVO, "preentreno": pre, "es": ES, "seed": SEED}
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "data" / "vocabulario.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    usar = set(OBJETIVO) if args.solo_objetivo else set(OBJETIVO) | set(pre)
    sub = df[df["Gloss"].isin(usar)].copy()
    sub["grupo"] = sub["Gloss"].map(lambda g: "objetivo" if g in OBJETIVO else "preentreno")
    sub.to_csv(ROOT / "data" / "subconjunto.csv", index=False)
    print(sub.groupby(["grupo", "split"]).size())

    destino = RAW / "ASL_Citizen" / "videos"
    destino.mkdir(parents=True, exist_ok=True)
    pendientes = [f for f in sub["Video file"] if not (destino / f).exists()]
    print(f"Videos a descargar: {len(pendientes)} (ya existen {len(sub) - len(pendientes)})")

    def bajar(lote):
        # Cada hilo abre su propia conexión al zip remoto.
        with RemoteZip(URL) as z:
            for f in lote:
                z.extract(f"ASL_Citizen/videos/{f}", RAW)
        return len(lote)

    lotes = [pendientes[i::args.hilos * 4] for i in range(args.hilos * 4)]
    with ThreadPoolExecutor(args.hilos) as ex, tqdm(total=len(pendientes)) as bar:
        for fut in as_completed([ex.submit(bajar, l) for l in lotes if l]):
            bar.update(fut.result())


if __name__ == "__main__":
    main()
