# Prueba de punta a punta de la app (reproducible)

Valida que lo medido offline se sostiene en el producto: los videos pasan por **el mismo flujo del
navegador** que la cámara (MediaPipe Tasks JS en modo VIDEO + segmentación automática + `/api/predict`).

1. Generar la lista de videos (60 de prueba, personas no vistas):

```bash
.venv/bin/python -c "
import pandas as pd, json, urllib.parse
s=pd.read_csv('data/subconjunto.csv'); t=s[(s.grupo=='objetivo')&(s.split=='test')].sample(60,random_state=11)
json.dump([[urllib.parse.quote(f), g] for f,g in zip(t['Video file'],t.Gloss)], open('app/static/_prueba_e2e.json','w'))"
```

2. Levantar la app (`uvicorn app.server:app --port 8000`), abrir http://localhost:8000 y pegar en la
   consola del navegador (la ruta `/dev/video/...` solo sirve videos locales a `localhost`):

```js
window.__res = []; (async () => {
  const L = await (await fetch("/static/_prueba_e2e.json")).json();
  document.getElementById("modelo").value = "v2";          // o v0 / v1
  for (const [f, g] of L) {
    document.querySelectorAll(".tarjeta").forEach(e => e.remove());
    await reproducirVideo("/dev/video/" + f);
    await new Promise(r => setTimeout(r, 300));
    window.__res.push([g, [...document.querySelectorAll(".tarjeta .glosa")].map(e => e.textContent)]);
  }
  const R = window.__res;
  console.log("top1", R.filter(r => r[1][0] === r[0]).length, "top3", R.filter(r => r[1].slice(0,3).includes(r[0])).length, "de", R.length);
})();
```

3. Borrar `app/static/_prueba_e2e.json` al terminar (no publicarlo).

Resultados (7-nov-2026): `reports/tablas/prueba_app_e2e.json` — v2 57/60 top-1 y 59/60 top-3; v0 44/60 y 54/60.
