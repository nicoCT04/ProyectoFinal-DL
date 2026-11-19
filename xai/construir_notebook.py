"""Genera xai/lab10.ipynb (luego se ejecuta con nbconvert para dejar las salidas)."""
from pathlib import Path

import nbformat as nbf

AQUI = Path(__file__).resolve().parent
nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))  # noqa: E731
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))    # noqa: E731

md("""
# Laboratorio #10 — Explainable AI sobre SeñaBusca

**CC3092 · Deep Learning y Sistemas Inteligentes** — Universidad del Valle de Guatemala
**Esteban Cárcamo (23016) · Nicolás Concuá (23197)** — noviembre de 2026

Enlace al repositorio: https://github.com/nicoCT04/ProyectoFinal-DL

Explicamos el modelo **v2** del proyecto final (Transformer con transfer learning, `it7_tl_tf_webcam_s1`,
el que usa la app según `models/despliegue.json`) **sin reentrenarlo**. Este notebook regenera todas las
figuras y tablas del informe con semilla fija; el trabajo pesado está en `xai/generar_xai.py` y las
funciones en `xai/xai_asl.py`.
""")

md("""
## 1. Investigación: métodos de XAI para nuestro tipo de dato

Cada seña es una **serie de tiempo multivariada**: 64 cuadros × 161 variables (53 puntos de MediaPipe en
x, y, z normalizados a los hombros + 2 banderas de presencia de mano).

| Método | Idea | Requiere | Costo | Limitaciones |
|---|---|---|---|---|
| Saliencia / SmoothGrad [1] | gradiente de la clase respecto a la entrada (promediado con ruido) | gradientes | 1 o n pasadas | ruidoso, saturación |
| **Integrated Gradients** [2] | integra el gradiente de una línea base a la entrada | gradientes + línea base | n pasos (64) | depende de la línea base |
| SHAP (Kernel, Gradient, Deep) [3] | valores de Shapley sobre variables o grupos | caja negra o gradientes + fondo | Kernel: miles de evaluaciones | caro, depende del fondo |
| LIME / Anchors [4] | modelo lineal (o regla) local en un vecindario perturbado | caja negra | cientos-miles de evaluaciones | inestable, sensible al kernel |
| Oclusión por ventanas [5] | reemplazar bloques parte × tiempo y medir la caída | caja negra | 1 pasada por bloque | fuera de distribución, sin interacciones |
| TimeSHAP, Dynamask, WinIT [6–8] | SHAP por eventos, máscaras aprendidas, ventanas | variable | de medio a alto | pensados para otros modelos o caros |
| Atención [CLS] | pesos de atención del Transformer | nada | gratis | no es una explicación fiel [9] |
| Grad-CAM | mapas de activación convolucionales | capas conv | bajo | **no aplica** (no hay convoluciones) |

* **Interpretable por diseño vs. a posteriori:** la regresión logística del proyecto es interpretable por
  sus coeficientes; IG explica a posteriori un modelo opaco. Usamos la logística como baseline de comparación.
* **Propiedades formales:** IG cumple *completitud* (Σ atribuciones = f(x) − f(x′)), sensibilidad e
  invariancia a la implementación; SHAP, aditividad/eficiencia; LIME, solo fidelidad local.
* **Parámetros que afectan la explicación:** línea base y pasos de IG, objetivo (logit o probabilidad),
  fondo de SHAP, kernel de LIME, tamaño y reemplazo de los bloques de oclusión, y la agregación.
""")

md("""
## 2. Selección del método

| Criterio | Integrated Gradients (**elegido**) | Oclusión por bloques (alternativa) |
|---|---|---|
| Compatibilidad | modelo diferenciable en PyTorch, captum directo | cualquier modelo |
| Tipo de explicación | atribución con signo por punto y cuadro | caída de probabilidad por 32 bloques |
| Costo | ~100 ms por seña (CPU, 1 hilo) | ~30 ms por seña |
| Comunicación | mano y momento sobre el esqueleto | fácil, pero de baja resolución |
| Limitaciones | depende de la línea base | reemplazo fuera de distribución |

KernelSHAP por bloques se descartó por costo (≈ 2 000 evaluaciones por seña). La oclusión se calcula
de todas formas para **comparar la fidelidad** de ambos métodos (sección 3.4).
""")

md("## 3. Implementación\n\n### 3.0 Entorno y semilla")
code("""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from IPython.display import Image, display

# Rutas robustas: funciona desde la raíz del repo o desde xai/
ROOT = Path.cwd() if (Path.cwd() / "asl").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT / "xai")); sys.path.insert(0, str(ROOT))
import xai_asl as X
import generar_xai as G
from asl.data import cargar_grupo

SEED = X.SEED                      # 23197
G.fijar_semillas()                 # numpy + torch, CPU con 1 hilo, algoritmos deterministas
REGENERAR = True                   # False = solo mostrar lo que ya está en reports/xai/
OUT = X.OUT
print("semilla:", SEED, "· salida:", OUT.relative_to(ROOT))
""")

md("""
### 3.1 Modelo y datos

Se carga el mismo archivo y umbral que usa la app. No se reentrena nada.
""")
code("""
m, clases, info = X.cargar_modelo("v2")
data, _ = cargar_grupo("objetivo")
Xte, yte, pte, _ = data["test"]
P = X.probabilidades(m, Xte); pred = P.argmax(1)
print(info)
print(f"prueba: {len(yte)} videos de {len(set(pte))} personas · top-1 = {(pred == yte).mean():.3f}")
""")

md("""
### 3.2 Integrated Gradients en una seña: la línea base y la completitud

La línea base **«reposo»** es la misma persona con la pose promediada en el tiempo y **sin manos
detectadas** (coordenadas 0 y bandera 0, igual que codifica el preprocesamiento una mano ausente).
Comprobamos la completitud: la suma de atribuciones debe ser el logit de la seña menos el de la línea base.
""")
code("""
i = 0
x = Xte[i:i + 1]; t = pred[i:i + 1]
b = X.linea_base(x, "reposo")
A, delta = X.integrated_gradients(m, x, t, b, n_steps=64)
with torch.no_grad():
    lx = m(torch.from_numpy(x))[0, t[0]].item(); lb = m(torch.from_numpy(b))[0, t[0]].item()
print(f"seña: {clases[yte[i]]} · predicción: {clases[t[0]]}")
print(f"Σ atribuciones = {A.sum():.4f}   logit(x) − logit(base) = {lx - lb:.4f}   error = {abs(delta[0]):.1e}")
print("cuota por parte:", dict(zip(X.GRUPOS_IG, np.round(np.abs(X.agregar(A)).sum(2)[0] / np.abs(X.agregar(A)).sum(), 3))))
""")

md("""
### 3.3 Corrida completa (todas las figuras y tablas del informe)

`G.main()` calcula IG con tres líneas base sobre los 616 videos de prueba, la oclusión, las curvas de
borrado, la prueba de cordura, el baseline interpretable (logística), los casos locales y el análisis de
señas fuera del vocabulario. Tarda ≈ 6 min en CPU.
""")
code("""
if REGENERAR:
    t0 = time.time(); G.main(); print(f"listo en {time.time() - t0:.0f} s")
R = json.loads((OUT / "resumen.json").read_text())
pd.read_csv(OUT / "tabla_parametros.csv")
""")

md("### 3.4 Elección de la línea base y fidelidad (IG vs. oclusión)")
code("""
display(pd.read_csv(OUT / "tabla_lineas_base.csv"))
display(pd.read_csv(OUT / "tabla_metodos.csv"))
print("ρ Spearman IG vs oclusión (mediana):", round(R["spearman_ig_oclusion_mediana"], 2),
      "· IG vs modelo con pesos al azar:", round(R["spearman_cordura_mediana"], 2))
display(Image(OUT / "fig_fidelidad.png"))
""")
md("""
La curva de borrado reemplaza primero los bloques que cada método marca como más importantes: si la
probabilidad cae rápido, la explicación es fiel. **IG con línea base de reposo** tiene la menor área
(la mitad que la oclusión). Con pesos aleatorios, el ranking de IG cambia por completo (ρ ≈ 0): la
explicación depende de lo aprendido y pasa la prueba de cordura.
""")

md("### 3.5 Explicaciones locales (6 casos de prueba)")
code("""
display(pd.read_csv(OUT / "tabla_casos_locales.csv"))
display(Image(OUT / "fig_casos_locales.png"))
""")

md("### 3.6 Explicación global y baseline interpretable")
code("""
display(Image(OUT / "fig_global.png"))
pd.DataFrame({"IG v2": R["cuota_global"], "logística": R["cuota_logistica"]}).round(3)
""")

md("""
## 4. Interpretación en el contexto del proyecto
""")
code("""
pd.read_csv(OUT / "tabla_atajos_subgrupos.csv")
""")
md("""
* **Señales con sentido:** la mano dominante concentra ~53 % de la atribución, la no dominante ~18 %, los
  brazos ~10 % y la cara ~3 %, en el tramo central de la seña. Coincide con la fonología de ASL (forma,
  ubicación y movimiento de la mano).
* **Atajos buscados:** identidad por la cara (3 %, no), lateralidad (41.6 % de señas con mano izquierda
  dominante; top-1 0.898 vs 0.911 y el modelo usa la mano dominante en ambos grupos), bordes del recorte
  (11 % en errores vs 8 % en aciertos). **Hallazgo:** las banderas de presencia de mano reciben ~17 %. Es un
  proxy de la calidad del detector: si MediaPipe pierde la mano, la predicción cambia.
* **Errores:** HOT → BAD y GOOD → THANKYOU comparten ubicación (boca/mentón) y dirección. Las distingue la
  forma de la mano o el contacto con la mano no dominante, que el modelo casi no usa. SSHH (fuera del
  vocabulario) → WHO por el índice junto a los labios.
* **Decisiones:** aumento que apague la detección de una mano por tramos, variable de distancia entre
  manos, regla de pares confundibles en la app y una clase «otra seña».
* **Para el usuario:** botón «¿por qué?» con el esqueleto y la parte decisiva resaltada, más una frase.
""")
code("""
print(f"señas fuera del vocabulario aceptadas sobre el umbral: {R['oov_aceptadas']:.1%} de {R['n_oov']}")
print(f"errores con confianza sobre el umbral: {R['errores_con_conf_sobre_umbral']:.1%}")
print("pares más confundidos:", R["pares_confusion"])
""")

md("""
## 5. Discusión

* **IG vs. oclusión:** mantendríamos IG (más fiel, completitud y resolución de punto y cuadro); la línea
  base es la decisión que más cambia la fidelidad (con la salvedad de que la curva borra con la base de
  reposo, lo que favorece a IG-reposo frente a «ceros» y «media»).
* **Lo que no mostraban las métricas:** la dependencia de la presencia de mano y el poco uso de la mano no
  dominante en pares que dependen de ella.
* **Costo:** ~100 ms por seña contra ~1 ms de la inferencia: viable a demanda o por lote para auditoría.
* **Riesgos:** falsa confianza en explicaciones de errores con confianza alta; IG no explica los sesgos de
  MediaPipe; no usar para interpretación médica o legal, ni para calificar a estudiantes.

### Referencias
[1] Smilkov et al., 2017 · [2] Sundararajan et al., 2017 · [3] Lundberg y Lee, 2017 · [4] Ribeiro et al., 2016 ·
[5] Zeiler y Fergus, 2014 · [6] Bento et al., 2021 · [7] Crabbé y van der Schaar, 2021 · [8] Leung et al., 2023 ·
[9] Jain y Wallace, 2019 · [10] Samek et al., 2017 · [11] Adebayo et al., 2018 · [12] Kokhlikyan et al., 2020
""")

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, AQUI / "lab10.ipynb")
print("escrito", AQUI / "lab10.ipynb")
