/*
 * Genera reports/Informe_ProyectoFinal_ASL.docx (estilo de los labs: portada UVG, Arial 11/12, Carta).
 * Los números se leen de reports/tablas/ (generadas por los scripts 05 y 06), así que el informe
 * se puede regenerar si cambian los resultados.
 *
 *   cd scripts/informe && npm install docx && node build_informe.js
 */
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, ImageRun, PageBreak,
  ExternalHyperlink, LevelFormat, Table, TableRow, TableCell, WidthType, BorderStyle, ShadingType,
} = require("docx");

const ROOT = path.resolve(__dirname, "..", "..");
const TAB = path.join(ROOT, "reports", "tablas");
const FIG = path.join(ROOT, "reports", "figuras");
const OUT = path.join(ROOT, "reports", "Informe_ProyectoFinal_ASL.docx");
const LOGO = path.join(__dirname, "uvg_logo.png");
const REPO_URL = "https://github.com/nicoCT04/ProyectoFinal-DL";
const VIDEO_URL = "https://youtu.be/HbqyY3AJdvc";
const FECHA = "Guatemala, noviembre de 2026";

// ----------------------------------------------------------------------------- datos
function csv(nombre) {
  const txt = fs.readFileSync(path.join(TAB, nombre), "utf8").trim();
  const filas = [];
  for (const linea of txt.split(/\r?\n/)) {
    const out = []; let cur = "", q = false;
    for (let i = 0; i < linea.length; i++) {
      const c = linea[i];
      if (q) { if (c === '"' && linea[i + 1] === '"') { cur += '"'; i++; } else if (c === '"') q = false; else cur += c; }
      else if (c === '"') q = true; else if (c === ",") { out.push(cur); cur = ""; } else cur += c;
    }
    out.push(cur); filas.push(out);
  }
  const [cab, ...resto] = filas;
  return resto.map(f => Object.fromEntries(cab.map((k, i) => [k, f[i]])));
}
const J = (n) => JSON.parse(fs.readFileSync(path.join(TAB, n), "utf8"));
const N = J("numeros_informe.json");          // generado por scripts/08_numeros_informe.py
const iter = csv("tabla_iteraciones.csv");
const vers = csv("tabla_versiones.csv");
const fmt = (x, d = 0) => Number(x).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
const pct = (x, d = 1) => (100 * Number(x)).toFixed(d) + " %";

// ----------------------------------------------------------------------------- estilo
const FONT = "Arial", BODY = 22, TITLE = 24;
function runs(parts) {
  if (typeof parts === "string") parts = [{ t: parts }];
  return parts.map(p => new TextRun({ text: p.t, bold: !!p.b, italics: !!p.i, font: FONT, size: p.size || BODY, color: "000000" }));
}
function P(parts, o = {}) {
  return new Paragraph({ alignment: o.align || AlignmentType.JUSTIFIED, spacing: { after: o.after == null ? 100 : o.after, line: 264 }, children: runs(parts) });
}
function H(text, level, before = 200, after = 80) {
  return new Paragraph({ heading: level, spacing: { before, after }, keepNext: true,
    children: [new TextRun({ text, bold: true, font: FONT, size: TITLE, color: "000000" })] });
}
const H1 = t => H(t, HeadingLevel.HEADING_1);
const H2 = t => H(t, HeadingLevel.HEADING_2, 140, 60);
function LI(parts) {
  return new Paragraph({ numbering: { reference: "bullets", level: 0 }, alignment: AlignmentType.JUSTIFIED,
    spacing: { after: 50, line: 264 }, children: runs(parts) });
}
function C(text, { b = false, size = BODY, after = 40 } = {}) {
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after }, children: [new TextRun({ text, bold: b, font: FONT, size, color: "000000" })] });
}
function pngSize(buf) { return { w: buf.readUInt32BE(16), h: buf.readUInt32BE(20) }; }
let nFig = 0, nTab = 0;
function IMG(file, ancho, pie) {
  const buf = fs.readFileSync(path.join(FIG, file));
  const { w, h } = pngSize(buf);
  nFig += 1;
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 80, after: 20 }, keepNext: true,
      children: [new ImageRun({ type: "png", data: buf, transformation: { width: ancho, height: Math.round(ancho * h / w) } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 140 },
      children: [new TextRun({ text: `Figura ${nFig}. ${pie}`, italics: true, font: FONT, size: 18, color: "000000" })] }),
  ];
}
const borde = { style: BorderStyle.SINGLE, size: 4, color: "999999" };
function TABLA(cab, filas, anchos, pie, fs_ = 16) {
  nTab += 1;
  const total = 10080;
  const w = anchos.map(a => Math.round(a / anchos.reduce((x, y) => x + y, 0) * total));
  const celda = (t, i, head) => new TableCell({
    width: { size: w[i], type: WidthType.DXA },
    borders: { top: borde, bottom: borde, left: borde, right: borde },
    shading: head ? { fill: "E8E8E8", type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 40, bottom: 40, left: 70, right: 70 },
    children: [new Paragraph({ alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
      children: [new TextRun({ text: String(t), bold: head, font: FONT, size: fs_, color: "000000" })] })],
  });
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 80, after: 40 }, keepNext: true,
      children: [new TextRun({ text: `Tabla ${nTab}. ${pie}`, italics: true, font: FONT, size: 18, color: "000000" })] }),
    new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: w,
      rows: [new TableRow({ tableHeader: true, children: cab.map((t, i) => celda(t, i, true)) }),
             ...filas.map(f => new TableRow({ children: f.map((t, i) => celda(t, i, false)) }))] }),
    new Paragraph({ spacing: { after: 120 }, children: [] }),
  ];
}

// ----------------------------------------------------------------------------- portada
const cover = [
  new Paragraph({ spacing: { before: 1200 }, children: [] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 },
    children: [new ImageRun({ type: "png", data: fs.readFileSync(LOGO), transformation: { width: 100, height: 150 } })] }),
  C("Universidad del Valle de Guatemala", { b: true }),
  C("Facultad de Ingeniería"),
  C("CC3092 — Deep Learning y Sistemas Inteligentes", { after: 260 }),
  C("Proyecto Final", { b: true, size: 40, after: 120 }),
  C("SeñaBusca: diccionario inverso de lengua de señas americana con cámara", { b: true, size: 26, after: 320 }),
  C("Esteban Cárcamo — 23016", { b: true, after: 40 }),
  C("Nicolás Concuá — 23197", { b: true, after: 60 }),
  C(FECHA),
  new Paragraph({ children: [new PageBreak()] }),
];
const enlace = (etiqueta, url, after) => new Paragraph({ spacing: { after }, children: [
  new TextRun({ text: etiqueta, bold: true, font: FONT, size: BODY }),
  new ExternalHyperlink({ link: url, children: [new TextRun({ text: url, font: FONT, size: BODY, color: "0563C1", underline: {} })] }),
] });
const repo = [enlace("Enlace al repositorio: ", REPO_URL, 40), enlace("Video de demostración: ", VIDEO_URL, 160)];

// ----------------------------------------------------------------------------- contenido
const S = N.supuestos;
const v = Object.fromEntries(vers.map(r => [r["Versión"], r]));
const it = Object.fromEntries(iter.map(r => [r["Iteración"], r]));
const dep = N.desplegado;
const opt = (n) => (fs.existsSync(path.join(TAB, n)) ? J(n) : null);
const RV = J("resumen_valor.json");
const PU = opt("prueba_usuarios.json"), VP = opt("validacion_propia.json"), CS = opt("prueba_camara_simulada.json");
const EXP = J("explicabilidad_oclusion.json")[dep];
const PP = csv("por_participante.csv");
const IDAT = J("iteracion_datos_resumen.json")["it2_tf_aug"];
const LC = Object.fromEntries(csv("curva_aprendizaje.csv").map(r => [`${r.familia}_${r.frac}`, r]));
const RB = Object.fromEntries((fs.existsSync(path.join(TAB, "robustez_webcam.csv")) ? csv("robustez_webcam.csv") : []).map(r => [r["perturbación"], r]));
const ic = (k) => RV.ic95_diferencias_valor[k].map(x => fmt(x)).join(", ");
const top = (r, k) => String(r[k]).split(" ")[0];
const usd = (r) => fmt(Number(String(r["Valor neto USD/mes"]).replace(/,/g, "")));
const V2 = v[dep], V0 = v.v0, V1 = v.v1;
const dV = (a, b) => fmt(Number(a["Valor neto USD/mes"]) - Number(b["Valor neto USD/mes"]));

const content = [
  ...repo,
  H1("Resumen"),
  P(`SeñaBusca es un diccionario inverso de lengua de señas americana (ASL): la persona hace una seña frente a la cámara y recibe las tres coincidencias más probables con su significado en español. El núcleo es un modelo de deep learning que diseñamos y entrenamos nosotros sobre secuencias de puntos clave del cuerpo y las manos; no usamos ningún LLM. La versión final (v2, un Transformer preentrenado con 450 señas y ajustado a 50) acierta el ${pct(V2["Top-1 test"])} en la primera opción y el ${pct(V2["Top-3 test"])} dentro de las tres mostradas, con personas que nunca vio. Con nuestro modelo de valor, un programa de 500 estudiantes ahorraría ${fmt(V2["Horas ahorradas/mes"])} horas al mes (≈ ${usd(V2)} USD netos), ${dV(V2, V0)} USD/mes más que la primera versión (LSTM).`),

  H1("1. Contexto y problema"),
  P(`Quien aprende ASL ve a menudo señas que no conoce. Los diccionarios en línea (Handspeak, Lifeprint, SignASL) van de palabra a seña, no al revés: para buscar una seña hay que navegar por forma de la mano y ubicación, o preguntarle al docente. En nuestra prueba piloto, buscar a mano tomó una mediana de ${fmt(PU ? PU.manual.t_mediana_s : S.t_manual)} s por seña y no encontramos ninguna de tres. El usuario es un estudiante de un curso o programa de ASL; quien paga es la institución, que valora el tiempo de sus estudiantes.`),
  P(`¿Por qué deep learning? La misma seña cambia entre personas, cámaras y velocidades, y lo que la define es una trayectoria en el tiempo. Probamos primero una alternativa sin deep learning (regresión logística sobre estadísticos de los puntos): llega a ${N.clasico_top3} de top-3. El mejor modelo baja el error top-3 de ${N.clasico_err3} a ${N.dep_err3}, y eso vale ${N.delta_vs_clasico} USD/mes adicionales.`),
  P(`KPIs: segundos por búsqueda (línea base ${fmt(N.costo_manual_seg)} s equivalentes a mano, contando las búsquedas fallidas), porcentaje de búsquedas resueltas (${pct(S.p_manual, 0)} a mano) y tasa de "engaños" (quedarse con un significado equivocado).`),

  H1("2. Datos"),
  P(`Usamos ASL Citizen [1] (Microsoft Research, 2023): 83 399 videos de webcam de 2 731 señas grabados por 52 personas sordas o con discapacidad auditiva. Tomamos 50 señas cotidianas (${N.eda.obj_videos} videos, ≈ 30 por seña, sin desbalance) y otras 450 señas al azar (${N.eda.pre_videos} videos) solo para preentrenar. Respetamos la división oficial por persona (${N.eda.obj_train} / ${N.eda.obj_val} / ${N.eda.obj_test} videos), así que el modelo se evalúa con personas que nunca vio, como en el uso real.`),
  P(`Preprocesamiento. Con MediaPipe [4] extraemos por cuadro 53 puntos (11 del torso y la cara, 21 por mano). Recortamos el reposo antes y después de la seña, centramos en los hombros y escalamos por su ancho (invariancia a posición y distancia) y remuestreamos a 64 cuadros: cada seña es un tensor 64 × 161 (53 × 3 coordenadas más dos indicadores de presencia de mano). El mismo código lo usan el entrenamiento y la app. En el EDA vimos que la mano se detecta solo en el ${pct(N.eda.frac_manos, 0)} de los cuadros y que varía mucho por persona (${pct(N.eda.manos_min_part, 0)}–${pct(N.eda.manos_max_part, 0)}), una fuente de sesgo que retomamos en la sección 7.`),
  P(`Licencia y privacidad. ASL Citizen es de uso de investigación no comercial: no publicamos videos ni puntos derivados. En la app el video no sale del navegador; solo viajan los puntos clave.`),
  ...IMG("preprocesamiento.png", 520, "De los puntos de MediaPipe al tensor de entrada: recorte, normalización por hombros y remuestreo a 64 cuadros."),

  H1("3. Modelo de valor"),
  P(`Por búsqueda comparamos el costo en segundos con y sin la app. Sin la app: t_manual + (1 − p_manual) · c_no_resuelta. Con la app: grabar la seña (t_grabar) y, si la confianza del top-3 supera el umbral τ, revisar tres opciones (t_rev3); si no, la app avisa "no estoy seguro" y muestra diez (t_rev10). Si la correcta no aparece, el usuario puede quedarse con una equivocada (p_engano, costo c_error) o terminar buscando a mano. El valor neto mensual es el ahorro × búsquedas/mes × valor de la hora − costo de operación. Las señas que el modelo no conoce (15 % de las búsquedas) no las suponemos: las medimos con videos de las 450 señas extra.`),
  ...TABLA(["Supuesto", "Valor", "Fuente"], [
    ["t_manual", `${fmt(S.t_manual)} s`, "Medido: mediana de la búsqueda en diccionarios (prueba piloto)"],
    ["p_manual", pct(S.p_manual, 0), "Estimación conservadora [2], [3] (en la prueba piloto: 0 de 3)"],
    ["t_grabar / t_rev3", `${fmt(S.t_grabar, 1)} s / ${fmt(S.t_rev3, 1)} s`, "Medidos en la app (prueba piloto)"],
    ["t_rev10", `${fmt(S.t_rev10)} s`, "Estimación: 10 opciones a ≈ 4–5 s c/u"],
    ["p_engano / c_error", `${pct(S.p_engano, 0)} / ${fmt(S.c_error)} s`, "Estimación: confiar en una sugerencia equivocada y corregirlo después"],
    ["c_no_resuelta", `${fmt(S.c_no_resuelta)} s`, "Estimación: preguntar al docente o retrasar la tarea"],
    ["p_vocab", pct(S.p_vocab, 0), "Búsquedas dentro del vocabulario (el efecto del resto se mide)"],
    ["Escala y dinero", `${fmt(S.busquedas_mes)} búsq./mes · ${fmt(S.valor_hora)} USD/h`, "500 estudiantes × 20 búsquedas; costo de oportunidad de un estudiante"],
    ["Operación", `${fmt(S.costo_fijo_mes)} USD/mes`, "Un servidor CPU pequeño"],
  ], [2, 2.2, 5.8], "Supuestos del modelo de valor (asl/valor.py)."),
  P(`Punto de operación. Elegimos τ maximizando el valor en validación y reportamos el valor en prueba. Medir los supuestos cambió la decisión: revisar tres opciones toma ${fmt(S.t_rev3, 1)} s (estimábamos 15), así que conviene mostrarlas más seguido y el umbral óptimo de v2 bajó de 0.83 a ${V2["Umbral"]}. A cambio, la app marca "no estoy seguro" en el ${pct(V2["OOV detectadas"], 0)} de las señas fuera del vocabulario.`),
  ...IMG("valor_vs_umbral.png", 430, "Valor neto según el umbral de confianza; el punto es el umbral elegido en validación."),
  P(`Sensibilidad. Variamos cada supuesto ±20 % y ±50 %, re-optimizando el umbral. Ninguno por sí solo vuelve negativo el valor; los que más pesan son el valor de la hora, el volumen de búsquedas y la cobertura del vocabulario. Aun con la búsqueda manual a la mitad de lo medido, v2 genera valor, y la ventaja de v2 sobre v0 se mantiene positiva en todos los escenarios.`),
  ...IMG("sensibilidad_tornado.png", 430, "Sensibilidad del valor neto de v2 a cambios de ±20 % y ±50 % en cada supuesto."),

  H1("4. Metodología"),
  P("Todos los modelos reciben el mismo tensor 64 × 161 y terminan en una capa lineal con softmax sobre las 50 señas. Probamos tres familias vistas en clase:"),
  LI([{ t: "LSTM bidireccional ", b: true }, { t: "(2 capas, 192 unidades, promedio temporal): procesa la seña cuadro a cuadro en ambos sentidos [6]." }]),
  LI([{ t: "Transformer encoder ", b: true }, { t: "(4 capas, d = 192, 4 cabezas, token [CLS] y embeddings posicionales aprendidos) [5]: la autoatención relaciona cualquier par de cuadros, por ejemplo la forma inicial y final de la mano, sin pasar por un estado recurrente." }]),
  LI([{ t: "Transfer learning: ", b: true }, { t: "preentrenamos el mismo Transformer para clasificar 450 señas distintas (solo con personas de entrenamiento, sin fuga) y lo ajustamos a las 50 con dos variantes: backbone congelado (solo la cabeza nueva) y fine-tuning completo con una tasa de aprendizaje 5 veces menor en el backbone." }]),
  P(`Entrenamiento. Entropía cruzada con label smoothing 0.1: el producto usa la distribución completa (la masa del top-3 decide el modo de la interfaz), así que evitamos probabilidades sobreconfiadas. AdamW (lr 5·10⁻⁴ en el Transformer, weight decay 0.05), calentamiento de 5 epochs y decaimiento coseno, recorte de gradiente, dropout y early stopping por validación. Aumentos de datos: espejo (personas zurdas), rotación, escala, traslación, cambios de velocidad, jitter y cuadros perdidos. Cada iteración se corre con 3 semillas.`),
  P(`Costo. Un modelo desde cero tarda de 0.6 a 4 minutos por semilla en una laptop con GPU; el preentrenamiento, ≈ 4 minutos, y el fine-tuning, menos de uno. Lo más caro es extraer los puntos (≈ 1 h de CPU para 15 300 videos). Entrenar cualquier versión en la nube costaría centavos.`),

  H1("5. Resultados de iteraciones"),
  ...TABLA(["Iteración", "Cambio respecto a la anterior", "Top-1 val.", "Top-1 prueba", "Top-3 prueba", "Valor USD/mes"],
    ["clasico_logreg", "it0_lstm_base", "it1_lstm_aug", "it2_tf_aug", "it4_tl_tf_congelado", "it5_tl_tf_finetune", "it7_tl_tf_webcam"].filter(k => it[k]).map(k => [
      k, {
        clasico_logreg: "Sin deep learning (línea base)",
        it0_lstm_base: "LSTM bidireccional, sin aumentos (v0)",
        it1_lstm_aug: "+ aumentos de datos y label smoothing",
        it2_tf_aug: "Transformer con aumentos (v1, modelo base)",
        it4_tl_tf_congelado: "TL: backbone preentrenado congelado",
        it5_tl_tf_finetune: "TL: fine-tuning completo",
        it7_tl_tf_webcam: "+ aumentos que imitan la cámara en vivo (v2)",
      }[k], it[k]["Top-1 val"], it[k]["Top-1 test"], it[k]["Top-3 test"], it[k]["Valor neto USD/mes (test)"]]),
    [2.1, 3.7, 1.5, 1.5, 1.5, 1.5], "Iteraciones (media ± desviación de 3 semillas; validación = 6 personas, prueba = 11 personas no vistas).", 14),
  P(`Qué aprendimos en cada paso. La LSTM sin aumentos sobreajusta: llega a ${it.it0_lstm_base["Top-1 train"]} de top-1 en entrenamiento y ${top(it.it0_lstm_base, "Top-1 test")} en prueba. Los aumentos la suben a ${top(it.it1_lstm_aug, "Top-1 test")}. El salto mayor es pasar a atención: el Transformer llega a ${top(it.it2_tf_aug, "Top-1 test")}, y un Transformer más chico rinde casi igual, señal de que el límite está en los datos y no en la capacidad.`),
  P(`Transfer learning frente al modelo base. Con la misma arquitectura, el fine-tuning (it5) sube un poco sobre el Transformer desde cero (it2) cuando hay 35 personas de entrenamiento, pero su ventaja real es la eficiencia en datos: entrenando con solo 9 personas, el modelo desde cero obtiene ${fmt(LC["tf_0.25"].top3, 3)} de top-3 y el preentrenado ${fmt(LC["tl_0.25"].top3, 3)}. Eso es lo que importa para crecer el vocabulario, donde cada seña nueva tendrá pocas grabaciones.`),
  P(`Dos problemas que solo vimos al probar el producto. (1) Primero extrajimos los puntos en modo IMAGE de MediaPipe, pero el navegador usa modo VIDEO (con seguimiento entre cuadros): el Transformer bajaba de ${fmt(IDAT.A_top1, 3)} a ${fmt(IDAT.B_top1, 3)} de top-1 con datos como los de la app; re-extraer todo en VIDEO lo corrigió (${fmt(IDAT.C_top1, 3)}). (2) Al pasar los videos por la app como una webcam en tiempo real, el modelo sufría con la seña cortada por la segmentación o la cámara muy cerca${RB["seña cortada: último 60 %"] ? ` (top-3 de ${RB["original"]["it5_tl_tf_finetune top3"]} a ${RB["seña cortada: último 60 %"]["it5_tl_tf_finetune top3"]})` : ""}. La iteración it7 repite el fine-tuning con aumentos que imitan esas fallas; en prueba limpia no pierde nada y es más robusta${RB["seña cortada: último 60 %"] ? ` (${RB["seña cortada: último 60 %"]["it7_tl_tf_webcam top3"]} en el mismo caso)` : ""}. Como también fue la mejor en validación, quedó como v2.`),
  ...IMG("curvas_entrenamiento.png", 520, "Curvas de pérdida (entrenamiento punteada, validación continua) y top-1 de validación."),
  ...TABLA(["Versión", "Modelo", "Top-1 [IC95]", "Top-3 [IC95]", "s/búsqueda", "Valor USD/mes [IC95]"],
    ["v0", "v1", "v2"].map(k => [k, v[k]["Iteración"], `${v[k]["Top-1 test"]} ${v[k]["IC95 top-1"]}`, `${v[k]["Top-3 test"]} ${v[k]["IC95 top-3"]}`,
      fmt(v[k]["Seg/búsqueda app"]), `${usd(v[k])} ${v[k]["IC95 valor"]}`]),
    [0.9, 2.3, 2.3, 2.3, 1.2, 2.6], "Versiones comparadas en valor (IC 95 % por bootstrap sobre personas).", 15),
  P(`En dinero, v1 genera ${dV(V1, V0)} USD/mes más que v0 y v2 ${dV(V2, V1)} más que v1; v2 frente a v0: +${dV(V2, V0)} USD/mes (IC 95 % [${ic("v2_menos_v0")}]). Las tres cuestan lo mismo de operar y tardan ≈ 2 ms por seña en CPU.`),

  H1("6. Despliegue y experiencia de usuario"),
  ...IMG("app_senabusca.png", 560, "SeñaBusca durante una búsqueda. Se oculta la imagen de la cámara (solo se ven los puntos que detecta MediaPipe): la mano en verde cuenta como seña; a la derecha, el top-3 con su probabilidad y un esqueleto animado de cada opción."),
  P(`La app es un servidor FastAPI con una interfaz web. MediaPipe corre en el navegador; la grabación empieza sola cuando la mano sube por encima de una línea de reposo y termina cuando baja, y solo los puntos viajan al servidor, que responde en ≈ ${fmt(N.latencia_api_ms[dep][0])} ms. Con un servidor de ${fmt(S.costo_fijo_mes)} USD/mes y ${fmt(S.busquedas_mes)} búsquedas, cada búsqueda cuesta ≈ ${((S.costo_fijo_mes + S.costo_pred * S.busquedas_mes) / S.busquedas_mes).toFixed(4)} USD; un solo proceso atiende decenas de búsquedas por segundo. Si la confianza es baja, la app lo dice ("no estoy seguro") y muestra diez opciones: así comunicamos la incertidumbre.`),
  ...IMG("arquitectura.png", 520, "Arquitectura del sistema desplegado."),
  P(`Validación en el producto. ${CS ? `Pasamos los ${CS.videos} videos de prueba por la app como una webcam en tiempo real: v2 acertó el ${pct(CS.modelos.v2.top1)} en la primera opción y el ${pct(CS.modelos.v2.top3)} en las tres mostradas (v0: ${pct(CS.modelos.v0.top1)} y ${pct(CS.modelos.v0.top3)}), unos 5 puntos menos que offline. Esa prueba nos mostró que la segmentación descartaba señas cortas; corregirlo subió el top-3 de 48/60 a 58/60. ` : ""}${PU ? `En una prueba piloto con un integrante del equipo (no tenemos más participantes, así que la reportamos como piloto), buscar a mano tomó ${fmt(PU.manual.t_mediana_s)} s y no encontró ninguna de ${PU.ensayos_manual} señas; con SeñaBusca resolvió ${PU.ensayos_app} de ${PU.ensayos_app} en una mediana de ${fmt(PU.app.t_total_mediana_s)} s. ` : ""}${VP ? `En una validación con ${VP.ensayos} señas aprendidas en el momento, la app acertó a la primera el ${pct(VP.top1_primer_intento, 0)} de las veces (${pct(VP.top3_primer_intento, 0)} en el top-3) y el ${pct(VP.top1_ultimo_intento, 0)} al repetir la seña.` : ""}`),
  P(`Monitoreo. La app registra cada predicción y cada respuesta del usuario ("Es esta" con su posición o "Ninguna es"). En producción vigilaríamos la confianza media y la fracción de búsquedas "no estoy seguro" (deriva por cámaras nuevas o señas fuera de vocabulario), la tasa de "Ninguna es" (proxy del top-3 real), la detección de manos y la latencia; las respuestas de los usuarios sirven como datos etiquetados para re-entrenar.`),

  H1("7. Explicabilidad, sesgos y riesgos"),
  P(`Ocultamos grupos de puntos para ver de qué depende el modelo: el top-3 cae de ${fmt(EXP.base_top3, 3)} a ${fmt(EXP["sin muñecas (pose)"], 3)} sin las muñecas, ${fmt(EXP["sin cara (nariz, ojos, boca)"], 3)} sin la cara y ${fmt(EXP["sin mano derecha"], 3)} sin la mano derecha. Es decir, usa la trayectoria de las manos y dónde se hacen respecto al cuerpo, que es lo que distingue señas como MOTHER (barbilla) y FATHER (frente). La saliencia por gradiente y la atención del token [CLS] se concentran en el centro de la seña y bajan en los cuadros de reposo. Con la seña reflejada (persona zurda) el modelo no cambia (${fmt(EXP["espejo (zurdo)"], 3)}), gracias al aumento de espejo.`),
  ...IMG("explicabilidad_oclusion.png", 400, "Oclusión por parte del cuerpo y robustez del modelo desplegado (top-3 en prueba)."),
  P(`Sesgos. El desempeño varía por persona: entre las ${PP.length} personas de prueba, el top-3 de v2 va de ${fmt(Math.min(...PP.map(r => +r.top3)), 2)} a ${fmt(Math.max(...PP.map(r => +r.top3)), 2)} y el top-1 de ${fmt(Math.min(...PP.map(r => +r.top1)), 2)} a ${fmt(Math.max(...PP.map(r => +r.top1)), 2)}. El factor que observamos es la detección de manos (luz, cámara, contraste con el fondo); el dataset no publica datos demográficos, así que no podemos atribuirlo a otra variable. El costo de esos errores recae en quien ya tiene peor detección: más búsquedas manuales o aprender un significado equivocado. Los pares que más se confunden comparten forma o ubicación (PLEASE ↔ HAPPY, GOOD ↔ THANKYOU).`),
  P(`Riesgos y límites. Reconoce 50 señas aisladas, no frases; no sustituye a un intérprete ni debe usarse en situaciones médicas o legales. Los puntos del cuerpo pueden considerarse biométricos: los registros no se publican y deberían anonimizarse. La licencia del dataset impide un uso comercial: un producto real necesitaría datos propios recolectados con consentimiento.`),

  H1("8. Conclusiones"),
  LI([{ t: "Qué movió más el valor: ", b: true }, { t: `en métrica técnica, pasar de LSTM a Transformer; en el producto, cosas que no se ven offline: alinear el preprocesamiento de entrenamiento con el de la app, corregir la segmentación y elegir el umbral por valor.` }]),
  LI([{ t: "Rendimientos decrecientes: ", b: true }, { t: "más grabaciones de las mismas 50 señas ya casi no suman valor; lo que paga es ampliar el vocabulario (p_vocab es de los supuestos más sensibles) y medir los supuestos con más estudiantes." }]),
  LI([{ t: "Complejidad, costo y valor: ", b: true }, { t: "las tres versiones cuestan lo mismo de operar; el preentrenamiento se justifica sobre todo por su eficiencia en datos para crecer el vocabulario." }]),
  LI([{ t: "Viabilidad: ", b: true }, { t: "técnicamente es barato (CPU, ≈ 2 ms por seña, funciona sin internet). Los obstáculos reales son la licencia de los datos y validar con más usuarios." }]),

  H1("9. Repositorio"),
  P("El repositorio incluye preprocesamiento, entrenamiento y evaluación (scripts 01–04), el modelo de valor (asl/valor.py y script 05), explicabilidad (06), la app (app/), las pruebas de la app (scripts 11–14 y la sesión guiada /?prueba=usuarios), el notebook ejecutado y un README con los pasos para instalar, correr la app y reproducir las métricas (scripts/run_all.sh)."),
  ...repo,

  H1("Referencias"),
  P("[1] A. Desai, L. Berger, F. O. Minakov, V. Milan, C. Singh, K. Pumphrey, R. E. Ladner, H. Daumé III, A. X. Lu, N. Caselli y D. Bragg, \"ASL Citizen: A Community-Sourced Dataset for Advancing Isolated Sign Language Recognition,\" arXiv:2304.05934, 2023.", { after: 60 }),
  P("[2] O. Alonzo, A. Glasser y M. Huenerfauth, \"Effect of Automatic Sign Recognition Performance on the Usability of Video-Based Search Interfaces for Sign Language Dictionaries,\" en Proc. ACM ASSETS, 2019, doi:10.1145/3308561.3353791.", { after: 60 }),
  P("[3] S. Hassan, O. Alonzo, A. Glasser y M. Huenerfauth, \"Effect of Ranking and Precision of Results on Users' Satisfaction with Search-by-Video Sign-Language Dictionaries,\" ACM Trans. Accessible Computing, 2021, doi:10.1145/3470650.", { after: 60 }),
  P("[4] C. Lugaresi et al., \"MediaPipe: A Framework for Building Perception Pipelines,\" arXiv:1906.08172, 2019.", { after: 60 }),
  P("[5] A. Vaswani et al., \"Attention Is All You Need,\" en NeurIPS, 2017.", { after: 60 }),
  P("[6] S. Hochreiter y J. Schmidhuber, \"Long Short-Term Memory,\" Neural Computation, vol. 9, no. 8, 1997.", { after: 0 }),
];

const doc = new Document({
  creator: "Esteban Cárcamo y Nicolás Concuá",
  title: "Proyecto Final — SeñaBusca",
  styles: {
    default: { document: { run: { font: FONT, size: BODY, color: "000000" } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: TITLE, bold: true, color: "000000" } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: TITLE, bold: true, color: "000000" } },
    ],
  },
  numbering: { config: [{ reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 400, hanging: 240 } } } }] }] },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1080, bottom: 1080, left: 1080, right: 1080 } } },
    children: [...cover, ...content],
  }],
});
Packer.toBuffer(doc).then(buf => { fs.writeFileSync(OUT, buf); console.log("WROTE", OUT, buf.length, "bytes"); });
