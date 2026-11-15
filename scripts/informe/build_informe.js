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
const repo = new Paragraph({ spacing: { after: 160 }, children: [
  new TextRun({ text: "Enlace al repositorio: ", bold: true, font: FONT, size: BODY }),
  new ExternalHyperlink({ link: REPO_URL, children: [new TextRun({ text: REPO_URL, font: FONT, size: BODY, color: "0563C1", underline: {} })] }),
] });

// ----------------------------------------------------------------------------- contenido
const S = N.supuestos;
const v = Object.fromEntries(vers.map(r => [r["Versión"], r]));
const dep = N.desplegado;

const content = [
  repo,
  H1("Resumen"),
  P([{ t: "SeñaBusca es una aplicación web que invierte la búsqueda en un diccionario de lengua de señas americana (ASL): el usuario hace una seña frente a la cámara y recibe las tres coincidencias más probables con su significado en español. El núcleo es un modelo de deep learning entrenado por el grupo sobre secuencias de puntos clave (MediaPipe) de 50 señas cotidianas del dataset ASL Citizen, evaluado con personas que el modelo nunca vio. " },
     { t: `La mejor versión (${v[dep]["Iteración"]}) logra ${pct(v[dep]["Top-1 test"])} de top-1 y ${pct(v[dep]["Top-3 test"])} de top-3 en prueba, ` , b: true },
     { t: `con ${fmt(v[dep]["Latencia CPU ms"], 1)} ms de inferencia en CPU. Con el modelo de valor propuesto, un programa de 500 estudiantes ahorraría ${fmt(v[dep]["Horas ahorradas/mes"])} horas al mes (≈ ${fmt(v[dep]["Valor neto USD/mes"])} USD/mes netos). ` },
     { t: N.frase_resumen_versiones },
     { t: " Ninguna parte del sistema usa un LLM: el modelo lo diseñamos y entrenamos nosotros, y la versión con transfer learning se compara siempre contra su modelo base." }]),

  // 4.1
  H1("1. Contexto de negocio y definición del problema"),
  P("Quien aprende ASL se encuentra constantemente con señas que no conoce: en clase, en videos o en una conversación con una persona sorda. Los diccionarios (Lifeprint, Handspeak, ASL Signbank) están organizados por palabra en inglés, así que sirven para ir de palabra → seña, pero no de seña → palabra. Hoy la alternativa es buscar por rasgos (forma de la mano, ubicación, movimiento) o preguntarle al docente; es lento y muchas veces no se encuentra. Investigaciones de accesibilidad sobre diccionarios de ASL por video muestran que la satisfacción del usuario depende de que la seña buscada aparezca en las primeras posiciones de la lista de resultados [2], [3], lo que hace del top-k la métrica técnica natural de este producto."),
  P([{ t: "Usuario y organización. ", b: true }, { t: "Un programa de enseñanza de ASL (curso universitario o plataforma de aprendizaje) con 500 estudiantes activos que hacen en promedio 20 búsquedas de señas desconocidas al mes (10 000 búsquedas/mes). El vocabulario del prototipo (50 señas) se alinea con el de un primer curso: saludos, familia, escuela, preguntas, emociones, necesidades básicas y tiempo." }]),
  P([{ t: "¿Por qué deep learning? ", b: true }, { t: `La misma seña varía entre personas (velocidad, amplitud, mano dominante), cámaras e iluminación, y lo que la define es la trayectoria temporal de las manos respecto al cuerpo. Se probó una alternativa más simple —regresión logística y k-NN sobre estadísticos de los puntos clave, sin deep learning—: alcanza ${N.clasico_top1} de top-1 y ${N.clasico_top3} de top-3 en prueba. El mejor modelo de deep learning reduce el error top-3 de ${N.clasico_err3} a ${N.dep_err3}, y eso vale ${N.delta_vs_clasico} USD/mes adicionales según el modelo de valor (sección 3): la complejidad se justifica.` }]),
  P([{ t: "KPIs y línea base. ", b: true }, { t: `(1) tiempo por búsqueda: ${fmt(S.t_manual)} s promedio en la búsqueda manual, ${fmt(N.costo_manual_seg)} s equivalentes al sumar el costo de las búsquedas fallidas; (2) búsquedas resueltas: ${pct(S.p_manual, 0)} con el método manual; (3) tasa de "engaños" (el usuario se queda con un significado equivocado): 0 % en manual porque no hay sugerencias; (4) costo de operación por búsqueda.` }]),

  // 4.2
  H1("2. Datos y análisis exploratorio"),
  P([{ t: "Fuente. ", b: true }, { t: `ASL Citizen [1] (Microsoft Research, 2023): 83 399 videos de 2 731 señas grabados con webcam por 52 personas sordas o con discapacidad auditiva, con división oficial train/val/test por participante. Se usan dos subconjuntos: las 50 señas objetivo (${N.eda.obj_videos} videos: ${N.eda.obj_train} train / ${N.eda.obj_val} val / ${N.eda.obj_test} test) y 450 señas adicionales elegidas al azar con semilla fija para preentrenar el modelo v2 (${N.eda.pre_videos} videos). En lugar de descargar los 46 GB del zip oficial, el script 01 lee el índice del zip remoto y extrae por rangos HTTP solo los ~15 300 videos usados.` }]),
  P([{ t: "Calidad y representatividad. ", b: true }, { t: `Las 50 clases están prácticamente balanceadas (≈ 30 videos por seña), por lo que accuracy y F1 macro casi coinciden. Los videos duran en mediana ${N.eda.cuadros_mediana} cuadros (≈ ${N.eda.seg_mediana} s a 30 fps). MediaPipe detecta la pose en el ${pct(N.eda.frac_pose, 0)} de los cuadros, pero las manos solo en el ${pct(N.eda.frac_manos, 0)} (incluye el reposo antes y después de la seña, cuando las manos están fuera de cuadro). La detección de manos varía mucho por persona: del ${pct(N.eda.manos_min_part, 0)} al ${pct(N.eda.manos_max_part, 0)} de los cuadros según el participante, una fuente de sesgo que se retoma en la sección 6. Las grabaciones son caseras (fondos, luz y cámaras variadas), lo que se parece al uso real de la app.` }]),
  P([{ t: "Preprocesamiento. ", b: true }, { t: "Por cuadro se extraen 53 puntos (x, y, z): 11 de la parte superior del cuerpo (nariz, ojos, boca, hombros, codos, muñecas) y 21 por mano. Cada mano se asigna a izquierda/derecha por cercanía a la muñeca de la pose, no con la etiqueta de MediaPipe, para que el resultado sea idéntico en el video original y en la cámara reflejada del navegador. Luego se recortan los cuadros de reposo, se centra en el punto medio de los hombros y se escala por el ancho de hombros (invariancia a posición y distancia), y se remuestrea a 64 cuadros; con dos indicadores de presencia de mano cada seña queda como un tensor 64 × 161. El mismo módulo (asl/features.py) lo usan el entrenamiento y la API, lo que evita diferencias entre entrenamiento y producción." }]),
  P([{ t: "División y fugas. ", b: true }, { t: "Se respeta la división oficial por participante (35 personas en train, 6 en val, 11 en test): el usuario de la app nunca está en el entrenamiento, que es justamente el uso real. El preentrenamiento de v2 usa solo participantes de entrenamiento y solo señas distintas de las 50 objetivo, así que no filtra información a prueba." }]),
  P([{ t: "Privacidad y licencia. ", b: true }, { t: "La licencia de ASL Citizen permite uso de investigación no comercial y prohíbe redistribuir los datos o sus modificaciones; por eso los videos, los landmarks y los cachés no se publican (.gitignore), y el README documenta cómo obtenerlos. En la app, el video del usuario nunca sale del navegador: solo se envían los puntos clave." }]),
  ...IMG("eda_calidad_deteccion.png", 600, "Largo de los videos, fracción de cuadros con mano detectada y detección por participante (señas objetivo)."),

  // 4.3
  H1("3. Modelo de valor"),
  P("Cada búsqueda se valora en segundos equivalentes de tiempo del estudiante. Sin la app, el costo esperado es C_manual = t_manual + (1 − p_manual) · c_no_resuelta. Con la app, el usuario graba la seña (t_grabar) y el modelo entrega una distribución de probabilidad; la confianza c es la masa de probabilidad del top-3 y el umbral τ decide qué se muestra:"),
  LI([{ t: "c ≥ τ, modo seguro: ", b: true }, { t: "se muestran 3 opciones (t_rev3). Si la correcta está ahí, la búsqueda se resuelve. Si no está, con probabilidad p_engano el usuario acepta una opción equivocada (costo c_error: aprender o usar mal una seña); si no se engaña, abre la lista de 10 (t_rev10) y, si tampoco está, busca a mano." }]),
  LI([{ t: "c < τ, modo \"no estoy seguro\": ", b: true }, { t: "se muestran 10 opciones con un aviso (t_rev10); el aviso reduce el engaño a p_engano_aviso. Si la correcta no está, se busca a mano." }]),
  LI([{ t: "Señas fuera del vocabulario: ", b: true }, { t: `una fracción 1 − p_vocab de las búsquedas son de señas que el prototipo no conoce. Su comportamiento no se supone: se mide con los videos de validación y prueba de las 450 señas de preentrenamiento (personas no vistas), que hacen de "consultas fuera de vocabulario". Ahí el modo "no estoy seguro" es lo único que evita engaños.` }]),
  P([{ t: "Valor neto mensual = (C_manual − E[C_app]) · búsquedas/mes · valor_hora / 3600 − costo de operación.", b: true }]),
  ...TABLA(["Supuesto", "Valor", "Fuente / justificación"], N.tabla_supuestos, [2.2, 1, 6.5], "Supuestos del modelo de valor. t_manual, t_rev3 y t_grabar se midieron en la prueba piloto (docs/RESULTADOS_PRUEBA_USUARIOS.md); el resto son estimaciones justificadas."),
  P([{ t: "Punto de operación. ", b: true }, { t: `El umbral τ se elige maximizando el valor en VALIDACIÓN y el valor se reporta en PRUEBA, para no sobreajustar la decisión de negocio. Con el modelo desplegado (${dep}) el umbral óptimo es τ = ${v[dep]["Umbral"]}: ${N.frase_umbral}` }]),
  ...IMG("valor_vs_umbral.png", 470, "Curva de valor según el umbral de confianza para v0, v1 y v2 (prueba). El punto marca el umbral elegido en validación."),
  ...IMG("valor_vs_top3.png", 470, "Vínculo métrica técnica → valor: cada punto es una corrida (iteración × semilla, incluidas las de menos datos). La línea punteada es la aproximación analítica."),
  P([{ t: "Curva de valor. ", b: true }, { t: N.frase_curva_valor }]),
  P([{ t: "Análisis de sensibilidad. ", b: true }, { t: N.frase_sensibilidad }]),
  ...IMG("sensibilidad_tornado.png", 470, `Sensibilidad del valor neto del modelo desplegado (${dep}) a cambios de ±20 % y ±50 % en cada supuesto, re-optimizando el umbral.`),

  // 4.4
  H1("4. Metodología de desarrollo"),
  P([{ t: "Arquitecturas. ", b: true }, { t: "Se compararon cuatro familias, todas sobre la misma entrada de 64 × 161: (a) baseline clásico sin deep learning; (b) LSTM bidireccional de 2 capas (192 unidades) con promedio temporal, como en el laboratorio de RNN/LSTM; (c) Transformer encoder (4 capas, d = 192, 4 cabezas, pre-norm, token [CLS] y embeddings posicionales aprendidos), como en la hoja de trabajo de atención; y (d) transfer learning: el mismo Transformer (o la LSTM) preentrenado para clasificar 450 señas distintas y luego ajustado a las 50 objetivo, como en el laboratorio de transfer learning, en dos variantes: backbone congelado (solo se entrena la cabeza) y fine-tuning completo con una tasa de aprendizaje 5 veces menor para el backbone. La atención es adecuada porque una seña se define por la relación entre su configuración inicial y final, y el preentrenamiento aprovecha que las 450 señas extra enseñan \"cómo se mueven las manos\" con 9 veces más datos." }]),
  P([{ t: "Restricción del curso. ", b: true }, { t: `El proyecto no es un caso de uso de un LLM. Todos los modelos (LSTM y Transformer) se diseñaron y entrenaron desde cero sobre puntos clave, y el transfer learning se hizo con un preentrenamiento propio sobre 450 señas de ASL Citizen. Como pide la excepción de fine-tuning y transfer learning, el cambio en el desempeño se documenta frente al modelo base con la misma arquitectura (it2, Transformer desde cero): ${N.frase_tl ? "ver sección 5" : ""}.` }]),
  P([{ t: "Pérdida y alineación con el negocio. ", b: true }, { t: "Entropía cruzada con label smoothing 0.1. El producto no usa solo la clase más probable sino la distribución completa (top-3 y su masa de probabilidad decide el modo de la interfaz), así que importa que las probabilidades sean útiles para separar aciertos de fallos; el label smoothing evita probabilidades sobreconfiadas (de hecho, deja al modelo algo subconfiado, ver sección 6). La selección de epoch se hace con top-1 de validación y el punto de operación con valor de validación." }]),
  P([{ t: "Optimización y regularización. ", b: true }, { t: "AdamW (lr 1e-3 para LSTM, 5e-4 para Transformer; weight decay 0.05 en Transformer), calentamiento lineal de 5 epochs y decaimiento coseno, recorte de gradiente a 1.0, dropout 0.2–0.3, early stopping por paciencia, lotes de 64. Aumentos de datos en entrenamiento: espejo horizontal (simula personas zurdas e intercambia manos y puntos del cuerpo), rotación ±15°, escala ±15 %, traslación, deformación temporal (recorte y re-interpolación: velocidades distintas), jitter gaussiano y pérdida de cuadros. Cada iteración se corrió con 3 semillas y se reporta media ± desviación." }]),
  P([{ t: "Costo computacional. ", b: true }, { t: N.frase_costo }]),

  // 4.5
  H1("5. Resultados de iteraciones"),
  ...TABLA(["Iteración", "Cambio", "Top-1 val", "Top-1 test", "Top-3 test", "F1 macro test", "Valor USD/mes", "Entren. (min)"],
    iter.map(r => [r["Iteración"], r["Cambio"], r["Top-1 val"], r["Top-1 test"], r["Top-3 test"], r["F1 macro test"], r["Valor neto USD/mes (test)"], r["Entrenamiento (min, M4 Pro)"]]),
    [1.6, 3.2, 1.1, 1.1, 1.1, 1.1, 1.2, 1.1], "Una fila por iteración (media ± desviación sobre 3 semillas; clásicos, 1 corrida). Valor neto con el umbral elegido en validación.", 14),
  ...TABLA(["Versión", "Iteración", "Top-1 test [IC95]", "Top-3 test [IC95]", "Seg/búsqueda", "Resueltas", "Engaños", "Valor neto USD/mes [IC95]", "Latencia ms"],
    vers.map(r => [r["Versión"], r["Iteración"], `${r["Top-1 test"]} ${r["IC95 top-1"]}`, `${r["Top-3 test"]} ${r["IC95 top-3"]}`, r["Seg/búsqueda app"], r["Resueltas"], r["Engaños"], `${fmt(r["Valor neto USD/mes"])} ${r["IC95 valor"]}`, r["Latencia CPU ms"]]),
    [0.8, 1.6, 1.6, 1.6, 1, 1, 1, 1.9, 0.9], "Las tres versiones comparadas en valor (semilla con mejor validación). IC 95 % por bootstrap remuestreando participantes de prueba.", 14),
  P([{ t: "Lectura. ", b: true }, { t: N.frase_iteraciones }]),
  P([{ t: "Comparación en valor. ", b: true }, { t: N.frase_versiones }]),
  P([{ t: "Transfer learning frente al modelo base. ", b: true }, { t: N.frase_tl }]),
  ...IMG("curvas_entrenamiento.png", 560, "Curvas de pérdida (train punteada, validación continua) y top-1 de validación de las iteraciones relevantes (semilla 0)."),
  P([{ t: "Problemas encontrados. ", b: true }, { t: N.frase_problemas }]),
  P([{ t: "Iteración de datos: desajuste entre entrenamiento y producción. ", b: true }, { t: N.frase_iteracion_datos }]),
  ...(N.frase_webcam ? [P([{ t: "Iteración guiada por la cámara en vivo (it7). ", b: true }, { t: N.frase_webcam }])] : []),
  ...TABLA(["Iteración", "A: entrena IMAGE / evalúa IMAGE", "B: entrena IMAGE / evalúa VIDEO (= app)", "C: entrena VIDEO / evalúa VIDEO"],
    csv("tabla_iteracion_datos.csv").map(r => Object.values(r)),
    [2, 2.6, 2.6, 2.6], "Top-1 | top-3 en prueba (media de 3 semillas) según el modo de extracción de MediaPipe usado al entrenar y al evaluar.", 15),
  ...IMG("curva_aprendizaje.png", 520, "Curva de aprendizaje: desempeño y valor al entrenar con 25, 50, 75 y 100 % de los participantes."),

  // 4.6
  H1("6. Despliegue, experiencia de usuario, explicabilidad y riesgos"),
  H2("6.1 Arquitectura desplegada"),
  ...IMG("arquitectura.png", 600, "Arquitectura del sistema desplegado."),
  P(`La app (app/) es un servidor FastAPI que sirve la interfaz web y expone /api/predict. MediaPipe Tasks corre en el navegador (WASM/GPU, ≈ 15 ms por cuadro), así que solo viajan los puntos clave (≈ 60 KB por seña) y no el video. La grabación empieza sola cuando aparecen las manos por encima de la línea de reposo durante 4 cuadros y termina cuando bajan por 18 (0.6 s), de modo que el usuario no toca botones; hay respaldo manual (grabar 4 s) y la opción de subir un video, que pasa por exactamente el mismo flujo. Un selector permite cambiar en vivo entre v0, v1 y v2 para la demostración. ${N.frase_latencia} Costo por predicción: con un servidor CPU de ${fmt(S.costo_fijo_mes)} USD/mes y ${fmt(S.busquedas_mes)} búsquedas al mes, cada búsqueda cuesta ≈ ${((S.costo_fijo_mes + S.costo_pred * S.busquedas_mes) / S.busquedas_mes).toFixed(4)} USD (el costo marginal es ${S.costo_pred} USD); con 100 veces más usuarios el mismo servidor seguiría sobrando y el costo por búsqueda bajaría a ≈ ${((S.costo_fijo_mes + S.costo_pred * S.busquedas_mes * 100) / (S.busquedas_mes * 100)).toFixed(5)} USD.`),
  P([{ t: "Validación en el producto. ", b: true }, { t: N.frase_e2e }]),
  ...(N.frase_camara ? [P([{ t: "Cámara en tiempo real. ", b: true }, { t: N.frase_camara }])] : []),
  ...(N.frase_usuarios ? [P([{ t: "Prueba con usuarios. ", b: true }, { t: N.frase_usuarios }])] : []),
  P([{ t: "Experiencia de usuario. ", b: true }, { t: `Un mejor modelo cambia lo que ve el usuario: con mayor top-3 hay menos búsquedas que terminan en la lista de 10 o en la búsqueda manual, y con mejor separación de confianza hay menos "engaños". ${N.frase_ux}` }]),
  P([{ t: "Incertidumbre, errores y casos límite. ", b: true }, { t: "Si la confianza del top-3 está por debajo de τ, la interfaz lo dice (\"No estoy seguro\") y muestra 10 opciones; si hay menos de 5 cuadros con manos, pide repetir la seña acercándose a la cámara; cada tarjeta muestra la probabilidad y un esqueleto animado de cómo se hace la seña para que el usuario verifique antes de aceptar." }]),
  P([{ t: "Monitoreo. ", b: true }, { t: "Cada predicción y cada respuesta del usuario (\"Es esta\" con su posición, o \"Ninguna es\") se registran en logs/app_eventos.jsonl. En producción se vigilaría: (1) la confianza media y la fracción de búsquedas en modo \"no estoy seguro\" (deriva de datos: cámaras nuevas, señas fuera de vocabulario), (2) la tasa de \"Ninguna es\" y la posición elegida (proxy del top-3 real), (3) la fracción de cuadros con manos detectadas (calidad de entrada) y (4) la latencia p95. Los ejemplos etiquetados por los usuarios sirven para re-entrenar." }]),
  H2("6.2 Explicabilidad"),
  P(N.frase_explicabilidad),
  ...IMG("explicabilidad_oclusion.png", 470, "Oclusión por parte del cuerpo y pruebas de robustez del modelo desplegado (top-3 en prueba)."),
  ...IMG("explicabilidad_saliencia.png", 560, "Saliencia por gradiente por punto y a lo largo del tiempo, y atención del token [CLS] en la última capa."),
  H2("6.3 Sesgos"),
  P(N.frase_sesgos),
  ...IMG("por_participante.png", 470, "Desempeño por participante de prueba (personas no vistas)."),
  ...IMG("calibracion.png", 560, "Calibración de la confianza top-3 y separación entre aciertos, fallos y señas fuera de vocabulario."),
  H2("6.4 Riesgos de uso indebido y límites"),
  LI([{ t: "No es un traductor. ", b: true }, { t: "Reconoce señas aisladas de un vocabulario de 50; no traduce oraciones, no entiende gramática de ASL (expresiones faciales, clasificadores) y no reconoce LENSEGUA. No debe usarse para comunicarse en situaciones críticas (salud, legal, emergencias), donde se necesita un intérprete." }]),
  LI([{ t: "Privacidad. ", b: true }, { t: "El video no sale del navegador; aun así, los puntos clave del cuerpo pueden ser biométricos. Los registros de la app no se publican y deberían anonimizarse y tener retención limitada. La licencia del dataset prohíbe usarlo comercialmente: un producto real requeriría licenciar datos o recolectar los propios con consentimiento." }]),
  LI([{ t: "Comunidad. ", b: true }, { t: "Herramientas de reconocimiento de señas hechas sin la comunidad sorda han sido criticadas; un despliegue real debería co-diseñarse con personas sordas y docentes de ASL, como lo propone el propio proyecto ASL Citizen." }]),

  // 4.8
  H1("7. Discusión y conclusiones"),
  P([{ t: "Qué movió más el valor. ", b: true }, { t: N.frase_mayor_impacto }]),
  P([{ t: "Rendimientos decrecientes. ", b: true }, { t: N.frase_rendimientos }]),
  P([{ t: "Complejidad, costo y valor. ", b: true }, { t: N.frase_tradeoff }]),
  P([{ t: "Limitaciones y viabilidad. ", b: true }, { t: "El vocabulario de 50 señas es el principal límite comercial: el valor depende de que las búsquedas caigan dentro del vocabulario (p_vocab), por lo que el siguiente paso es escalar a cientos de señas (ASL Citizen tiene 2 731; con este pipeline la extracción y el entrenamiento escalan linealmente y el costo de inferencia no cambia). Los supuestos de tiempo y comportamiento deben medirse con usuarios reales (protocolo en docs/). La inferencia cuesta fracciones de centavo por búsqueda y corre en un servidor CPU pequeño, así que la viabilidad técnica es alta; la barrera real es la licencia de datos y la adopción en un programa de ASL." }]),

  // 4.9
  H1("8. Repositorio"),
  P("El repositorio contiene el preprocesamiento (scripts 01–02, asl/features.py), el entrenamiento y la evaluación (03, 03b, 04), el modelo de valor (asl/valor.py, script 05), la explicabilidad (06), la app desplegada (app/), el notebook ejecutado (notebook/proyecto_final.ipynb) y un README con instrucciones para instalar, ejecutar la app y reproducir todas las métricas (scripts/run_all.sh)."),
  repo,

  H("Referencias", HeadingLevel.HEADING_1, 200, 60),
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
