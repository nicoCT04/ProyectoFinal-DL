/*
 * Informe del Laboratorio 10 (Explainable AI) — genera reports/Informe_Lab10_XAI.docx
 *
 *   npm install docx@9            (una vez, en esta carpeta o con NODE_PATH)
 *   node xai/informe/build_informe_lab10.js
 *
 * Lee los números de reports/xai/resumen.json y las figuras de reports/xai/ (salida de
 * xai/generar_xai.py). Estilo de los informes del curso: Arial 11 / títulos Arial 12 negrita,
 * carta, portada UVG en página propia + 4 páginas de contenido.
 */
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, ImageRun, PageBreak,
  ExternalHyperlink, LevelFormat, Table, TableRow, TableCell, WidthType, BorderStyle, ShadingType,
} = require("docx");

const ROOT = path.resolve(__dirname, "..", "..");
const XAI = path.join(ROOT, "reports", "xai");
const OUT = path.join(ROOT, "reports", "Informe_Lab10_XAI.docx");
const REPO_URL = "https://github.com/nicoCT04/ProyectoFinal-DL";
const LOGO = [path.join(ROOT, "scripts", "informe", "uvg_logo.png"),
              path.resolve(ROOT, "..", "uvg_logo.png")].find(fs.existsSync);
const R = JSON.parse(fs.readFileSync(path.join(XAI, "resumen.json"), "utf8"));
const f2 = x => x.toFixed(2), f3 = x => x.toFixed(3), pct = x => (100 * x).toFixed(1) + " %";

const FONT = "Arial", BODY = 22, SMALL = 18, TITLE = 24;
function runs(parts, size = BODY) {
  if (typeof parts === "string") parts = [{ t: parts }];
  return parts.map(p => new TextRun({ text: p.t, bold: !!p.b, italics: !!p.i, font: FONT, size, color: "000000" }));
}
const P = (parts, after = 60) => new Paragraph({ alignment: AlignmentType.JUSTIFIED,
  spacing: { after, line: 240 }, children: runs(parts) });
const H1 = t => new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 120, after: 40 },
  children: [new TextRun({ text: t, bold: true, font: FONT, size: TITLE, color: "000000" })] });
const LI = parts => new Paragraph({ numbering: { reference: "bullets", level: 0 }, alignment: AlignmentType.JUSTIFIED,
  spacing: { after: 30, line: 240 }, children: runs(parts) });
const C = (text, { b = false, size = BODY, after = 40 } = {}) => new Paragraph({ alignment: AlignmentType.CENTER,
  spacing: { after }, children: [new TextRun({ text, bold: b, font: FONT, size, color: "000000" })] });
function IMG(file, wIn, caption) {
  const buf = fs.readFileSync(path.join(XAI, file));
  const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20);          // tamaño real del PNG
  const W = Math.round(wIn * 96), H = Math.round(W * h / w);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 40, after: 10 },
      children: [new ImageRun({ type: "png", data: buf, transformation: { width: W, height: H } })] }),
    new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 80 },
      children: runs([{ t: caption, i: true }], SMALL) }),
  ];
}
const borde = { style: BorderStyle.SINGLE, size: 4, color: "808080" };
function TABLA(filas, anchos) {   // filas[0] = encabezado; celdas: string
  const total = anchos.reduce((a, b) => a + b, 0);
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: anchos,
    rows: filas.map((f, r) => new TableRow({ children: f.map((c, k) => new TableCell({
      width: { size: anchos[k], type: WidthType.DXA },
      borders: { top: borde, bottom: borde, left: borde, right: borde },
      shading: r === 0 ? { type: ShadingType.CLEAR, color: "auto", fill: "E7E9EE" } : undefined,
      margins: { top: 20, bottom: 20, left: 60, right: 60 },
      children: [new Paragraph({ children: runs([{ t: c, b: r === 0 }], SMALL) })],
    })) })),
  });
}
const ESP = () => new Paragraph({ spacing: { after: 60 }, children: [] });

// ---------------------------------------------------------------- portada
const cover = [
  new Paragraph({ spacing: { before: 1200 }, children: [] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 },
    children: [new ImageRun({ type: "png", data: fs.readFileSync(LOGO), transformation: { width: 150, height: 150 } })] }),
  C("Universidad del Valle de Guatemala", { b: true }),
  C("Facultad de Ingeniería"),
  C("CC3092 — Deep Learning y Sistemas Inteligentes", { after: 260 }),
  C("Laboratorio #10", { b: true, size: 40, after: 120 }),
  C("Explainable AI sobre SeñaBusca: ¿en qué se fija el reconocedor de señas?", { b: true, size: 26, after: 320 }),
  C("Esteban Cárcamo (23016)", { b: true, after: 20 }),
  C("Nicolás Concuá (23197)", { b: true, after: 60 }),
  C("Guatemala, noviembre de 2026"),
  new Paragraph({ children: [new PageBreak()] }),
];

const cg = R.cuota_global, cl = R.cuota_logistica, A = R.auc_borrado, K = R.costo_ms;

// ---------------------------------------------------------------- contenido
const content = [
  P([{ t: "Contexto. ", b: true }, { t: `Se explica el modelo que defendimos en el proyecto final: la v2 de SeñaBusca (Transformer preentrenado en 450 señas de ASL Citizen y ajustado a 50, it7), sin reentrenarlo. Su entrada es una seña como secuencia de 64 cuadros × 161 variables: 53 puntos de MediaPipe (cara, brazos, 21 por mano) normalizados a los hombros, más dos banderas de presencia de mano. En prueba (${R.n_test} videos de 11 personas no vistas) tiene top-1 = ${f3(R.top1)} y la app solo muestra resultados con confianza ≥ ${R.umbral}. La audiencia son estudiantes de ASL que buscan el significado de una seña.` }]),

  H1("1. Investigación: métodos de XAI para secuencias de puntos clave"),
  P("Nuestro dato es una serie de tiempo multivariada (un esqueleto en movimiento), así que aplican los métodos de series de tiempo y los de gradiente; los de imágenes que dependen de mapas convolucionales no."),
  LI([{ t: "Saliencia (|∂y/∂x|) y SmoothGrad [1]: ", b: true }, { t: "gradiente de la clase respecto a la entrada (SmoothGrad lo promedia con ruido). Requieren gradientes; cuestan 1 o n pasadas. Ruidosos y con saturación." }]),
  LI([{ t: "Integrated Gradients (IG) [2]: ", b: true }, { t: "integra el gradiente en el camino recto de una línea base x′ a x. Requiere gradientes y una línea base; cuesta n pasos (64 aquí). Cumple completitud, pero la explicación depende de la línea base." }]),
  LI([{ t: "SHAP (KernelSHAP, GradientSHAP, DeepSHAP) [3]: ", b: true }, { t: "valores de Shapley sobre variables o grupos. KernelSHAP es caja negra pero necesita miles de evaluaciones por instancia; GradientSHAP ≈ IG promediado sobre un conjunto de fondo." }]),
  LI([{ t: "LIME [4] y Anchors: ", b: true }, { t: "ajustan un modelo lineal (o una regla) en un vecindario perturbado. Caja negra; inestable entre semillas y sensible al kernel; con 10 304 entradas correlacionadas hay que agrupar en bloques." }]),
  LI([{ t: "Oclusión por ventanas [5]: ", b: true }, { t: "reemplaza un bloque (parte del cuerpo × ventana de tiempo) y mide la caída de probabilidad. Caja negra, una pasada por bloque; no capta interacciones y el reemplazo puede sacar la entrada de distribución." }]),
  LI([{ t: "Específicos de series (TimeSHAP, Dynamask, WinIT) [6–8]: ", b: true }, { t: "SHAP por eventos para RNN, máscaras aprendidas por instancia (costosas) y atribución por ventanas. Atención [CLS] del Transformer: gratis, pero no es una explicación fiel [9]. Grad-CAM no aplica: el modelo no tiene capas convolucionales." }]),
  P([{ t: "Interpretable por diseño vs. a posteriori. ", b: true }, { t: `Un modelo interpretable (regresión logística, árbol pequeño) es su propia explicación; una explicación a posteriori aproxima un modelo opaco y puede no serle fiel. Sí tenemos un baseline interpretable: la regresión logística del proyecto sobre estadísticos de cada coordenada, con top-1 = ${f3(R.top1_logistica)} (29 puntos menos que la v2). Lo usamos para comparar en qué partes del cuerpo se apoya cada modelo (Fig. 2).` }]),
  P([{ t: "Propiedades formales. ", b: true }, { t: "IG cumple completitud (las atribuciones suman f(x) − f(x′)), sensibilidad e invariancia a la implementación: garantiza que toda la diferencia de logit queda repartida, no que el reparto sea causal. SHAP cumple aditividad/eficiencia y simetría respecto al fondo elegido; LIME solo promete fidelidad local en el vecindario muestreado." }]),
  P([{ t: "Parámetros de diseño que cambian la explicación: ", b: true }, { t: "la línea base de IG y su número de pasos; el objetivo (logit vs. probabilidad); el conjunto de fondo de SHAP; el kernel y el número de muestras de LIME; el tamaño de bloque y la estrategia de reemplazo en oclusión, y cómo se agregan las 161 variables. Comparamos tres líneas base y dos métodos con una métrica de fidelidad." }]),

  H1("2. Selección del método"),
  TABLA([
    ["Criterio", "Integrated Gradients (elegido)", "Oclusión por bloques (alternativa)", "KernelSHAP por bloques (descartado)"],
    ["Compatibilidad", "Modelo diferenciable en PyTorch: captum directo", "Cualquier modelo (caja negra)", "Cualquier modelo (caja negra)"],
    ["Tipo de explicación", "Atribución con signo por punto y cuadro (10 304 valores), agregable", "Caída de probabilidad por 32 bloques", "Shapley por 32 bloques"],
    ["Costo (CPU, 1 hilo)", `${K.ig_ms.toFixed(0)} ms por seña (64 pasos)`, `${K.oclusion_ms.toFixed(0)} ms por seña`, "≈ 2 000 evaluaciones por seña: varios s"],
    ["Comunicación", "Mano y momento resaltados sobre el esqueleto", "Fácil de explicar, pero de baja resolución", "Igual que oclusión"],
    ["Limitaciones", "Depende de la línea base; gradiente local", "Reemplazo fuera de distribución; sin interacciones", "Muestreo ruidoso, caro"],
    ["Fidelidad medida (AUC de borrado ↓)", f3(A["IG (reposo)"]), f3(A["Oclusión"]), "no se calculó"],
  ], [1900, 2900, 2600, 2600]),
  P([{ t: `Elegimos IG porque el modelo es diferenciable, cumple completitud, da resolución de punto y cuadro y fue el más fiel en la curva de borrado (Fig. 3). La oclusión queda como chequeo de caja negra y KernelSHAP se descartó por costo.`, i: false }], 40),

  H1("3. Implementación"),
  TABLA([
    ["Parámetro", "Valor y justificación"],
    ["Línea base", "«Reposo»: la misma persona con la pose promediada en el tiempo y las manos no detectadas (coordenadas 0 y bandera 0), que es como el preprocesamiento codifica «sin manos». Es una entrada que la cámara sí produce. Se compararon «ceros» (fuera de distribución) y «seña media de entrenamiento»."],
    ["Objetivo y pasos", "Logit de la clase predicha (el softmax satura); 64 pasos con cuadratura de Gauss-Legendre: error de completitud < 10⁻³."],
    ["Agregación", "Suma con signo por parte del cuerpo (cara, brazos, mano izq., mano der., presencia de mano) y por cuadro. Para lo global, las manos se alinean a la dominante (la que más se mueve)."],
    ["Muestra", `Global: los ${R.n_test} videos de prueba. Fuera de vocabulario: ${R.n_oov} videos de las 450 señas de preentreno (personas no vistas).`],
    ["Fidelidad y cordura", "Curva de borrado de 32 bloques (4 partes × 8 ventanas) [10]; prueba de cordura reinicializando la cabeza y la última capa del encoder [11]."],
    ["Reproducibilidad", "SEED = 23197, CPU con 1 hilo y algoritmos deterministas: dos corridas dan el mismo resumen. Todo sale de xai/generar_xai.py (≈ 6 min)."],
  ], [1900, 8100]),
  ESP(),
  ...IMG("fig_casos_locales.png", 5.7, "Figura 1. Explicaciones locales con IG. Izquierda: atribución con signo por parte del cuerpo y cuadro (rojo = a favor de la clase predicha). Derecha: esqueleto en el cuadro de mayor atribución; el tamaño de cada punto es su |atribución|."),
  ...IMG("fig_global.png", 6.3, "Figura 2. Explicación global. Izquierda: |IG| promedio en prueba por parte y cuadro. Derecha: proporción de la importancia por parte, v2 (IG) contra regresión logística (|coeficientes| estandarizados)."),
  ...IMG("fig_fidelidad.png", 6.1, "Figura 3. Izquierda: probabilidad de la clase predicha al borrar bloques en el orden de cada explicación (más rápido = más fiel). Derecha: acuerdo entre rankings de bloques: IG vs. oclusión y IG vs. IG del modelo con pesos aleatorios."),

  H1("4. Interpretación en el contexto del proyecto"),
  P([{ t: "¿Señales con sentido? ", b: true }, { t: `Sí. La mano dominante se lleva ${pct(cg["mano dominante"])} de la atribución, la no dominante ${pct(cg["mano no dominante"])}, los brazos ${pct(cg["brazos"])} y la cara ${pct(cg["cara"])}. Coincide con la fonología de ASL (forma, ubicación y movimiento de la mano). La importancia se concentra en el tramo central de la seña: los 16 cuadros de los extremos (25 % del tiempo) pesan 8.4 %. La logística reparte casi igual entre las dos manos (${f2(cl["mano dominante"])} y ${f2(cl["mano no dominante"])}) porque no sabe cuál firma; la v2 sí distingue la dominante.` }]),
  P([{ t: "Atajos, fugas y proxies: qué buscamos. ", b: true }, { t: "(a) Identidad de la persona a través de la cara: la cara pesa 3 %, sin evidencia. (b) Lateralidad: el 41.6 % de las señas de prueba se hacen con la mano izquierda dominante (persona zurda o video espejado). Su top-1 es 0.898 contra 0.911 y el modelo usa la mano dominante en ambos grupos (0.51 y 0.55), así que el aumento de espejo funciona. (c) Bordes del recorte: en los errores pesan 11.0 % contra 8.2 % en los aciertos. (d) " }, { t: `Las banderas de presencia de mano reciben ${pct(cg["presencia de mano"])}`, b: true }, { t: ", más que la cara y los brazos juntos. Saber cuándo aparece cada mano es en parte legítimo (señas de una o dos manos), pero también es un proxy de la calidad del detector: si MediaPipe pierde la mano un instante, la predicción cambia. En el caso 4 es la parte que más separa GOOD de HAPPY. No encontramos fugas de etiqueta." }]),
  P([{ t: "Errores. ", b: true }, { t: "HOT → BAD (0.98): toda la evidencia está en la mano derecha en los cuadros 16–24, cuando la mano está en la boca y baja. Ambas señas empiezan en la boca y descienden; las distingue la forma de la mano (garra vs. plana), un detalle de dedos que el modelo no captó. GOOD → THANKYOU (0.81, el par más confundido con PLEASE → HAPPY): la atribución está en la mano dominante al final, mientras que la mano no dominante, donde termina GOOD, pesa solo 10 %. Lo corregiría una variable de distancia o contacto entre manos, o más ejemplos de esos pares. La seña SSHH, fuera del vocabulario, se aceptó como WHO (0.98): el índice junto a los labios coincide en forma y ubicación. El " + pct(R.oov_aceptadas) + " de las señas desconocidas supera el umbral, y el " + pct(R.errores_con_conf_sobre_umbral) + " de los errores también." }]),
  P([{ t: "Decisiones que cambian. ", b: true }, { t: "(1) Mantener las banderas de presencia, pero añadir un aumento que apague la detección de una mano por tramos y medir de nuevo su cuota. (2) Agregar la distancia entre manos como variable. (3) Una regla de la app: si la predicción cae en un par confundible conocido (GOOD/THANKYOU, PLEASE/HAPPY, SISTER/BROTHER, HOT/BAD), mostrar ambas con su video de ejemplo. (4) Entrenar una clase «otra seña» con las 450 señas extra, porque el umbral no basta para rechazar señas desconocidas. (5) El umbral se mantiene: el caso 4 (0.669) queda correctamente rechazado." }]),
  P([{ t: "Cómo se muestra al usuario. ", b: true }, { t: "Nada de mapas de calor. Debajo de cada resultado, un botón «¿por qué?» anima el esqueleto del usuario con los puntos que decidieron resaltados en rojo, junto a la plantilla de la seña, y una frase generada del bloque más importante: «me fijé en tu mano derecha cerca de la boca, al final del movimiento». Así el estudiante ve qué parte de su seña se parece a la propuesta y qué corregir." }]),
  P([{ t: "¿Usar el modelo interpretable? ", b: true }, { t: "SeñaBusca es de bajo riesgo (un diccionario de estudio) y la logística no alcanza un rendimiento similar (0.617 vs. 0.906 de top-1), así que se justifica la v2 con explicación a posteriori. En un uso de alto riesgo no usaríamos ninguno de los dos." }]),

  H1("5. Discusión y conclusiones"),
  LI([{ t: "Método. ", b: true }, { t: `Elegimos IG por completitud, resolución y costo, y después de usarlo lo mantendríamos: su AUC de borrado (${f3(A["IG (reposo)"])}) es la mitad que la de la oclusión (${f3(A["Oclusión"])}). Ambas coinciden solo moderadamente (ρ mediana = ${f2(R.spearman_ig_oclusion_mediana)}), porque la oclusión no ve interacciones. Lo que sí cambiaríamos es tratar la línea base como decisión central: con «ceros» o la «media», la fidelidad empeora (${f3(A["IG (ceros)"])} y ${f3(A["IG (media)"])}), aunque en parte por construcción: la curva borra reemplazando con la línea base de reposo, lo que favorece a IG-reposo. La prueba de cordura confirma que IG depende de lo aprendido (ρ = ${f2(R.spearman_cordura_mediana)} con pesos aleatorios).` }]),
  LI([{ t: "Lo que las métricas no mostraban. ", b: true }, { t: "Confirmó lo esperado (mano dominante, tramo central, robustez a la lateralidad), pero reveló la dependencia de las banderas de presencia, el poco uso de la mano no dominante en pares que dependen de ella y que los errores se apoyan más en los extremos del recorte. Conecta con las fallas de segmentación vistas con la cámara." }]),
  LI([{ t: "Costo. ", b: true }, { t: `IG cuesta ${K.ig_ms.toFixed(0)} ms por seña en CPU con un hilo, contra ${K.inferencia_ms.toFixed(1)} ms de la inferencia (≈ 100×). Es viable por predicción solo a demanda (botón «¿por qué?»), y por lote para auditoría: todo el conjunto de prueba con tres líneas base toma unos minutos.` }]),
  LI([{ t: "Riesgos y límites de uso. ", b: true }, { t: "Una explicación convincente puede dar falsa confianza en una respuesta equivocada (caso 2: 0.98 y mal). IG explica el clasificador, no a MediaPipe: los sesgos del detector (tono de piel, iluminación, encuadre) quedan fuera y se ven solo como «presencia de mano». Los datos son de 52 personas sordas en ASL; no cubren LESCO ni la lengua de señas guatemalteca ni variantes regionales. No debe usarse para interpretar en contextos médicos, legales o educativos con calificación, ni para sustituir a intérpretes, y siempre debe ofrecer «ninguna es»." }]),
  LI([{ t: "Conclusión. ", b: true }, { t: "La v2 decide con las señales correctas del dominio, sin atajos de identidad ni de lateralidad. Sus puntos débiles (detección de la mano, contacto entre manos y señas desconocidas) se traducen en cambios concretos de datos, variables y reglas de la app." }]),

  H1("Referencias"),
  ...[
    "[1] Smilkov et al. (2017). SmoothGrad: removing noise by adding noise. arXiv:1706.03825.",
    "[2] Sundararajan, Taly y Yan (2017). Axiomatic attribution for deep networks. ICML.",
    "[3] Lundberg y Lee (2017). A unified approach to interpreting model predictions. NeurIPS.",
    "[4] Ribeiro, Singh y Guestrin (2016). “Why should I trust you?”. KDD.",
    "[5] Zeiler y Fergus (2014). Visualizing and understanding convolutional networks. ECCV.",
    "[6] Bento et al. (2021). TimeSHAP: explaining recurrent models through sequence perturbations. KDD.",
    "[7] Crabbé y van der Schaar (2021). Explaining time series predictions with dynamic masks. ICML.",
    "[8] Leung et al. (2023). Temporal dependencies in feature importance for time series prediction (WinIT). ICLR.",
    "[9] Jain y Wallace (2019). Attention is not explanation. NAACL.",
    "[10] Samek et al. (2017). Evaluating the visualization of what a deep neural network has learned. IEEE TNNLS.",
    "[11] Adebayo et al. (2018). Sanity checks for saliency maps. NeurIPS.",
    "[12] Kokhlikyan et al. (2020). Captum: a unified and generic model interpretability library for PyTorch. arXiv:2009.07896.",
    "[13] Desai et al. (2023). ASL Citizen: a community-sourced dataset for advancing isolated sign language recognition. NeurIPS.",
  ].map(t => new Paragraph({ spacing: { after: 0 }, children: runs(t, 16) })),
  new Paragraph({ spacing: { before: 100 }, children: [
    new TextRun({ text: "Enlace al repositorio: ", bold: true, font: FONT, size: BODY, color: "000000" }),
    new ExternalHyperlink({ link: REPO_URL, children: [new TextRun({ text: REPO_URL, font: FONT, size: BODY, color: "0563C1", underline: {} })] }),
    new TextRun({ text: " (carpeta xai/; cómo ejecutarlo en el README).", font: FONT, size: BODY, color: "000000" }),
  ] }),
];

const doc = new Document({
  creator: "Esteban Cárcamo y Nicolás Concuá", title: "Informe Laboratorio 10 — Explainable AI",
  styles: {
    default: { document: { run: { font: FONT, size: BODY, color: "000000" } } },
    paragraphStyles: [{ id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
      run: { font: FONT, size: TITLE, bold: true, color: "000000" } }],
  },
  numbering: { config: [{ reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•",
    alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 360, hanging: 220 } } } }] }] },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 900, bottom: 900, left: 1080, right: 1080 } } },
    children: [...cover, ...content],
  }],
});
Packer.toBuffer(doc).then(buf => { fs.writeFileSync(OUT, buf); console.log("WROTE", OUT, buf.length, "bytes"); });
