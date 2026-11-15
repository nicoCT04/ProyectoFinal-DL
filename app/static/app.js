// SeñaBusca — lógica del cliente.
// MediaPipe (manos + pose) corre en el navegador; solo se envían los landmarks al servidor,
// nunca el video (privacidad y menos ancho de banda).
// Se usa la copia local de MediaPipe (scripts/00_descargar_mediapipe.sh) si existe; si no, el CDN.
const MP_CDN = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
const MP_LOCAL = "/static/vendor/tasks-vision";
let MP_BASE = MP_LOCAL, vision;
try { vision = await import(`${MP_LOCAL}/vision_bundle.mjs`); }
catch { MP_BASE = MP_CDN; vision = await import(`${MP_CDN}/vision_bundle.mjs`); }
const { FilesetResolver, HandLandmarker, PoseLandmarker } = vision;

const $ = (id) => document.getElementById(id);
const video = $("video"), overlay = $("overlay"), ctx = overlay.getContext("2d");
const SESION = Math.random().toString(36).slice(2, 10);
// ?debug=1: la segmentación reporta sus eventos y miniaturas al servidor (logs/ local, nunca se publica).
const DEBUG = new URLSearchParams(location.search).has("debug");
const thumbCanvas = document.createElement("canvas");
thumbCanvas.width = 160; thumbCanvas.height = 120;
let thumbnails = [], frameCount = 0;

function debugEvent(ev, extra = {}) {
  if (!DEBUG) return;
  fetch("/api/debug", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ev, sesion: SESION, t_video: video.currentTime, estado, conManos, reposo, armado,
                           buffer: buffer.length, ...extra }) });
}
function grabThumbnail() {
  thumbCanvas.getContext("2d").drawImage(video, 0, 0, 160, 120);
  thumbnails.push(thumbCanvas.toDataURL("image/jpeg", 0.6));
}
function sendThumbnails(key) {
  if (!DEBUG || !thumbnails.length) return;
  fetch("/api/debug_frames", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ key, images: thumbnails }) });
  thumbnails = [];
}

// Parámetros de la segmentación automática (cuadros a ~30 fps).
// framesFin 18 (0.6 s) evita partir una seña cuando MediaPipe pierde la mano un instante;
// minConManos 8: en ASL Citizen el 4.3 % de las señas tiene menos de 15 cuadros con mano activa
// (MOTHER, LIKE, HELLO…) y con 8 solo el 0.3 %; los gestos sueltos los atrapa el umbral de confianza.
const CFG = { framesInicio: 4, framesFin: 18, preRoll: 6, minConManos: 8, maxFrames: 180,
              enfriamientoMs: 800, vivoVentana: 60, vivoCadaMs: 450,
              kReposo: 1.3 };   // mano "activa" si la muñeca o el nudillo está por encima de hombros + 1.3·ancho

// Una mano cuenta como activa si está en el espacio de señas (no en reposo sobre la mesa o el regazo).
// Validado en ASL Citizen: conserva el 99.9 % de los cuadros con mano durante las señas.
function limiteReposo(fr) {
  // Nunca por debajo del 90 % de la imagen: con la cámara de laptop los hombros quedan abajo y,
  // sin este tope, cualquier mano visible (incluso en reposo) contaba como activa.
  if (!fr.pose) return 0.85;
  const ls = fr.pose[11], rs = fr.pose[12];
  return Math.min(0.9, (ls[1] + rs[1]) / 2 + CFG.kReposo * Math.hypot(ls[0] - rs[0], ls[1] - rs[1]));
}
function manoActiva(fr) {
  if (!fr.hands.length) return false;
  const lim = limiteReposo(fr);
  return fr.hands.some(h => h[0][1] < lim || h[9][1] < lim);
}

let hand, pose, info, estado = "apagado", buffer = [], previos = [], sinManos = 0, conManos = 0;
let ultimoFin = 0, ultimaPred = null, tMostrado = 0, manualHasta = 0, vivoUltimo = 0, fpsEMA = 0;
let armado = true, reposo = 0, ultimoTiempoVideo = -1;
let medidas = {};

// ------------------------------------------------------------------ inicio
async function init() {
  info = await (await fetch("/api/info")).json();
  const sel = $("modelo");
  for (const [alias, m] of Object.entries(info.modelos)) {
    const o = document.createElement("option");
    o.value = alias; o.textContent = m.desc; if (alias === info.default) o.selected = true;
    sel.appendChild(o);
  }
  $("vocab").innerHTML = info.vocabulario.map(v => `<span>${v.es} · ${v.glosa}</span>`).join("");
  const fs = await FilesetResolver.forVisionTasks(`${MP_BASE}/wasm`);
  // Sin WebGL (aceleración por hardware apagada, algunos navegadores embebidos) el delegado GPU falla: se usa CPU.
  const create = async (delegate) => {
    hand = await HandLandmarker.createFromOptions(fs, {
      baseOptions: { modelAssetPath: "/mediapipe/hand_landmarker.task", delegate },
      runningMode: "VIDEO", numHands: 2, minHandDetectionConfidence: 0.4, minHandPresenceConfidence: 0.4 });
    pose = await PoseLandmarker.createFromOptions(fs, {
      baseOptions: { modelAssetPath: "/mediapipe/pose_landmarker_lite.task", delegate },
      runningMode: "VIDEO", numPoses: 1 });
  };
  try { await create("GPU"); } catch (err) { console.warn("MediaPipe sin GPU, se usa CPU:", err); await create("CPU"); }
  setEstado("espera", "Listo. Enciende la cámara.");
}

$("btnCamara").onclick = async () => {
  const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
  video.removeAttribute("src"); video.onended = null; video.onloadeddata = null;
  video.srcObject = stream; await video.play();
  overlay.width = video.videoWidth; overlay.height = video.videoHeight;
  $("btnCamara").disabled = true; $("btnGrabar").disabled = false;
  estado = "espera"; armado = true; setEstado("espera", "Esperando una seña…");
  if (!corriendo) { corriendo = true; requestAnimationFrame(bucle); }
};

$("btnGrabar").onclick = () => {           // respaldo: graba 4 s sin depender de la detección
  buffer = []; manualHasta = performance.now() + 4000; setEstado("grabando", "Grabando 4 s…");
  estado = "manual";
};

// Respaldo: subir un video ya grabado. Se reproduce en el mismo <video> y pasa por EXACTAMENTE
// el mismo flujo que la cámara (MediaPipe en el navegador + segmentación automática).
$("archivo").onchange = (e) => {
  const f = e.target.files[0]; if (!f) return;
  reproducirVideo(URL.createObjectURL(f));
  e.target.value = "";
};

// El archivo se recorre cuadro por cuadro (a 30 fps) en lugar de reproducirlo en tiempo real:
// así el resultado no depende de la velocidad de la computadora.
let corriendo = false, analizandoArchivo = false;
async function reproducirVideo(url) {
  if (video.srcObject) { video.srcObject.getTracks().forEach(t => t.stop()); video.srcObject = null; }
  $("btnCamara").disabled = false; $("btnGrabar").disabled = true;
  analizandoArchivo = true;
  video.src = url; video.loop = false; video.pause();
  await new Promise(r => video.onloadeddata = r);
  overlay.width = video.videoWidth; overlay.height = video.videoHeight;
  estado = "espera"; buffer = []; previos = []; conManos = 0; sinManos = 0; ultimoFin = -1e9; armado = true;
  setEstado("espera", "Analizando video…");
  const dur = video.duration;
  for (let t = 0; t < dur; t += 1 / 30) {
    video.currentTime = t;
    await new Promise(r => video.onseeked = r);
    const fr = detectar();
    dibujar(fr);
    procesar(fr, sigTs());
    if (estado === "procesando" || estado === "listo") break;
  }
  if (estado === "grabando") await finalizar();
  else if (estado === "espera") setEstado("espera", "No detecté una seña completa en el video.");
  analizandoArchivo = false;
}
window.reproducirVideo = reproducirVideo;   // útil para pruebas y para la demo

// ------------------------------------------------------------------ bucle por cuadro
// MediaPipe exige marcas de tiempo estrictamente crecientes.
let ultimoTs = 0;
function sigTs() { ultimoTs = Math.max(performance.now(), ultimoTs + 1); return ultimoTs; }

function detectar() {
  const ts = sigTs(), t0 = performance.now();
  const rp = pose.detectForVideo(video, ts), rh = hand.detectForVideo(video, ts + 0.5);
  ultimoTs = ts + 0.5;
  const dt = performance.now() - t0; fpsEMA = fpsEMA ? 0.9 * fpsEMA + 0.1 * dt : dt;
  return {
    pose: rp.landmarks?.[0]?.map(p => [p.x, p.y, p.z]) ?? null,
    hands: (rh.landmarks || []).map(h => h.map(p => [p.x, p.y, p.z])),
  };
}

function bucle() {
  // Solo se procesa cuando la cámara entrega un cuadro nuevo (≈ 30 fps, como el dataset).
  if (!analizandoArchivo && video.srcObject && video.readyState >= 2 && video.currentTime !== ultimoTiempoVideo) {
    ultimoTiempoVideo = video.currentTime;
    const fr = detectar();
    dibujar(fr);
    procesar(fr, performance.now());
  }
  requestAnimationFrame(bucle);
}

function procesar(fr, ts) {
  const hay = manoActiva(fr);
  if (estado === "manual") {
    buffer.push(fr);
    if (ts >= manualHasta) finalizar();
    return;
  }
  if ($("envivo").checked) enVivo(fr, ts, hay);

  if (estado === "espera" || estado === "listo") {
    previos.push(fr); if (previos.length > CFG.preRoll) previos.shift();
    conManos = hay ? conManos + 1 : 0;
    // Tras una búsqueda hay que bajar las manos antes de grabar otra (evita grabaciones en bucle).
    reposo = hay ? 0 : reposo + 1;
    if (!armado && reposo >= CFG.framesFin) { armado = true; debugEvent("rearmado"); }
    if (armado && conManos >= CFG.framesInicio && ts - ultimoFin > CFG.enfriamientoMs) {
      estado = "grabando"; buffer = [...previos]; sinManos = 0; thumbnails = [];
      setEstado("grabando", "Grabando… baja las manos al terminar");
      debugEvent("inicio");
    }
  } else if (estado === "grabando") {
    buffer.push(fr);
    if (DEBUG && buffer.length % 4 === 0) grabThumbnail();
    sinManos = hay ? 0 : sinManos + 1;
    if (sinManos >= CFG.framesFin || buffer.length >= CFG.maxFrames) finalizar();
  }
}

async function finalizar() {
  // Se envía solo el tramo activo de la seña (± 3 cuadros), como en el entrenamiento.
  const act = buffer.map(manoActiva);
  const a = Math.max(act.indexOf(true) - 3, 0), b = Math.min(act.lastIndexOf(true) + 4, buffer.length);
  const frames = act.includes(true) ? buffer.slice(a, b) : buffer;
  const muyLarga = buffer.length >= CFG.maxFrames;
  window.__ultimoEnvio = frames;   // para depuración
  buffer = []; sinManos = 0; conManos = 0; ultimoFin = performance.now(); armado = false; reposo = 0;
  const n = frames.filter(manoActiva).length;
  if (n < CFG.minConManos) {
    debugEvent("descartada", { cuadros: frames.length, activos: n, muyLarga });
    sendThumbnails(`descartada_${SESION}_${Date.now()}`);
    estado = "espera"; setEstado("espera", "Muy corto o no vi tus manos. Intenta de nuevo.");
    return;
  }
  estado = "procesando"; setEstado("procesando", "Buscando…");
  const t0 = performance.now();
  const r = await (await fetch("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ frames, modelo: $("modelo").value, sesion: SESION }) })).json();
  medidas = { total: performance.now() - t0, servidor: r.ms_servidor, modelo: r.ms_modelo, cuadros: frames.length };
  debugEvent("enviada", { cuadros: frames.length, activos: n, muyLarga, id_prediccion: r.id_prediccion,
                          top3: (r.candidatos || []).slice(0, 3).map(c => c.glosa), confianza: r.confianza_top3 });
  sendThumbnails(r.id_prediccion || `sin_id_${Date.now()}`);
  mostrar(r);
  estado = "listo"; ultimoFin = performance.now();
  setEstado("listo", muyLarga ? "Grabación muy larga: baja las manos por debajo de la línea al terminar"
                              : "Listo · baja las manos y haz otra seña");
}

async function enVivo(fr, ts, hay) {
  previosVivo.push(fr); if (previosVivo.length > CFG.vivoVentana) previosVivo.shift();
  if (vivoPendiente || !hay || ts - vivoUltimo < CFG.vivoCadaMs ||
      previosVivo.filter(manoActiva).length < 15) return;
  vivoUltimo = ts; vivoPendiente = true;
  let r;
  try {
    r = await (await fetch("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ frames: previosVivo, modelo: $("modelo").value, modo: "en_vivo" }) })).json();
  } finally { vivoPendiente = false; }
  if (!r.ok) return;
  const el = $("vivo"); el.classList.remove("oculto");
  el.textContent = "En vivo: " + r.candidatos.slice(0, 3)
    .map(c => `${c.es} (${Math.round(c.prob * 100)}%)`).join(" · ");
}
const previosVivo = [];
let vivoPendiente = false;
$("envivo").onchange = (e) => { if (!e.target.checked) $("vivo").classList.add("oculto"); };

// ------------------------------------------------------------------ resultados
function mostrar(r) {
  $("vacio").classList.add("oculto"); $("resultado").classList.remove("oculto"); $("gracias").classList.add("oculto");
  window.dispatchEvent(new CustomEvent("senabusca:resultado", { detail: r }));
  if (!r.ok) {
    $("banner").className = "banner no_seguro"; $("banner").textContent = r.motivo;
    $("tarjetas").innerHTML = ""; return;
  }
  ultimaPred = r; tMostrado = performance.now();
  const seguro = r.modo === "seguro";
  $("banner").className = "banner " + r.modo;
  $("banner").textContent = seguro
    ? `Probablemente es una de estas (confianza ${Math.round(r.confianza_top3 * 100)} %).`
    : `No estoy seguro (confianza ${Math.round(r.confianza_top3 * 100)} %). Revisa estas 10 opciones o repite la seña.`;
  pintarTarjetas(seguro ? 3 : 10);
  $("btnMas").classList.toggle("oculto", !seguro);
  const m = medidas;
  $("metricas").textContent =
    `Respuesta: ${m.total?.toFixed(0)} ms (servidor ${m.servidor?.toFixed(1)} ms` +
    (m.modelo !== undefined ? `, modelo ${m.modelo.toFixed(1)} ms` : "") + `) · ` +
    `detector en navegador ≈ ${fpsEMA.toFixed(0)} ms/cuadro · cuadros ${r.cuadros} · modelo ${r.modelo}`;
}

function pintarTarjetas(k) {
  const cont = $("tarjetas"); cont.innerHTML = "";
  ultimaPred.candidatos.slice(0, k).forEach((c, i) => {
    const d = document.createElement("div");
    d.className = "tarjeta" + (i === 0 ? " top" : "") + (k > 3 ? " mini" : "");
    d.innerHTML = (k > 3 ? "" : `<canvas width="192" height="144" title="Ver cómo se hace"></canvas>`) +
      `<div><div class="es">${c.es}</div><div class="glosa">${c.glosa}</div>
        <div class="barra"><div style="width:${(c.prob * 100).toFixed(1)}%"></div></div>
        <div class="prob">${(c.prob * 100).toFixed(1)} %</div></div>
       <button>Es esta</button>`;
    d.querySelector("button").onclick = () => enviarFeedback(c.glosa, i);
    const cv = d.querySelector("canvas"); if (cv) animarEjemplo(cv, c.glosa);
    cont.appendChild(d);
  });
}
$("btnMas").onclick = () => { pintarTarjetas(10); $("btnMas").classList.add("oculto"); };
$("btnNinguna").onclick = () => enviarFeedback(null, null);

async function enviarFeedback(glosa, rank) {
  if (!ultimaPred) return;
  const ms = performance.now() - tMostrado;
  await fetch("/api/feedback", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id_prediccion: ultimaPred.id_prediccion, glosa_correcta: glosa, rank_elegido: rank,
                           ms_hasta_elegir: ms, sesion: SESION }) });
  window.dispatchEvent(new CustomEvent("senabusca:feedback", { detail: { glosa, rank, ms } }));
  $("gracias").classList.remove("oculto");
}

// ------------------------------------------------------------------ dibujo
const MANO = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],
              [9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
const POSE33 = [[11,12],[11,13],[13,15],[12,14],[14,16]];

function dibujar(fr) {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  const W = overlay.width, H = overlay.height;
  ctx.lineWidth = 3;
  if (fr.pose) {
    ctx.strokeStyle = "rgba(255,255,255,.7)";
    for (const [a, b] of POSE33) linea(fr.pose[a], fr.pose[b], W, H);
  }
  // Línea de reposo: por debajo de ella las manos no cuentan como seña.
  const yl = limiteReposo(fr) * H;
  ctx.save(); ctx.setLineDash([10, 8]); ctx.lineWidth = 2; ctx.strokeStyle = "rgba(255,200,80,.85)";
  ctx.beginPath(); ctx.moveTo(0, yl); ctx.lineTo(W, yl); ctx.stroke(); ctx.restore();
  // Mano activa en verde, mano en reposo en gris.
  for (const h of fr.hands) {
    ctx.strokeStyle = (h[0][1] < yl / H || h[9][1] < yl / H) ? "#5fcf8e" : "rgba(180,180,180,.8)";
    for (const [a, b] of MANO) linea(h[a], h[b], W, H);
  }
}
function linea(p, q, W, H) { ctx.beginPath(); ctx.moveTo(p[0] * W, p[1] * H); ctx.lineTo(q[0] * W, q[1] * H); ctx.stroke(); }

// Plantilla: 64 cuadros x 53 puntos (normalizados: centro en hombros, unidad = ancho de hombros).
const cachePlantillas = {};
async function animarEjemplo(cv, glosa) {
  let fr = cachePlantillas[glosa];
  if (fr === undefined) {
    const r = await fetch(`/api/ejemplo/${encodeURIComponent(glosa)}`);
    fr = cachePlantillas[glosa] = r.ok ? (await r.json()).frames : null;
  }
  if (!fr) { cv.style.display = "none"; return; }
  const c = cv.getContext("2d"), W = cv.width, H = cv.height;
  const css = getComputedStyle(document.documentElement);
  const colorPose = css.getPropertyValue("--suave").trim(), colorMano = css.getPropertyValue("--acento").trim();
  const P = (p) => [W / 2 - p[0] * W * 0.23, H * 0.58 + p[1] * W * 0.23];   // en espejo, como la cámara
  const POSE = [[5,6],[5,7],[7,9],[6,8],[8,10]];
  let t = 0;
  const paso = () => {
    if (!cv.isConnected) return;
    const f = fr[Math.floor(t) % fr.length];
    c.clearRect(0, 0, W, H); c.lineWidth = 2.5;
    c.strokeStyle = colorPose;
    for (const [a, b] of POSE) seg(c, P(f[a]), P(f[b]));
    c.beginPath(); const n = P(f[0]); c.arc(n[0], n[1], W * 0.05, 0, 7); c.stroke();
    c.strokeStyle = colorMano;
    for (const off of [11, 32]) for (const [a, b] of MANO) seg(c, P(f[off + a]), P(f[off + b]));
    t += 0.5; requestAnimationFrame(paso);
  };
  paso();
}
function seg(c, p, q) { c.beginPath(); c.moveTo(p[0], p[1]); c.lineTo(q[0], q[1]); c.stroke(); }

function setEstado(cls, txt) { const e = $("estado"); e.className = "estado " + cls; e.textContent = txt; }

init().catch(err => setEstado("espera", "Error al iniciar: " + err));
if (new URLSearchParams(location.search).has("prueba")) import("/static/prueba.js?v=4");
