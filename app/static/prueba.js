// Sesión de prueba guiada (docs/PROTOCOLO_PRUEBA_USUARIOS.md).
//   /?prueba=usuarios  tarea A (búsqueda manual) vs. tarea B (SeñaBusca) + preguntas finales
//   /?prueba=propia    validación propia: recorre señas del vocabulario mostrando cómo se hacen
// Todo se guarda en logs/prueba_usuarios.jsonl; lo resume scripts/12_resumen_prueba_usuarios.py.
const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const MODE = params.get("prueba") === "propia" ? "propia" : "usuarios";
const USER_SIGNS = ["HELLO", "THANKYOU", "SCHOOL", "UNDERSTAND", "BATHROOM", "TOMORROW"];
const LIMIT_S = 300;

let info, participant, trials = [], current = -1, trial = null, timer = null;

const panel = document.createElement("aside");
panel.id = "pruebaPanel";
panel.className = "prueba-panel";
document.body.appendChild(panel);
document.body.classList.add("con-prueba");

function post(event) {
  return fetch("/api/prueba", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ modo: MODE, participante: participant, ...event }) });
}
function spanish(gloss) { return info.vocabulario.find(v => v.glosa === gloss)?.es ?? ""; }
function rotate(list, n) { const k = n % list.length; return [...list.slice(k), ...list.slice(0, k)]; }
function shuffle(list) {
  const a = [...list];
  for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
  return a;
}
function seconds() { return trial ? (performance.now() - trial.t0) / 1000 : 0; }
function startTimer(onTick) {
  clearInterval(timer);
  timer = setInterval(() => {
    $("pruebaReloj").textContent = `${Math.floor(seconds())} s`;
    onTick?.();
  }, 250);
}
function videoTag(gloss, k) {
  return `<video class="prueba-video" src="/api/prueba/video/${encodeURIComponent(gloss)}/${k}"
                 autoplay loop muted playsinline></video>`;
}
function render(html) { panel.innerHTML = html; }

async function setup() {
  info = await (await fetch("/api/info")).json();
  const next = (await (await fetch(`/api/prueba/participantes?modo=${MODE}`)).json()).siguiente;
  render(MODE === "usuarios" ? `
    <h2>Prueba con usuarios</h2>
    <p>Participante <b>P${next}</b>. Unos 15 minutos: 3 señas buscando a mano en un diccionario en
       línea y 3 con SeñaBusca, y 5 preguntas al final. No se guarda video ni nombres.</p>
    <label class="prueba-check"><input type="checkbox" id="pruebaConsent">
      La persona aceptó participar (consentimiento verbal).</label>
    <button class="primario" id="pruebaEmpezar" disabled>Empezar</button>` : `
    <h2>Validación propia</h2>
    <p>Se muestran señas del vocabulario con un video de ejemplo. Apréndela, hazla frente a la
       cámara y toca la opción correcta (o «Ninguna es»). Cada intento queda etiquetado.</p>
    <label>Cuántas señas <input type="number" id="pruebaN" value="${params.get("n") || 20}" min="3" max="50"></label>
    <button class="primario" id="pruebaEmpezar">Empezar</button>`);
  if (MODE === "usuarios") $("pruebaConsent").onchange = (e) => { $("pruebaEmpezar").disabled = !e.target.checked; };
  $("pruebaEmpezar").onclick = () => begin(next);
}

function begin(n) {
  participant = `P${n}`;
  if (MODE === "usuarios") {
    const signs = rotate(USER_SIGNS, n - 1);
    const manual = signs.slice(0, 3).map(g => ({ kind: "manual", gloss: g }));
    const app = signs.slice(3).map(g => ({ kind: "app", gloss: g }));
    trials = n % 2 === 1 ? [...manual, ...app] : [...app, ...manual];
    trials.forEach(t => { t.video = n; });
  } else {
    const k = Math.max(3, Math.min(50, +$("pruebaN").value || 20));
    trials = shuffle(info.vocabulario.map(v => v.glosa)).slice(0, k).map((g, i) => ({ kind: "app", gloss: g, video: i }));
  }
  post({ tipo: "inicio", orden: trials.map(t => `${t.kind}:${t.gloss}`), modelo: $("modelo").value });
  if (!$("btnCamara").disabled) $("btnCamara").click();
  nextTrial();
}

function header() {
  const label = MODE === "propia" ? "Validación propia"
    : trial.kind === "manual" ? "Tarea A · búsqueda manual" : "Tarea B · con SeñaBusca";
  return `<div class="prueba-paso">${participant} · ${current + 1} de ${trials.length} · ${label}
          <span id="pruebaReloj" class="prueba-reloj"></span></div>`;
}

function nextTrial() {
  clearInterval(timer);
  current += 1;
  if (current >= trials.length) return MODE === "usuarios" ? survey() : ownSummary();
  trial = { ...trials[current], attempts: [], t0: null };
  trial.kind === "manual" ? manualIntro() : appIntro();
}

// ---------------------------------------------------------------- tarea A
function manualIntro() {
  render(`${header()}${videoTag(trial.gloss, trial.video)}
    <p>Mira esta seña. Búscala en el diccionario en línea que quieras, en otra pestaña (por ejemplo
       <b>handspeak.com</b>, <b>lifeprint.com</b> o <b>signasl.org</b>). Tienes ${LIMIT_S / 60} minutos.</p>
    <button class="primario" id="pruebaIr">Empezar a buscar</button>`);
  $("pruebaIr").onclick = () => {
    trial.t0 = performance.now();
    render(`${header()}${videoTag(trial.gloss, trial.video)}
      <p>Buscando… cuando sepas qué significa, escríbelo.</p>
      <input id="pruebaRespuesta" placeholder="¿Qué significa?" autocomplete="off">
      <div class="prueba-botones"><button class="primario" id="pruebaEncontre">La encontré</button>
      <button id="pruebaRindo">Me rindo</button></div>`);
    startTimer(() => { if (seconds() >= LIMIT_S) manualEnd(null, true); });
    $("pruebaEncontre").onclick = () => manualEnd($("pruebaRespuesta").value.trim() || "(sin texto)", false);
    $("pruebaRindo").onclick = () => manualEnd(null, false);
  };
}

function manualEnd(answer, timedOut) {
  clearInterval(timer);
  const t = seconds();
  const save = (found) => {
    post({ tipo: "ensayo", tarea: "manual", glosa: trial.gloss, t_s: t, respuesta: answer,
           encontrada: found, tiempo_agotado: timedOut });
    nextTrial();
  };
  if (answer === null) return save(false);
  render(`${header()}<p>Respuesta: <b>${answer}</b></p>
    <p>Era <b>${spanish(trial.gloss)}</b> (${trial.gloss}). ¿Coincide?</p>
    <div class="prueba-botones"><button class="primario" id="pruebaSi">Sí, correcta</button>
    <button id="pruebaNo">No</button></div>`);
  $("pruebaSi").onclick = () => save(true);
  $("pruebaNo").onclick = () => save(false);
}

// ---------------------------------------------------------------- tarea B
function appIntro() {
  const own = MODE === "propia";
  render(`${header()}${videoTag(trial.gloss, trial.video)}
    ${own ? `<p class="prueba-glosa">${spanish(trial.gloss)} · ${trial.gloss}</p>` : ""}
    <p>${own ? "Aprende la seña con el video." : "Mira esta seña."} Cuando estés listo, presiona
       «Listo» y hazla frente a la cámara: empieza y termina con las manos abajo. Luego toca
       <b>«Es esta»</b> en la opción correcta o <b>«Ninguna es»</b>.</p>
    <button class="primario" id="pruebaIr">Listo</button>`);
  $("pruebaIr").onclick = () => {
    trial.t0 = performance.now();
    $("resultado").classList.add("oculto"); $("vacio").classList.remove("oculto");
    panel.classList.add("compacto");
    render(`${header()}${videoTag(trial.gloss, trial.video)}
      <p>Haz la seña frente a la cámara. Si sale mal, repítela.</p>
      <button id="pruebaSaltar">No funcionó, saltar</button>`);
    startTimer(() => { if (seconds() >= LIMIT_S) appEnd(null); });
    $("pruebaSaltar").onclick = () => appEnd(null);
  };
}

function appEnd(choice) {
  if (!trial || trial.kind !== "app" || trial.t0 === null) return;
  clearInterval(timer);
  panel.classList.remove("compacto");
  const last = trial.attempts.at(-1);
  post({ tipo: "ensayo", tarea: "app", glosa: trial.gloss, t_s: seconds(), intentos: trial.attempts.length,
         glosa_elegida: choice?.glosa ?? null, rank_elegido: choice?.rank ?? null,
         ms_hasta_elegir: choice?.ms ?? null, correcta: choice?.glosa === trial.gloss,
         t_primer_resultado_s: trial.attempts[0]?.t_s ?? null, t_ultimo_resultado_s: last?.t_s ?? null,
         modo_app: last?.modo ?? null, confianza: last?.confianza ?? null,
         top3: last?.top3 ?? null, id_prediccion: last?.id ?? null, modelo: $("modelo").value });
  trial = null;
  setTimeout(nextTrial, choice ? 900 : 0);
}

window.addEventListener("senabusca:resultado", (e) => {
  if (!trial || trial.kind !== "app" || trial.t0 === null) return;
  const r = e.detail;
  trial.attempts.push({ t_s: seconds(), ok: r.ok, modo: r.modo, confianza: r.confianza_top3, id: r.id_prediccion,
                        top3: (r.candidatos || []).slice(0, 3).map(c => c.glosa) });
});
window.addEventListener("senabusca:feedback", (e) => { if (trial?.attempts.length) appEnd(e.detail); });

// ---------------------------------------------------------------- cierre
function scale(id, text) {
  return `<div class="prueba-escala" data-id="${id}"><span>${text}</span><div>${[1, 2, 3, 4, 5]
    .map(v => `<button data-v="${v}">${v}</button>`).join("")}</div></div>`;
}

function survey() {
  render(`<h2>Preguntas finales</h2><p class="prueba-nota">1 = nada · 5 = mucho</p>
    ${scale("facil_manual", "¿Qué tan fácil fue encontrar la seña buscando a mano?")}
    ${scale("facil_app", "¿Qué tan fácil fue con SeñaBusca?")}
    ${scale("confianza", "¿Confiarías en la respuesta de la app sin revisar el ejemplo?")}
    ${scale("usaria", "¿La usarías en un curso de ASL?")}
    <label>¿Cuánto pagarías al mes? (USD, 0 = nada) <input type="number" id="pruebaPago" min="0" step="0.5"></label>
    <textarea id="pruebaComentario" placeholder="Comentario libre (opcional)"></textarea>
    <button class="primario" id="pruebaGuardar">Guardar</button>`);
  const answers = {};
  panel.querySelectorAll(".prueba-escala").forEach(row => row.querySelectorAll("button").forEach(b => {
    b.onclick = () => {
      answers[row.dataset.id] = +b.dataset.v;
      row.querySelectorAll("button").forEach(x => x.classList.toggle("elegido", x === b));
    };
  }));
  $("pruebaGuardar").onclick = async () => {
    const pay = $("pruebaPago").value;
    await post({ tipo: "encuesta", ...answers, pagaria_usd_mes: pay === "" ? null : +pay,
                 comentario: $("pruebaComentario").value.trim() || null });
    render(`<h2>¡Gracias!</h2><p>${participant} quedó registrado.</p>
      <button class="primario" onclick="location.reload()">Siguiente participante</button>`);
  };
}

async function ownSummary() {
  render(`<h2>Listo</h2><p>Se registraron ${trials.length} señas. Resumen con
    <code>.venv/bin/python scripts/12_resumen_prueba_usuarios.py</code>.</p>
    <button class="primario" onclick="location.reload()">Otra ronda</button>`);
}

setup();
