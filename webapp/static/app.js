/* NORMACORE web console - vanilla JS, no dependencies, works offline.
   Consumes every server message type: state, frame, tool_use, tool_result, image,
   final, error, cancelled, agent_done, estop_ack, cancel_ack, telemetry_error.
   Sends: {type:'chat',text}, {type:'estop'}, {type:'cancel'} plus REST fallbacks. */
"use strict";

const $ = (id) => document.getElementById(id);
const input = $("input"), transcript = $("transcript");

let ws = null, retryDelay = 1000;
let agentActive = false;
let statusInfo = { backend: null, human_confirmed: true, human_window_s: 0 };
let humanRemainingS = -1;   // station-mode countdown mirror of /api/status
let framesThisSecond = 0;

// ---------------------------------------------------------------- transcript
function addMsg(kind, text, who) {
  const div = document.createElement("div");
  div.className = "msg " + kind;
  if (who) {
    const w = document.createElement("span");
    w.className = "who";
    w.textContent = who;
    div.appendChild(w);
  }
  div.appendChild(document.createTextNode(text));
  transcript.appendChild(div);
  transcript.scrollTop = transcript.scrollHeight;
  return div;
}

function addToolUse(name, toolInput) {
  const div = document.createElement("div");
  div.className = "msg";
  const det = document.createElement("details");
  const sum = document.createElement("summary");
  sum.textContent = "TOOL " + name;
  const pre = document.createElement("pre");
  pre.textContent = JSON.stringify(toolInput, null, 1);
  det.appendChild(sum);
  det.appendChild(pre);
  div.appendChild(det);
  transcript.appendChild(div);
  transcript.scrollTop = transcript.scrollHeight;
}

function addToolResult(name, result) {
  let pretty = String(result);
  try { pretty = JSON.stringify(JSON.parse(result), null, 1); } catch (e) { /* plain text */ }
  const div = document.createElement("div");
  div.className = "msg result";
  const w = document.createElement("span");
  w.className = "who";
  w.textContent = name + " ->";
  const pre = document.createElement("pre");
  pre.textContent = pretty;
  div.appendChild(w);
  div.appendChild(pre);
  transcript.appendChild(div);
  transcript.scrollTop = transcript.scrollHeight;
}

function addImage(jpegB64) {
  const div = document.createElement("div");
  div.className = "msg";
  const img = document.createElement("img");
  img.className = "thumb";
  img.alt = "camera snapshot from look";
  img.src = "data:image/jpeg;base64," + jpegB64;
  img.addEventListener("click", () => img.classList.toggle("full"));
  div.appendChild(img);
  transcript.appendChild(div);
  transcript.scrollTop = transcript.scrollHeight;
}

// ---------------------------------------------------------------- telemetry
const canvas = $("joints"), ctx = canvas.getContext("2d");

function drawJoints(stateMsg) {
  const joints = stateMsg.joints && stateMsg.joints.joints;
  if (!joints) return;
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  const rows = 8, rh = H / rows;
  for (let i = 1; i <= rows; i++) {
    const j = joints[String(i)];
    const y = (i - 1) * rh;
    const label = i === 8 ? "GRIP" : "J" + i;
    ctx.fillStyle = "#5c6f84";
    ctx.font = "11px monospace";
    ctx.fillText(label, 0, y + 14);
    const bx = 42, bw = W - 42;
    ctx.strokeStyle = "#16283a";
    ctx.strokeRect(bx, y + 4, bw, 12);
    if (!j) continue;
    const norm = Math.max(0, Math.min(1, j.norm || 0));
    ctx.fillStyle = "#35f0ff";
    ctx.fillRect(bx, y + 4, bw * norm, 12);
    const goal = Math.max(0, Math.min(1, j.goal != null ? j.goal : norm));
    ctx.fillStyle = "#ffb347";
    ctx.fillRect(bx + bw * goal - 1, y + 2, 2, 16);
    ctx.fillStyle = "#5c6f84";
    ctx.font = "10px monospace";
    ctx.fillText(
      norm.toFixed(2) + "  " + (j.current_ma || 0) + "mA " + (j.temp_c || 0) + "C",
      bx, y + 30);
  }
}

function setEstopped(estopped) {
  $("estop-banner").classList.toggle("hidden", !estopped);
  const led = $("estop-led");
  led.className = "led " + (estopped ? "stop" : "ok");
  led.textContent = estopped ? "STOP" : "CLEAR";
}

function onState(msg) {
  drawJoints(msg);
  setEstopped(!!msg.estopped);
  setAgentActive(!!msg.agent_active);
}

function onFrame(msg) {
  $("cam-img").src = "data:image/jpeg;base64," + msg.jpeg_b64;
  $("cam-name").textContent = msg.camera || "--";
  framesThisSecond++;
}

setInterval(() => {
  $("cam-fps").textContent = framesThisSecond + " fps";
  framesThisSecond = 0;
}, 1000);

// ---------------------------------------------------------------- agent state
function setAgentActive(active) {
  agentActive = active;
  $("agent-status").classList.toggle("hidden", !active);
  $("btn-send").classList.toggle("hidden", active);
  $("btn-cancel").classList.toggle("hidden", !active);
  refreshComposerLock();
}

function stationBlocked() {
  return statusInfo.backend === "station" && !statusInfo.human_confirmed;
}

function refreshComposerLock() {
  const blocked = stationBlocked();
  input.disabled = blocked;
  $("btn-send").disabled = blocked || agentActive;
  input.placeholder = blocked
    ? "station mode: confirm human presence first"
    : "task the arm... (Enter to send)";
}

// ---------------------------------------------------------------- websocket
function setConn(state) {  // "on" | "retry" | "off"
  const b = $("badge-conn");
  if (state === "on") { b.className = "badge green"; b.textContent = "LINK UP"; }
  else if (state === "retry") { b.className = "badge amber"; b.textContent = "RETRY"; }
  else { b.className = "badge red"; b.textContent = "OFFLINE"; }
}

const handlers = {
  state: onState,
  frame: onFrame,
  tool_use: (m) => { addToolUse(m.name, m.input); speakTool(m.name, m.input); },
  tool_result: (m) => addToolResult(m.name, m.result),
  image: (m) => addImage(m.jpeg_b64),
  final: (m) => { addMsg("assistant", m.text, "CLAUDE"); speak(m.text); },
  error: (m) => addMsg("error", m.text || "unknown error", "ERR"),
  cancelled: () => addMsg("cancelled", "-- agent cancelled --"),
  agent_done: () => setAgentActive(false),
  estop_ack: () => setEstopped(true),
  cancel_ack: () => addMsg("note", "cancel sent"),
  telemetry_error: (m) => {
    const el = $("cam-err");
    el.textContent = "telemetry: " + (m.text || "error");
    el.classList.remove("hidden");
    clearTimeout(el._t);
    el._t = setTimeout(() => el.classList.add("hidden"), 4000);
  },
};

function connect() {
  const proto = location.protocol === "https:" ? "wss://" : "ws://";
  ws = new WebSocket(proto + location.host + "/ws");
  ws.onopen = () => { retryDelay = 1000; setConn("on"); };
  ws.onmessage = (ev) => {
    let msg;
    try { msg = JSON.parse(ev.data); } catch (e) { return; }
    const h = handlers[msg.type];
    if (h) h(msg);
  };
  ws.onclose = () => {
    setConn("retry");
    setTimeout(connect, retryDelay);
    retryDelay = Math.min(retryDelay * 2, 10000);
  };
  ws.onerror = () => { try { ws.close(); } catch (e) { /* already closed */ } };
}

function wsSend(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
}

// ---------------------------------------------------------------- chat
function sendChat() {
  const text = input.value.trim();
  if (!text || agentActive || stationBlocked()) return;
  addMsg("user", text, "YOU");
  wsSend({ type: "chat", text: text });
  setAgentActive(true);
  input.value = "";
}

$("btn-send").addEventListener("click", sendChat);
input.addEventListener("keydown", (e) => { if (e.key === "Enter") sendChat(); });
$("btn-cancel").addEventListener("click", () => wsSend({ type: "cancel" }));

document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => { input.value = chip.textContent; input.focus(); });
});

// ---------------------------------------------------------------- e-stop
$("btn-estop").addEventListener("click", () => {
  wsSend({ type: "estop" });                       // belt
  fetch("/api/estop", { method: "POST" })          // and braces
    .then(() => setEstopped(true))
    .catch(() => setConn("off"));
});

$("btn-release").addEventListener("click", () => {
  fetch("/api/release", { method: "POST" })
    .then(() => setEstopped(false))
    .catch(() => setConn("off"));
});

// ---------------------------------------------------------------- status + human gate
function fmtCountdown(s) {
  const m = Math.floor(s / 60), r = Math.floor(s % 60);
  return m + ":" + String(r).padStart(2, "0");
}

function applyStatus(d) {
  statusInfo = d;
  $("badge-backend").textContent = "BACKEND " + (d.backend || "--").toUpperCase();
  $("badge-model").textContent = "MODEL " + (d.model || "--");
  $("badge-sim").classList.toggle("hidden", d.backend !== "sim");
  $("badge-key").classList.toggle("hidden", !!d.anthropic_key);
  humanRemainingS = d.backend === "station" ? (d.human_window_s || 0) : -1;
  updateHumanUI();
  refreshComposerLock();
}

function updateHumanUI() {
  const station = statusInfo.backend === "station";
  $("badge-human").classList.toggle("hidden", !station);
  if (!station) { $("modal-human").classList.add("hidden"); return; }
  $("badge-human").textContent = "HUMAN " + fmtCountdown(Math.max(0, humanRemainingS));
  const needConfirm = !statusInfo.human_confirmed || humanRemainingS <= 0;
  $("modal-human").classList.toggle("hidden", !needConfirm);
}

function pollStatus() {
  fetch("/api/status")
    .then((r) => r.json())
    .then(applyStatus)
    .catch(() => setConn("off"));
}

setInterval(() => {  // local 1 s countdown between 5 s polls
  if (humanRemainingS > 0) {
    humanRemainingS -= 1;
    if (humanRemainingS <= 0) { statusInfo.human_confirmed = false; pollStatus(); }
    updateHumanUI();
    refreshComposerLock();
  }
}, 1000);
setInterval(pollStatus, 5000);

$("btn-confirm-human").addEventListener("click", () => {
  fetch("/api/confirm-human", { method: "POST" })
    .then((r) => r.json())
    .then(() => pollStatus())
    .catch(() => setConn("off"));
});

// ---------------------------------------------------------------- voice (zero-GPU v1)
const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
let rec = null, micOn = false, ttsOn = false;

function toolSummary(name, toolInput) {
  const cell = toolInput && toolInput.cell ? " cell " + toolInput.cell : "";
  switch (name) {
    case "pick_cell": return "picking" + cell;
    case "place_cell": return "placing" + cell;
    case "reach_cell": return "reaching" + cell;
    case "home": return "going home";
    case "look": return "looking";
    case "gripper": return "gripper";
    case "arm_state": return "reading arm state";
    case "stop": return "stopping";
    default: return String(name).replace(/_/g, " ");
  }
}

function speak(text) {  // assistant finals; browser-local TTS, no audio leaves the page
  if (!micOn || !ttsOn || !("speechSynthesis" in window) || !text) return;
  const u = new SpeechSynthesisUtterance(text.slice(0, 300));
  window.speechSynthesis.speak(u);
}

function speakTool(name, toolInput) {
  if (!micOn || !ttsOn || !("speechSynthesis" in window)) return;
  window.speechSynthesis.speak(new SpeechSynthesisUtterance(toolSummary(name, toolInput)));
}

function stopTts() { if ("speechSynthesis" in window) window.speechSynthesis.cancel(); }

function setupVoice() {
  const micBtn = $("btn-mic"), ttsBtn = $("btn-tts");
  const hasTts = "speechSynthesis" in window;
  if (!SR) {
    micBtn.disabled = true;
    ttsBtn.disabled = !hasTts ? true : false;
    $("voice-hint").classList.remove("hidden");
    return;
  }
  rec = new SR();
  rec.lang = "en-US";
  rec.interimResults = true;
  rec.continuous = false;
  rec.onresult = (ev) => {
    let interim = "", finalText = "";
    for (let i = ev.resultIndex; i < ev.results.length; i++) {
      if (ev.results[i].isFinal) finalText += ev.results[i][0].transcript;
      else interim += ev.results[i][0].transcript;
    }
    if (interim) input.value = interim;              // live interim in the input
    if (finalText.trim()) {
      input.value = finalText.trim();
      sendChat();                                    // final transcripts auto-send
    }
  };
  rec.onend = () => {
    if (micOn) { try { rec.start(); } catch (e) { /* already started */ } }
  };
  rec.onerror = () => { /* stay silent; typing still works */ };
  micBtn.addEventListener("click", () => {
    micOn = !micOn;
    micBtn.classList.toggle("on", micOn);
    micBtn.textContent = micOn ? "MIC*" : "MIC";
    if (micOn) { try { rec.start(); } catch (e) { /* noop */ } }
    else { try { rec.stop(); } catch (e) { /* noop */ } stopTts(); }
  });
  if (hasTts) {
    ttsBtn.addEventListener("click", () => {
      ttsOn = !ttsOn;
      ttsBtn.classList.toggle("on", ttsOn);
      if (!ttsOn) stopTts();
    });
  } else {
    ttsBtn.disabled = true;
  }
}

// ---------------------------------------------------------------- boot
connect();
pollStatus();
setupVoice();
addMsg("note", "normakit web console - Claude Code in a web for the ElRobot arm. Sim by default.");
