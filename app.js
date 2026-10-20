const CLASSES = ["motorbike", "person", "helmet"];
const COLORS = ["#ff8c00", "#00aaff", "#00c800"];
const STATUS_COLORS = { violation: "#e62828", ok: "#3cc864", unknown: "#aaaaaa" };
const RIDER_MIN_PX = 80;
const IOU_NMS = 0.6;

const $ = (id) => document.getElementById(id);
const view = $("view"), ctx = view.getContext("2d");
const video = $("video");
const prep = document.createElement("canvas"), pctx = prep.getContext("2d", { willReadFrequently: true });

let session = null, backend = "", modelSize = 640, conf = 0.3;
let running = false, facing = "environment", stream = null, still = null;
let fpsHist = [], msHist = [], lastT = performance.now();
const tracker = { tracks: [], nextId: 1, violators: new Set() };

async function loadModel(size) {
  $("loadmsg").textContent = `Loading model (${size} px)…`;
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;
  const parts = [];
  for (let i = 1; i <= 9; i++) {
    const r = await fetch(`model_${size}_part${i}.bin`);
    if (!r.ok) break;
    parts.push(new Uint8Array(await r.arrayBuffer()));
  }
  if (!parts.length) throw new Error("model files not found");
  const url = new Uint8Array(parts.reduce((n, x) => n + x.length, 0));
  parts.reduce((o, x) => (url.set(x, o), o + x.length), 0);
  for (const ep of ["webgpu", "wasm"]) {
    try {
      if (ep === "webgpu" && !navigator.gpu) continue;
      session = await ort.InferenceSession.create(url, { executionProviders: [ep], graphOptimizationLevel: "all" });
      backend = ep; break;
    } catch (e) { console.warn(ep, "failed:", e); }
  }
  if (!session) throw new Error("could not load the model");
  modelSize = size;
  prep.width = prep.height = size;
  $("backend").textContent = backend.toUpperCase();
  $("msize").textContent = `${size} px`;
  $("sizeBtn").textContent = size === 640 ? "Mode: fast (640)" : "Mode: accurate (960)";
}

function preprocess(src, w, h) {
  const S = modelSize, r = Math.min(S / w, S / h);
  const nw = Math.round(w * r), nh = Math.round(h * r), px = Math.floor((S - nw) / 2), py = Math.floor((S - nh) / 2);
  pctx.fillStyle = "rgb(114,114,114)"; pctx.fillRect(0, 0, S, S);
  pctx.drawImage(src, 0, 0, w, h, px, py, nw, nh);
  const d = pctx.getImageData(0, 0, S, S).data, n = S * S, x = new Float32Array(3 * n);
  for (let i = 0, j = 0; i < n; i++, j += 4) { x[i] = d[j] / 255; x[i + n] = d[j + 1] / 255; x[i + 2 * n] = d[j + 2] / 255; }
  return { tensor: new ort.Tensor("float32", x, [1, 3, S, S]), r, px, py };
}

function iou(a, b) {
  const ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])), iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const inter = ix * iy, u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return u > 0 ? inter / u : 0;
}

function postprocess(out, meta, w, h) {
  const data = out.data, nc = CLASSES.length, N = out.dims[2], cand = [];
  for (let i = 0; i < N; i++) {
    let best = 0, cls = 0;
    for (let c = 0; c < nc; c++) { const s = data[(4 + c) * N + i]; if (s > best) { best = s; cls = c; } }
    if (best < conf) continue;
    const cx = data[i], cy = data[N + i], bw = data[2 * N + i], bh = data[3 * N + i];
    const x1 = Math.min(Math.max((cx - bw / 2 - meta.px) / meta.r, 0), w), y1 = Math.min(Math.max((cy - bh / 2 - meta.py) / meta.r, 0), h);
    const x2 = Math.min(Math.max((cx + bw / 2 - meta.px) / meta.r, 0), w), y2 = Math.min(Math.max((cy + bh / 2 - meta.py) / meta.r, 0), h);
    cand.push([x1, y1, x2, y2, best, cls]);
  }
  cand.sort((a, b) => b[4] - a[4]);
  const keep = [];
  for (const d of cand) {
    if (keep.length >= 300) break;
    if (!keep.some((k) => k[5] === d[5] && iou(k, d) > IOU_NMS)) keep.push(d);
  }
  return keep;
}

function composeRiders(dets) {
  const bikes = dets.filter((d) => d[5] === 0), persons = dets.filter((d) => d[5] === 1);
  const heads = dets.filter((d) => d[5] === 2).sort((a, b) => b[4] - a[4]);
  const out = bikes.map((b) => ({ box: b, riders: [], status: "unknown" }));
  const riders = [];
  for (const p of persons) {
    let best = -1, bestS = 0; const pw = p[2] - p[0];
    bikes.forEach((b, bi) => {
      const bh = b[3] - b[1];
      const xov = Math.max(0, Math.min(p[2], b[2]) - Math.max(p[0], b[0])) / Math.max(pw, 1e-6);
      if (xov < 0.3 || p[1] > b[1] + 0.5 * bh || p[3] < b[1] + 0.1 * bh || p[3] > b[3] + 0.3 * bh) return;
      const yov = Math.max(0, Math.min(p[3], b[3]) - Math.max(p[1], b[1])) / Math.max(bh, 1e-6);
      if (xov + yov > bestS) { best = bi; bestS = xov + yov; }
    });
    if (best >= 0) riders.push({ person: p, bike: best, head: null });
  }
  const unassigned = [];
  for (const hd of heads) {
    const hx = (hd[0] + hd[2]) / 2, hy = (hd[1] + hd[3]) / 2; let best = -1, bestD = 1e9;
    riders.forEach((r, ri) => {
      const p = r.person, pw = p[2] - p[0], ph = p[3] - p[1];
      if (r.head === null && p[0] - 0.1 * pw <= hx && hx <= p[2] + 0.1 * pw && p[1] - 0.15 * ph <= hy && hy <= p[1] + 0.45 * ph) {
        const d = Math.abs(hx - (p[0] + p[2]) / 2) / pw + Math.abs(hy - p[1]) / ph;
        if (d < bestD) { best = ri; bestD = d; }
      }
    });
    if (best >= 0) riders[best].head = hd; else unassigned.push(hd);
  }
  const pedestrians = persons.filter((p) => !riders.some((r) => r.person === p));
  for (const hd of unassigned) {
    const hx = (hd[0] + hd[2]) / 2, hy = (hd[1] + hd[3]) / 2, hw = hd[2] - hd[0];
    if (pedestrians.some((p) => p[0] <= hx && hx <= p[2] && p[1] <= hy && hy <= (p[1] + p[3]) / 2)) continue;
    let best = -1, bestD = 1e9;
    bikes.forEach((b, bi) => {
      const bw = b[2] - b[0], bh = b[3] - b[1];
      if (b[0] - 0.1 * bw <= hx && hx <= b[2] + 0.1 * bw && b[1] - 1.5 * bh <= hy && hy <= b[1] + 0.4 * bh && hw < bw) {
        const d = Math.abs(hx - (b[0] + b[2]) / 2) / bw + Math.abs(hy - b[1]) / bh;
        if (d < bestD) { best = bi; bestD = d; }
      }
    });
    if (best >= 0) {
      const free = riders.find((r) => r.bike === best && r.head === null);
      if (free) free.head = hd; else riders.push({ person: null, bike: best, head: hd });
    }
  }
  for (const r of riders) {
    r.status = r.head ? "helmet" : (r.person && r.person[3] - r.person[1] >= RIDER_MIN_PX ? "no_helmet" : "unknown");
    out[r.bike].riders.push(r);
  }
  for (const b of out) {
    const st = b.riders.map((r) => r.status);
    b.status = st.includes("no_helmet") ? "violation" : (st.length && st.every((s) => s === "helmet") ? "ok" : "unknown");
  }
  return out;
}

function track(bikes) {
  const used = new Set();
  for (const b of bikes) {
    let best = null, bestI = 0.3;
    for (const t of tracker.tracks) if (!used.has(t) && iou(t.box, b.box) > bestI) { best = t; bestI = iou(t.box, b.box); }
    if (!best) { best = { id: tracker.nextId++, votes: [], missed: 0 }; tracker.tracks.push(best); }
    used.add(best); best.box = b.box; best.missed = 0; b.id = best.id;
    best.votes.push(b.status === "violation"); if (best.votes.length > 8) best.votes.shift();
    if (best.votes.filter(Boolean).length >= 3) tracker.violators.add(best.id);
  }
  for (const t of tracker.tracks) if (!used.has(t)) t.missed++;
  tracker.tracks = tracker.tracks.filter((t) => t.missed < 15);
}

function label(text, x, y, color, s) {
  ctx.font = `600 ${Math.round(13 * s)}px system-ui`;
  const w = ctx.measureText(text).width + 8 * s, h = 18 * s; y = Math.max(h, y);
  ctx.fillStyle = color; ctx.fillRect(x, y - h, w, h);
  ctx.fillStyle = "#fff"; ctx.fillText(text, x + 4 * s, y - 5 * s);
}

function drawResults(dets, bikes) {
  const s = Math.max(1, Math.max(view.width, view.height) / 900);
  ctx.lineWidth = 2 * s;
  for (const d of dets) {
    if (d[5] === 0) continue;
    ctx.strokeStyle = COLORS[d[5]]; ctx.strokeRect(d[0], d[1], d[2] - d[0], d[3] - d[1]);
    if (d[5] === 2) label(`helmet ${d[4].toFixed(2)}`, d[0], d[1] - 2, COLORS[2], s * 0.85);
  }
  ctx.lineWidth = 3 * s;
  for (const b of bikes) {
    const [x1, y1, x2, y2] = b.box, col = STATUS_COLORS[b.status];
    ctx.strokeStyle = col; ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const n = b.riders.filter((r) => r.status === "no_helmet").length;
    const tag = { violation: `NO HELMET x${n}`, ok: "HELMET OK", unknown: "?" }[b.status];
    label(`#${b.id} ${tag}`, x1, y2 + 20 * s, col, s);
    for (const r of b.riders) if (r.person && r.status === "no_helmet") {
      ctx.strokeStyle = STATUS_COLORS.violation; ctx.strokeRect(r.person[0], r.person[1], r.person[2] - r.person[0], r.person[3] - r.person[1]);
    }
  }
}

function updateHud(dets, bikes, ms) {
  const now = performance.now(); fpsHist.push(1000 / Math.max(now - lastT, 1)); lastT = now; msHist.push(ms);
  if (fpsHist.length > 20) { fpsHist.shift(); msHist.shift(); }
  const avg = (a) => a.reduce((x, y) => x + y, 0) / a.length;
  $("fps").textContent = avg(fpsHist).toFixed(1); $("ms").textContent = avg(msHist).toFixed(0);
  $("nb").textContent = dets.filter((d) => d[5] === 0).length; $("np").textContent = dets.filter((d) => d[5] === 1).length;
  $("nh").textContent = dets.filter((d) => d[5] === 2).length; $("uv").textContent = tracker.violators.size;
  const nv = bikes.filter((b) => b.status === "violation").length;
  $("viol").textContent = nv ? `${nv} violation${nv > 1 ? "s" : ""} now` : "no violation";
  $("viol").className = nv ? "bad" : "ok";
}

const frameC = document.createElement("canvas"), fctx = frameC.getContext("2d");

async function processFrame(src, w, h) {
  if (view.width !== w || view.height !== h) { view.width = w; view.height = h; }
  if (frameC.width !== w || frameC.height !== h) { frameC.width = w; frameC.height = h; }
  fctx.drawImage(src, 0, 0, w, h);
  const t0 = performance.now();
  const meta = preprocess(frameC, w, h);
  const res = await session.run({ [session.inputNames[0]]: meta.tensor });
  const dets = postprocess(res[session.outputNames[0]], meta, w, h);
  const ms = performance.now() - t0;
  const bikes = composeRiders(dets); track(bikes);
  ctx.drawImage(frameC, 0, 0);
  drawResults(dets, bikes); updateHud(dets, bikes, ms);
}

async function loop() {
  if (!running) return;
  try {
    if (still) await processFrame(still, still.naturalWidth, still.naturalHeight);
    else if (video.readyState >= 2 && video.videoWidth) await processFrame(video, video.videoWidth, video.videoHeight);
  } catch (e) { console.error(e); window.lastError = String(e && e.stack || e); $("err").textContent = "Error: " + (e && e.message || e); }
  if (still) { running = false; return; }
  requestAnimationFrame(loop);
}

function showUi() {
  $("status").classList.add("hidden"); $("hud").classList.remove("hidden"); $("controls").classList.remove("hidden");
}

async function startCamera() {
  stopSource();
  stream = await navigator.mediaDevices.getUserMedia({ audio: false,
    video: { facingMode: { ideal: facing }, width: { ideal: 1280 }, height: { ideal: 720 } } });
  video.srcObject = stream; await video.play();
  still = null; showUi(); running = true; loop();
}

function stopSource() {
  running = false;
  if (stream) { stream.getTracks().forEach((t) => t.stop()); stream = null; }
  video.pause(); video.removeAttribute("src"); video.srcObject = null;
}

async function openUrl(url, isImage) {
  stopSource(); tracker.tracks = []; tracker.violators.clear();
  if (isImage) {
    still = new Image(); still.crossOrigin = "anonymous";
    await new Promise((ok, fail) => { still.onload = ok; still.onerror = () => fail(new Error("cannot load " + url)); still.src = url; });
  } else {
    still = null; video.src = url; video.loop = true; await video.play();
  }
  showUi(); running = true; loop();
}

$("startBtn").onclick = () => startCamera().catch((e) => alert("Camera error: " + e.message +
  "\nOpen this page over HTTPS (e.g. GitHub Pages) and allow camera access."));
$("flipBtn").onclick = () => { facing = facing === "environment" ? "user" : "environment"; startCamera().catch((e) => alert(e.message)); };
$("stopBtn").onclick = () => { stopSource(); $("status").classList.remove("hidden"); $("hud").classList.add("hidden"); $("controls").classList.add("hidden"); };
$("conf").oninput = (e) => { conf = parseFloat(e.target.value); $("confv").textContent = conf.toFixed(2); if (still && !running) { running = true; loop(); } };
$("sizeBtn").onclick = async () => {
  const wasRunning = running; running = false;
  await loadModel(modelSize === 640 ? 960 : 640);
  if (wasRunning || still) { running = true; loop(); }
};
$("file").onchange = (e) => {
  const f = e.target.files[0]; if (!f) return;
  openUrl(URL.createObjectURL(f), f.type.startsWith("image/"));
};

(async () => {
  try {
    await loadModel(640);
    $("loadmsg").textContent = `Model ready (${backend.toUpperCase()}). Point the camera at the street.`;
    $("startBtn").classList.remove("hidden"); $("fileLbl").classList.remove("hidden");
    const src = new URLSearchParams(location.search).get("src");
    if (src) openUrl(src, /\.(jpe?g|png|webp)$/i.test(src));
  } catch (e) {
    $("loadmsg").textContent = "Error: " + e.message;
  }
})();
