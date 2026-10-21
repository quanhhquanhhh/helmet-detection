const CLASSES = ["motorbike", "person", "helmet"];
const COLORS = ["#ff8c00", "#00aaff", "#00c800"];
const STATUS_COLORS = { violation: "#e62828", ok: "#3cc864", unknown: "#aaaaaa" };
const CONF = [0.5, 0.45, 0.4];
const CAND_CONF = 0.3;
const HELMET_ON_HEAD = 0.3;
const TARGET_MS = 62;
const IOU_NMS = 0.6;
const RIDER_MIN_PX = 80;
const BIG_FRAC = 0.08;
const SMALL_FRAC = 0.05;
const TILE = { size: 640, cols: 3, rows: 2, overlap: 0.15, maxZoom: 1.15, maxAgeMs: 500 };
const LEVELS = [
  { main: 1280, aux: 320, tile: true, name: "1280 + 320 + tiles" },
  { main: 1280, aux: 320, tile: false, name: "1280 + 320" },
  { main: 960, aux: 320, tile: true, name: "960 + 320 + tiles" },
  { main: 960, aux: 320, tile: false, name: "960 + 320" },
  { main: 640, aux: 320, tile: false, name: "640 + 320" },
  { main: 640, aux: 0, tile: false, name: "640" },
  { main: 480, aux: 0, tile: false, name: "480" },
];
const START_LEVEL = 1, WASM_LEVEL = 5;
const SLOW_MS = 95, FAST_MS = 50;

const $ = (id) => document.getElementById(id);
const view = $("view"), ctx = view.getContext("2d");
const video = $("video");
const prep = document.createElement("canvas"), pctx = prep.getContext("2d", { willReadFrequently: true });
const frameC = document.createElement("canvas"), fctx = frameC.getContext("2d");

let session = null, backend = "";
let level = START_LEVEL, benchLevel = START_LEVEL, emaMs = 0, slowCount = 0, fastCount = 0, settle = 3;
let running = false, facing = "environment", stream = null, still = null, wakeLock = null;
let latest = { dets: [], bikes: [] };
let renderTimes = [], inferTimes = [];
const stage = { prep: 0, run: 0 }, shown = { prep: 0, run: 0 }, runCost = {};
const tracker = { tracks: [], nextId: 1, violators: new Set() };
const tileCache = { next: 0, dets: [], ts: [], key: "" };

async function loadModel() {
  $("loadmsg").textContent = "Loading model…";
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;
  const parts = [];
  for (let i = 1; i <= 9; i++) {
    const r = await fetch(`model_part${i}.bin?v=v3c`);
    if (!r.ok) break;
    parts.push(new Uint8Array(await r.arrayBuffer()));
  }
  if (!parts.length) throw new Error("model files not found");
  const bytes = new Uint8Array(parts.reduce((n, x) => n + x.length, 0));
  parts.reduce((o, x) => (bytes.set(x, o), o + x.length), 0);
  for (const ep of ["webgpu", "wasm"]) {
    try {
      if (ep === "webgpu" && !navigator.gpu) continue;
      session = await ort.InferenceSession.create(bytes, { executionProviders: [ep], graphOptimizationLevel: "all" });
      backend = ep; break;
    } catch (e) { console.warn(ep, "failed:", e); }
  }
  if (!session) throw new Error("could not load the model");
  $("backend").textContent = backend.toUpperCase();
  $("loadmsg").textContent = "Preparing the GPU…";
  for (const S of [1280, 960, 640, 480, 320]) {
    const x = new ort.Tensor("float32", new Float32Array(3 * S * S), [1, 3, S, S]);
    await session.run({ [session.inputNames[0]]: x });
    const t = [];
    for (let k = 0; k < 3; k++) { const t0 = performance.now(); await session.run({ [session.inputNames[0]]: x }); t.push(performance.now() - t0); }
    runCost[S] = t.sort((a, b) => a - b)[1];
  }
  const est = (L) => runCost[L.main] + (L.aux ? runCost[L.aux] : 0) + (L.tile ? runCost[TILE.size] : 0);
  const first = LEVELS.findIndex((L, i) => (backend === "webgpu" || i >= WASM_LEVEL) && est(L) <= 0.75 * TARGET_MS);
  level = benchLevel = first >= 0 ? first : LEVELS.length - 1;
}

function preprocess(src, w, h, S, sx = 0, sy = 0) {
  if (prep.width !== S) { prep.width = prep.height = S; }
  const r = Math.min(S / w, S / h);
  const nw = Math.round(w * r), nh = Math.round(h * r), px = Math.floor((S - nw) / 2), py = Math.floor((S - nh) / 2);
  pctx.fillStyle = "rgb(114,114,114)"; pctx.fillRect(0, 0, S, S);
  pctx.drawImage(src, sx, sy, w, h, px, py, nw, nh);
  const d = pctx.getImageData(0, 0, S, S).data, n = S * S, x = new Float32Array(3 * n);
  for (let i = 0, j = 0; i < n; i++, j += 4) { x[i] = d[j] / 255; x[i + n] = d[j + 1] / 255; x[i + 2 * n] = d[j + 2] / 255; }
  return { tensor: new ort.Tensor("float32", x, [1, 3, S, S]), r, px, py };
}

async function infer(src, w, h, S, sx = 0, sy = 0) {
  const t0 = performance.now();
  const m = preprocess(src, w, h, S, sx, sy);
  const t1 = performance.now();
  const o = await session.run({ [session.inputNames[0]]: m.tensor });
  const t2 = performance.now();
  stage.prep += t1 - t0; stage.run += t2 - t1;
  return { out: o[session.outputNames[0]], m };
}

function iou(a, b) {
  const ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])), iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const inter = ix * iy, u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return u > 0 ? inter / u : 0;
}

function decode(out, meta, w, h, minSide) {
  const data = out.data, nc = CLASSES.length, N = out.dims[2], dets = [];
  for (let i = 0; i < N; i++) {
    let best = 0, cls = 0;
    for (let c = 0; c < nc; c++) { const s = data[(4 + c) * N + i]; if (s > best) { best = s; cls = c; } }
    if (best < CAND_CONF) continue;
    const cx = data[i], cy = data[N + i], bw = data[2 * N + i], bh = data[3 * N + i];
    const x1 = Math.min(Math.max((cx - bw / 2 - meta.px) / meta.r, 0), w), y1 = Math.min(Math.max((cy - bh / 2 - meta.py) / meta.r, 0), h);
    const x2 = Math.min(Math.max((cx + bw / 2 - meta.px) / meta.r, 0), w), y2 = Math.min(Math.max((cy + bh / 2 - meta.py) / meta.r, 0), h);
    if (Math.min(x2 - x1, y2 - y1) < minSide) continue;
    dets.push([x1, y1, x2, y2, best, cls]);
  }
  return dets;
}

function nms(cand) {
  cand.sort((a, b) => b[4] - a[4]);
  const keep = [];
  for (const d of cand) {
    if (keep.length >= 300) break;
    if (!keep.some((k) => k[5] === d[5] && iou(k, d) > IOU_NMS)) keep.push(d);
  }
  const dets = keep.filter((d) => d[4] >= CONF[d[5]]);
  dets.weakHelmets = keep.filter((d) => d[5] === 2 && d[4] < CONF[2] && d[4] >= HELMET_ON_HEAD);
  return dets;
}

function tileGrid(w, h) {
  const tw = Math.ceil(w / (TILE.cols - (TILE.cols - 1) * TILE.overlap)), th = Math.ceil(h / (TILE.rows - (TILE.rows - 1) * TILE.overlap));
  if (TILE.size / Math.max(tw, th) > TILE.maxZoom) return [];
  const xs = [...Array(TILE.cols).keys()].map((i) => Math.round(i * (w - tw) / Math.max(1, TILE.cols - 1)));
  const ys = [...Array(TILE.rows).keys()].map((i) => Math.round(i * (h - th) / Math.max(1, TILE.rows - 1)));
  return ys.flatMap((y) => xs.map((x) => [x, y, tw, th]));
}

async function runTile(src, w, h, t) {
  const [x0, y0, tw, th] = t;
  const { out, m } = await infer(src, tw, th, TILE.size, x0, y0);
  return decode(out, m, tw, th, 0).filter((d) =>
    !((d[0] <= 2 && x0 > 0) || (d[1] <= 2 && y0 > 0) || (d[2] >= tw - 2 && x0 + tw < w) || (d[3] >= th - 2 && y0 + th < h)) &&
    Math.max(d[2] - d[0], d[3] - d[1]) < SMALL_FRAC * w).map((d) => [d[0] + x0, d[1] + y0, d[2] + x0, d[3] + y0, d[4], d[5]]);
}

async function detect(src, w, h, allTiles = false) {
  const L = LEVELS[level];
  stage.prep = stage.run = 0;
  let r = await infer(src, w, h, L.main);
  let cand = decode(r.out, r.m, w, h, 0);
  if (L.aux) {
    r = await infer(src, w, h, L.aux);
    cand = cand.concat(decode(r.out, r.m, w, h, BIG_FRAC * w));
  }
  const grid = L.tile ? tileGrid(w, h) : [];
  if (grid.length) {
    const key = `${w}x${h}`;
    if (tileCache.key !== key) { tileCache.key = key; tileCache.next = 0; tileCache.dets = grid.map(() => []); tileCache.ts = grid.map(() => 0); }
    const now = performance.now();
    if (allTiles) {
      for (let i = 0; i < grid.length; i++) { tileCache.dets[i] = await runTile(src, w, h, grid[i]); tileCache.ts[i] = now; }
    } else {
      const i = tileCache.next; tileCache.next = (i + 1) % grid.length;
      tileCache.dets[i] = await runTile(src, w, h, grid[i]); tileCache.ts[i] = now;
    }
    const fresh = allTiles ? tileCache.dets : tileCache.dets.filter((_, i) => now - tileCache.ts[i] <= TILE.maxAgeMs);
    cand = cand.concat(fresh.flat());
  } else if (tileCache.dets.length) {
    tileCache.dets = []; tileCache.key = "";
  }
  shown.prep = stage.prep; shown.run = stage.run;
  return nms(cand);
}

function adapt(ms) {
  if (settle > 0) { settle--; return; }
  emaMs = emaMs ? 0.85 * emaMs + 0.15 * ms : ms;
  if (emaMs > SLOW_MS) { slowCount++; fastCount = 0; } else if (emaMs < FAST_MS) { fastCount++; slowCount = 0; } else { slowCount = fastCount = 0; }
  if (slowCount > 8 && level < LEVELS.length - 1) { level++; slowCount = 0; emaMs = 0; settle = 3; }
  if (fastCount > 30 && level > 0 && (backend === "webgpu" || level > WASM_LEVEL)) { level--; fastCount = 0; emaMs = 0; settle = 3; }
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
  for (const hd of (dets.weakHelmets || [])) {
    const hx = (hd[0] + hd[2]) / 2, hy = (hd[1] + hd[3]) / 2;
    const r = riders.find((r) => {
      if (r.head || !r.person) return false;
      const p = r.person, pw = p[2] - p[0], ph = p[3] - p[1];
      return p[0] <= hx && hx <= p[2] && p[1] - 0.1 * ph <= hy && hy <= p[1] + 0.3 * ph && hd[2] - hd[0] <= 0.8 * pw;
    });
    if (r) { r.head = hd; r.weak = true; }
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

function track(bikes, isVideo) {
  const used = new Set();
  for (const b of bikes) {
    let best = null, bestI = 0.25;
    for (const t of tracker.tracks) { const v = iou(t.raw, b.box); if (!used.has(t) && v > bestI) { best = t; bestI = v; } }
    if (!best) { best = { id: tracker.nextId++, hist: [], missed: 0, smooth: b.box.slice(0, 4) }; tracker.tracks.push(best); }
    used.add(best); best.raw = b.box; best.missed = 0;
    best.smooth = best.smooth.map((v, k) => 0.5 * v + 0.5 * b.box[k]);
    best.hist.push(b.status); if (best.hist.length > 8) best.hist.shift();
    const nv = best.hist.filter((s) => s === "violation").length, no = best.hist.filter((s) => s === "ok").length;
    const recent = best.hist.slice(-3);
    if (!isVideo) b.shown = b.status;
    else if (best.hist.length >= 4 && nv >= Math.ceil(0.6 * best.hist.length) && no <= 1 && !recent.includes("ok")) b.shown = "violation";
    else if (no >= 2 || (no >= 1 && nv === 0)) b.shown = "ok";
    else b.shown = "unknown";
    b.id = best.id; b.drawBox = isVideo ? best.smooth : b.box;
    if (b.shown === "violation") tracker.violators.add(best.id);
  }
  for (const t of tracker.tracks) if (!used.has(t)) t.missed++;
  tracker.tracks = tracker.tracks.filter((t) => t.missed < 10);
}

function label(text, x, y, color, s) {
  ctx.font = `600 ${Math.round(13 * s)}px system-ui`;
  const w = ctx.measureText(text).width + 8 * s, h = 18 * s; y = Math.max(h, y);
  ctx.fillStyle = color; ctx.fillRect(x, y - h, w, h);
  ctx.fillStyle = "#fff"; ctx.fillText(text, x + 4 * s, y - 5 * s);
}

function drawResults({ dets, bikes }) {
  const s = Math.max(1, Math.max(view.width, view.height) / 900);
  ctx.lineWidth = 2 * s;
  for (const d of dets) {
    if (d[5] === 0) continue;
    ctx.strokeStyle = COLORS[d[5]]; ctx.strokeRect(d[0], d[1], d[2] - d[0], d[3] - d[1]);
    if (d[5] === 2) label(`helmet ${d[4].toFixed(2)}`, d[0], d[1] - 2, COLORS[2], s * 0.85);
  }
  ctx.lineWidth = 3 * s;
  for (const b of bikes) {
    const [x1, y1, x2, y2] = b.drawBox || b.box, st = b.shown || b.status, col = STATUS_COLORS[st];
    ctx.strokeStyle = col; ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const tag = { violation: "NO HELMET", ok: "HELMET OK", unknown: "?" }[st];
    label(`#${b.id} ${tag}`, x1, y2 + 20 * s, col, s);
    if (st === "violation") for (const r of b.riders) if (r.person && r.status === "no_helmet") {
      ctx.strokeStyle = STATUS_COLORS.violation; ctx.strokeRect(r.person[0], r.person[1], r.person[2] - r.person[0], r.person[3] - r.person[1]);
    }
  }
}

function rate(ts) { return ts.length > 1 ? 1000 * (ts.length - 1) / (ts[ts.length - 1] - ts[0]) : 0; }

function updateHud() {
  $("fps").textContent = rate(renderTimes).toFixed(0); $("dps").textContent = rate(inferTimes).toFixed(1); $("ms").textContent = emaMs.toFixed(0);
  $("msize").textContent = `auto ${LEVELS[level].name} · prep ${shown.prep.toFixed(0)} · gpu ${shown.run.toFixed(0)} ms`;
  const { dets, bikes } = latest;
  $("nb").textContent = dets.filter((d) => d[5] === 0).length; $("np").textContent = dets.filter((d) => d[5] === 1).length;
  $("nh").textContent = dets.filter((d) => d[5] === 2).length; $("uv").textContent = tracker.violators.size;
  const nv = bikes.filter((b) => (b.shown || b.status) === "violation").length;
  $("viol").textContent = nv ? `${nv} violation${nv > 1 ? "s" : ""} now` : "no violation";
  $("viol").className = nv ? "bad" : "ok";
}

function renderLoop() {
  if (!running) return;
  if (video.readyState >= 2 && video.videoWidth) {
    const w = video.videoWidth, h = video.videoHeight;
    if (view.width !== w || view.height !== h) { view.width = w; view.height = h; }
    ctx.drawImage(video, 0, 0, w, h);
    drawResults(latest);
    renderTimes.push(performance.now()); if (renderTimes.length > 30) renderTimes.shift();
    updateHud();
  }
  requestAnimationFrame(renderLoop);
}

async function inferLoop() {
  while (running && !still) {
    if (video.readyState >= 2 && video.videoWidth) {
      const w = video.videoWidth, h = video.videoHeight;
      if (frameC.width !== w || frameC.height !== h) { frameC.width = w; frameC.height = h; }
      fctx.drawImage(video, 0, 0, w, h);
      try {
        const t0 = performance.now();
        const dets = await detect(frameC, w, h);
        adapt(performance.now() - t0);
        const bikes = composeRiders(dets); track(bikes, true);
        latest = { dets, bikes };
        inferTimes.push(performance.now()); if (inferTimes.length > 20) inferTimes.shift();
      } catch (e) { console.error(e); $("err").textContent = "Error: " + (e && e.message || e); await new Promise((r) => setTimeout(r, 500)); }
    } else {
      await new Promise((r) => setTimeout(r, 30));
    }
  }
}

async function processStill() {
  const w = still.naturalWidth, h = still.naturalHeight;
  view.width = w; view.height = h; frameC.width = w; frameC.height = h;
  fctx.drawImage(still, 0, 0, w, h);
  try {
    level = backend === "webgpu" ? 0 : WASM_LEVEL;
    const t0 = performance.now();
    const dets = await detect(frameC, w, h, true);
    emaMs = performance.now() - t0;
    const bikes = composeRiders(dets); track(bikes, false);
    latest = { dets, bikes };
  } catch (e) { console.error(e); $("err").textContent = "Error: " + (e && e.message || e); }
  ctx.drawImage(frameC, 0, 0); drawResults(latest); updateHud();
}

function showUi() {
  $("status").classList.add("hidden"); $("hud").classList.remove("hidden"); $("controls").classList.remove("hidden");
}

function resetState() {
  level = benchLevel; settle = 3; emaMs = 0; slowCount = fastCount = 0;
  tileCache.dets = []; tileCache.ts = []; tileCache.key = ""; tileCache.next = 0;
  tracker.tracks = []; tracker.violators.clear(); latest = { dets: [], bikes: [] }; renderTimes = []; inferTimes = [];
}

async function startCamera() {
  stopSource(); resetState();
  stream = await navigator.mediaDevices.getUserMedia({ audio: false,
    video: { facingMode: { ideal: facing }, width: { ideal: 1920 }, height: { ideal: 1080 } } });
  video.srcObject = stream; await video.play();
  try { wakeLock = await navigator.wakeLock.request("screen"); } catch (e) { wakeLock = null; }
  still = null; showUi(); running = true; renderLoop(); inferLoop();
}

function stopSource() {
  running = false;
  if (stream) { stream.getTracks().forEach((t) => t.stop()); stream = null; }
  if (wakeLock) { wakeLock.release().catch(() => {}); wakeLock = null; }
  video.pause(); video.removeAttribute("src"); video.srcObject = null;
}

async function openUrl(url, isImage) {
  stopSource(); resetState();
  if (isImage) {
    still = new Image(); still.crossOrigin = "anonymous";
    await new Promise((ok, fail) => { still.onload = ok; still.onerror = () => fail(new Error("cannot load " + url)); still.src = url; });
    showUi(); await processStill();
  } else {
    still = null; video.src = url; video.loop = true; await video.play();
    showUi(); running = true; renderLoop(); inferLoop();
  }
}

$("startBtn").onclick = () => startCamera().catch((e) => alert("Camera error: " + e.message +
  "\nOpen this page over HTTPS (e.g. GitHub Pages) and allow camera access."));
$("flipBtn").onclick = () => { facing = facing === "environment" ? "user" : "environment"; startCamera().catch((e) => alert(e.message)); };
$("stopBtn").onclick = () => { stopSource(); $("status").classList.remove("hidden"); $("hud").classList.add("hidden"); $("controls").classList.add("hidden"); };
$("file").onchange = (e) => {
  const f = e.target.files[0]; if (!f) return;
  openUrl(URL.createObjectURL(f), f.type.startsWith("image/"));
};
document.addEventListener("visibilitychange", async () => {
  if (document.visibilityState === "visible" && stream && !wakeLock) { try { wakeLock = await navigator.wakeLock.request("screen"); } catch (e) {} }
});

(async () => {
  try {
    await loadModel();
    $("loadmsg").textContent = `Model ready (${backend.toUpperCase()}, start level ${LEVELS[level].name}; ` +
      `GPU ms 1280/960/640: ${[1280, 960, 640].map((S) => runCost[S].toFixed(0)).join("/")}). Point the camera at the street.`;
    $("startBtn").classList.remove("hidden"); $("fileLbl").classList.remove("hidden");
    const src = new URLSearchParams(location.search).get("src");
    if (src) openUrl(src, /\.(jpe?g|png|webp)$/i.test(src));
  } catch (e) {
    $("loadmsg").textContent = "Error: " + e.message;
  }
})();
