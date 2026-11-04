const CLASSES = ["motorbike", "person", "helmet", "no_helmet"];
const COLORS = ["#f2f4f6", "#4ea1ff", "#2fb36d", "#ff8a3d"];
const STATUS_COLORS = { violation: "#e5383b", ok: "#2fb36d", unknown: "#c3cad2" };
const FONT = "'Barlow Semi Condensed', Barlow, -apple-system, system-ui, sans-serif";
// motorbike, person as v8; helmet 0.25 and no_helmet 0.35 from the Hanoi val frames (never trained on): bare 0.35-0.44 give the
// same violation F1 (0.76-0.77, 95 % CI 0.69-0.84), 0.35 catches 83 % of bare heads instead of 76 %; the helmet threshold
// changes nothing between 0.10 and 0.25 (0.7 % of "OK" heads are bare), 0.25 shows fewer weak helmet boxes
const DEFAULTS = { conf: [0.55, 0.5, 0.25, 0.35], bareNoPerson: 0.45 };
const CONF = DEFAULTS.conf.slice();  // changed from the Settings panel, kept in this browser
const INFER_BARE = false;  // the model marks bare heads itself; guessing from a missing head box lowered F1 on both test sets
const CAND_CONF = 0.15;  // candidate floor; lowered below a class threshold set lower in Settings
const HELMET_ON_HEAD = 0.3;
let BARE_NO_PERSON = DEFAULTS.bareNoPerson;  // bare head on a bike without a detected rider: Hanoi test 16 such violations -> 5 (mostly pedestrians, parked bikes), Hanoi val 4 -> 1
const TARGET_MS = 62;
const IOU_NMS = 0.6;
const RIDER_MIN_PX = 80;
const BIG_FRAC = 0.08;
const SMALL_FRAC = 0.05;
const TILE = { size: 640, cols: 3, rows: 2, overlap: 0.15, maxZoom: 1.15, maxAgeMs: 500 };
const LEVELS = [
  { main: 1280, aux: 320, tile: true, name: "1280 + 320 + ô nhỏ" },
  { main: 1280, aux: 320, tile: false, name: "1280 + 320" },
  { main: 960, aux: 320, tile: true, name: "960 + 320 + ô nhỏ" },
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
const tracker = { tracks: [], nextId: 1, violators: new Set(), number: new Map(), recent: [], logged: 0 };
const tileCache = { next: 0, dets: [], ts: [], key: "" };

// one session.run at a time: a new source can start a loop while the old one is still waiting on the GPU
let runQueue = Promise.resolve();
function runOnce(feeds) { const p = runQueue.then(() => session.run(feeds)); runQueue = p.catch(() => {}); return p; }

function setProgress(p) { $("bar").style.width = `${Math.round(100 * p)}%`; }

async function loadModel() {
  $("loadmsg").textContent = "Đang tải mô hình…"; setProgress(0.05);
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;
  const parts = [];
  for (let i = 1; i <= 9; i++) {
    const r = await fetch(`model_part${i}.bin?v=v11`);
    if (!r.ok) break;
    parts.push(new Uint8Array(await r.arrayBuffer()));
    setProgress(Math.min(0.7, 0.05 + 0.33 * parts.length));
  }
  if (!parts.length) throw new Error("không tìm thấy tệp mô hình");
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
  $("backend").textContent = backend === "webgpu" ? "WebGPU (chip đồ hoạ)" : "WebAssembly (bộ xử lý)";
  $("loadmsg").textContent = "Đang khởi động chip đồ hoạ…";
  for (const [k, S] of [1280, 960, 640, 480, 320].entries()) {
    setProgress(0.72 + 0.056 * k);
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
  const o = await runOnce({ [session.inputNames[0]]: m.tensor });
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
  const cand = Math.min(CAND_CONF, ...CONF);
  const data = out.data, nc = CLASSES.length, N = out.dims[2], dets = [];
  for (let i = 0; i < N; i++) {
    let best = 0, cls = 0;
    for (let c = 0; c < nc; c++) { const s = data[(4 + c) * N + i]; if (s > best) { best = s; cls = c; } }
    if (best < cand) continue;
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

// a bare head with no rider box under it (person not detected) makes a violation only when it is confident and sits
// over the middle of the bike at a rider's size; otherwise parked bikes next to pedestrians were flagged
function headOnBike(hd, b) {
  const bw = b[2] - b[0], hx = (hd[0] + hd[2]) / 2, hw = hd[2] - hd[0];
  return hd[4] >= BARE_NO_PERSON && hx >= b[0] + 0.15 * bw && hx <= b[2] - 0.15 * bw && hw >= 0.12 * bw && hw <= 0.6 * bw;
}

function composeRiders(dets) {
  const bikes = dets.filter((d) => d[5] === 0), persons = dets.filter((d) => d[5] === 1);
  const heads = dets.filter((d) => d[5] === 2 || d[5] === 3).sort((a, b) => b[4] - a[4]);
  const out = bikes.map((b) => ({ box: b, riders: [], status: "unknown" }));
  const riders = [];
  for (const p of persons) {
    let best = -1, bestS = 0; const pw = p[2] - p[0];
    bikes.forEach((b, bi) => {
      const bh = b[3] - b[1];
      const xov = Math.max(0, Math.min(p[2], b[2]) - Math.max(p[0], b[0])) / Math.max(pw, 1e-6);
      const pcx = (p[0] + p[2]) / 2;
      // a rider sits on the bike: body centre over the bike, feet not below its wheels (a pedestrian beside a bike fails one of these)
      if (xov < 0.3 || pcx < b[0] || pcx > b[2] || p[1] > b[1] + 0.5 * bh || p[3] < b[1] + 0.1 * bh || p[3] > b[3] + 0.15 * bh) return;
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
      if (free) free.head = hd;
      else if (hd[5] === 2 || headOnBike(hd, bikes[best])) riders.push({ person: null, bike: best, head: hd });
    }
  }
  for (const r of riders) {
    r.status = r.head ? (r.head[5] === 2 ? "helmet" : "no_helmet") : (INFER_BARE && r.person && r.person[3] - r.person[1] >= RIDER_MIN_PX ? "no_helmet" : "unknown");
    out[r.bike].riders.push(r);
  }
  for (const b of out) {
    const st = b.riders.map((r) => r.status);
    b.status = st.includes("no_helmet") ? "violation" : (st.length && st.every((s) => s === "helmet") ? "ok" : "unknown");
  }
  return out;
}

// fallback when boxes jump between frames: the closest recent track of a similar size
function nearestTrack(bx, used) {
  const cx = (bx[0] + bx[2]) / 2, cy = (bx[1] + bx[3]) / 2, sz = Math.max(bx[2] - bx[0], bx[3] - bx[1]);
  let best = null, bestD = 0.6 * sz;
  for (const t of tracker.tracks) {
    if (used.has(t) || t.missed > 5) continue;
    const r = t.raw, tz = Math.max(r[2] - r[0], r[3] - r[1]);
    if (tz < 0.6 * sz || tz > 1.6 * sz) continue;
    const d = Math.hypot((r[0] + r[2]) / 2 - cx, (r[1] + r[3]) / 2 - cy);
    if (d < bestD) { best = t; bestD = d; }
  }
  return best;
}

// a track that lost its id and came back as a new one is still the same rider: do not log it twice
function recentViolation(bx) {
  const now = performance.now();
  tracker.recent = tracker.recent.filter((r) => now - r.t < 3000);
  const sz = Math.max(bx[2] - bx[0], bx[3] - bx[1]);
  return tracker.recent.find((r) => r.track.missed > 0 && (iou(r.box, bx) > 0.2 ||
    Math.hypot((r.box[0] + r.box[2] - bx[0] - bx[2]) / 2, (r.box[1] + r.box[3] - bx[1] - bx[3]) / 2) < 0.8 * sz));
}

function track(bikes, isVideo) {
  const used = new Set();
  for (const b of bikes) {
    let best = null, bestI = 0.25;
    for (const t of tracker.tracks) { const v = iou(t.raw, b.box); if (!used.has(t) && v > bestI) { best = t; bestI = v; } }
    if (!best) best = nearestTrack(b.box, used);
    if (!best) { best = { id: tracker.nextId++, hist: [], missed: 0, smooth: b.box.slice(0, 4) }; tracker.tracks.push(best); }
    used.add(best); best.raw = b.box; best.missed = 0;
    best.smooth = best.smooth.map((v, k) => 0.5 * v + 0.5 * b.box[k]);
    best.hist.push(b.status); if (best.hist.length > 8) best.hist.shift();
    const nv = best.hist.filter((s) => s === "violation").length, no = best.hist.filter((s) => s === "ok").length;
    const recent = best.hist.slice(-3);
    if (!isVideo) b.shown = b.status;
    // violation as soon as one of the last 3 frames saw a bare head and bare calls are not outnumbered by helmet calls;
    // the old rule (60 % of 8 frames, at most one helmet call) halved the violators caught on the Hanoi video windows
    else if (recent.includes("violation") && nv >= no) b.shown = "violation";
    else if (no >= 2 || (no >= 1 && nv === 0)) b.shown = "ok";
    else b.shown = "unknown";
    b.id = best.id; b.drawBox = isVideo ? best.smooth : b.box;
    if (b.shown === "violation" && !tracker.violators.has(best.id)) {
      const same = recentViolation(b.box);
      if (same) tracker.number.set(best.id, same.n);
      else { tracker.logged++; tracker.number.set(best.id, tracker.logged); }
      tracker.recent.push({ box: b.box.slice(0, 4), t: performance.now(), n: tracker.number.get(best.id), track: best });
      tracker.violators.add(best.id);
    }
  }
  for (const t of tracker.tracks) if (!used.has(t)) t.missed++;
  tracker.tracks = tracker.tracks.filter((t) => t.missed < 10);
}

// tags drawn this frame, so later tags do not cover earlier ones
let placed = [];

function label(text, x, y, color, s, ink = "#fff", sub = "", optional = false) {
  ctx.font = `700 ${Math.round(15 * s)}px ${FONT}`;
  const tw = ctx.measureText(text).width;
  ctx.font = `500 ${Math.round(13 * s)}px ${FONT}`;
  const sw = sub ? ctx.measureText(sub).width + 6 * s : 0;
  const w = tw + sw + 14 * s, h = 22 * s; y = Math.max(h, Math.min(y, view.height));
  x = Math.max(0, Math.min(x, view.width - w));
  const hits = (yy) => placed.some((p) => x < p[2] && x + w > p[0] && yy - h < p[3] && yy > p[1]);
  if (hits(y)) {
    if (optional) return;
    for (let k = 1; k < 4 && hits(y); k++) y += h + 2 * s;
  }
  placed.push([x, y - h, x + w, y]);
  ctx.fillStyle = color; ctx.beginPath();
  if (ctx.roundRect) ctx.roundRect(x, y - h, w, h, 4 * s); else ctx.rect(x, y - h, w, h);
  ctx.fill();
  ctx.fillStyle = ink; ctx.font = `700 ${Math.round(15 * s)}px ${FONT}`; ctx.fillText(text, x + 7 * s, y - 6 * s);
  if (sub) { ctx.globalAlpha = 0.8; ctx.font = `500 ${Math.round(13 * s)}px ${FONT}`; ctx.fillText(sub, x + 7 * s + tw + 6 * s, y - 6.5 * s); ctx.globalAlpha = 1; }
}

function box(b, color, width, r, alpha = 1) {
  ctx.globalAlpha = alpha; ctx.strokeStyle = color; ctx.lineWidth = width; ctx.beginPath();
  if (ctx.roundRect) ctx.roundRect(b[0], b[1], b[2] - b[0], b[3] - b[1], r); else ctx.rect(b[0], b[1], b[2] - b[0], b[3] - b[1]);
  ctx.stroke(); ctx.globalAlpha = 1;
}

// motorbike: corner brackets in its status colour over a faint outline
function brackets(b, color, s) {
  const w = b[2] - b[0], h = b[3] - b[1], k = Math.max(8 * s, 0.2 * Math.min(w, h));
  box(b, color, 1.5 * s, 2 * s, 0.45);
  ctx.strokeStyle = color; ctx.lineWidth = 4 * s; ctx.lineCap = "round"; ctx.lineJoin = "round"; ctx.beginPath();
  for (const [x, y, dx, dy] of [[b[0], b[1], 1, 1], [b[2], b[1], -1, 1], [b[0], b[3], 1, -1], [b[2], b[3], -1, -1]]) {
    ctx.moveTo(x, y + dy * k); ctx.lineTo(x, y); ctx.lineTo(x + dx * k, y);
  }
  ctx.stroke(); ctx.lineCap = "butt";
}

// display only: one head box per head (helmet and no_helmet boxes often stack on the same head)
function shownDets(dets) {
  const heads = [];
  for (const d of dets.filter((d) => d[5] >= 2).sort((a, b) => b[4] - a[4])) if (!heads.some((k) => iou(k, d) > 0.3)) heads.push(d);
  return dets.filter((d) => d[5] < 2).concat(heads);
}

function drawResults({ dets, bikes }) {
  const cssW = Math.max(1, view.getBoundingClientRect().width);
  // t: tag size; phones get small tags and only the violation tag, so the boxes stay visible
  const phone = cssW < 700, s = Math.max(1, view.width / cssW), t = s * (phone ? 0.6 : Math.min(1, Math.max(0.7, cssW / 800)));
  const shownList = shownDets(dets); placed = [];
  for (const d of shownList) if (d[5] === 1) box(d, COLORS[1], 1.5 * s, 3 * s, 0.7);
  for (const b of bikes) {
    const st = b.shown || b.status;
    brackets(b.drawBox || b.box, STATUS_COLORS[st], s);
    if (st === "violation") for (const r of b.riders) if (r.person && r.status === "no_helmet") box(r.person, STATUS_COLORS.violation, 2.5 * s, 3 * s);
  }
  // tags only where they help: every violation; helmet-OK only on wide screens and big motorbikes
  for (const b of bikes) {
    const bb = b.drawBox || b.box, st = b.shown || b.status;
    let top = bb[1];
    for (const r of b.riders) for (const q of [r.person, r.head]) if (q) top = Math.min(top, q[1]);
    if (st === "violation") label(phone ? "Không mũ" : reasons(b), bb[0], top - 4 * t, STATUS_COLORS.violation, t, "#fff", `#${tracker.number.get(b.id) || ""}`);
    else if (!phone && st === "ok" && bb[2] - bb[0] > 0.12 * view.width) label("Có mũ", bb[0], top - 4 * t, STATUS_COLORS.ok, t * 0.85, "#fff", "", true);
  }
  for (const d of shownList) {
    if (d[5] < 2) continue;
    box(d, COLORS[d[5]], 2.5 * s, 3 * s);
    if (d[5] === 3 && !phone) label(`Không mũ ${Math.round(100 * d[4])}%`, d[0], d[1] - 3 * s, COLORS[3], t * 0.75, "#2a0d05", "", true);
  }
}

function clock(t) { const m = Math.floor(t / 60), x = Math.floor(t % 60); return `${String(m).padStart(2, "0")}:${String(x).padStart(2, "0")}`; }

// Law 36/2024/QH15 Art. 33(2): the rider and every passenger must wear a helmet
function reasons(b) {
  const n = b.riders.filter((r) => r.status === "no_helmet").length;
  return n > 1 ? `${n} người không mũ` : "Không đội mũ";
}

function rate(ts) { return ts.length > 1 ? 1000 * (ts.length - 1) / (ts[ts.length - 1] - ts[0]) : 0; }

let flagged = 0, lastFlash = 0;

function updateHud() {
  $("fps").textContent = rate(renderTimes).toFixed(0); $("ms").textContent = emaMs.toFixed(0);
  $("dps").textContent = still ? "1" : rate(inferTimes).toFixed(0);
  $("msize").textContent = `${LEVELS[level].name} px (chuẩn bị ${shown.prep.toFixed(0)} ms, GPU ${shown.run.toFixed(0)} ms)`;
  const { dets, bikes } = latest;
  $("nb").textContent = dets.filter((d) => d[5] === 0).length; $("np").textContent = dets.filter((d) => d[5] === 1).length;
  const heads = shownDets(dets);
  $("nh").textContent = heads.filter((d) => d[5] === 2).length; $("nn").textContent = heads.filter((d) => d[5] === 3).length;
  const total = tracker.logged; $("uv").textContent = total;
  const st = bikes.map((b) => b.shown || b.status), nv = st.filter((x) => x === "violation").length, nok = st.filter((x) => x === "ok").length;
  $("nv").textContent = nv;
  $("disc").className = "count " + (nv ? "bad" : nok && nok === st.length ? "ok" : "wait");
  $("viol").textContent = nv ? `${nv} người không đội mũ` :
    !bikes.length ? "Đang tìm xe máy" : nok === st.length ? "Mọi người trong khung hình đều đội mũ" :
    nok ? `${nok} xe đội mũ, ${st.length - nok} xe đang kiểm tra` : "Đang kiểm tra người trên xe";
  $("seen").textContent = total ? `Đã ghi ${total} trường hợp` : "Chưa có ai không đội mũ";
  if (total > flagged && !still && performance.now() - lastFlash > 4000) {
    lastFlash = performance.now(); const f = $("flash"); f.classList.remove("on"); void f.offsetWidth; f.classList.add("on");
  }
  flagged = total;
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
        latest = { dets, bikes }; $("err").textContent = "";
        inferTimes.push(performance.now()); if (inferTimes.length > 20) inferTimes.shift();
      } catch (e) { console.error(e); $("err").textContent = "Lỗi: " + (e && e.message || e); await new Promise((r) => setTimeout(r, 500)); }
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
  } catch (e) { console.error(e); $("err").textContent = "Lỗi: " + (e && e.message || e); }
  ctx.drawImage(frameC, 0, 0); drawResults(latest); updateHud();
}

function showUi() {
  flagged = 0; $("status").classList.add("hidden");
  for (const id of ["hud", "controls"]) $(id).classList.remove("hidden");
}

function resetState() {
  level = benchLevel; settle = 3; emaMs = 0; slowCount = fastCount = 0;
  tileCache.dets = []; tileCache.ts = []; tileCache.key = ""; tileCache.next = 0;
  tracker.tracks = []; tracker.nextId = 1; tracker.violators.clear(); tracker.number.clear(); tracker.recent = []; tracker.logged = 0; latest = { dets: [], bikes: [] }; renderTimes = []; inferTimes = [];
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

// Settings: per-class confidence thresholds, kept in this browser only
const SET_KEY = "helmetCheck.settings.v1";
const SLIDERS = [["sMoto", 0], ["sPerson", 1], ["sHelmet", 2], ["sBare", 3]];
function applySettings(save) {
  for (const [id, k] of SLIDERS) { CONF[k] = +$(id).value; $(id + "V").textContent = CONF[k].toFixed(2); }
  BARE_NO_PERSON = +$("sNoPerson").value; $("sNoPersonV").textContent = BARE_NO_PERSON.toFixed(2);
  const changed = CONF.some((v, k) => v !== DEFAULTS.conf[k]) || BARE_NO_PERSON !== DEFAULTS.bareNoPerson;
  $("setNote").textContent = changed ? "Đang dùng giá trị tự chỉnh (khác mức đã kiểm định)" : "Đang dùng mức khuyên dùng";
  if (save) try { localStorage.setItem(SET_KEY, JSON.stringify({ conf: CONF, bareNoPerson: BARE_NO_PERSON })); } catch (e) {}
}
function setSliders(v) {
  for (const [id, k] of SLIDERS) $(id).value = v.conf[k];
  $("sNoPerson").value = v.bareNoPerson;
}
try { const v = JSON.parse(localStorage.getItem(SET_KEY)); if (v && v.conf && v.conf.length === 4) setSliders(v); } catch (e) {}
applySettings(false);
for (const id of [...SLIDERS.map((x) => x[0]), "sNoPerson"]) $(id).oninput = () => applySettings(true);
$("setReset").onclick = () => { setSliders(DEFAULTS); applySettings(true); };
$("setBtn").onclick = () => {
  const open = $("settings").classList.toggle("hidden") === false; $("setBtn").setAttribute("aria-expanded", open);
};

$("startBtn").onclick = () => startCamera().catch((e) => {
  $("loadmsg").textContent = `Không bật được camera (${e.message}). Hãy cho phép trang này dùng camera trong Cài đặt rồi thử lại.`;
});
$("flipBtn").onclick = () => { facing = facing === "environment" ? "user" : "environment"; startCamera().catch((e) => alert(e.message)); };
$("speed").onclick = () => {
  const open = $("details").classList.toggle("hidden") === false; $("speed").setAttribute("aria-expanded", open);
};
$("stopBtn").onclick = () => {
  stopSource(); $("status").classList.remove("hidden");
  for (const id of ["hud", "controls", "details", "settings"]) $(id).classList.add("hidden");
};
$("file").onchange = (e) => {
  const f = e.target.files[0]; if (!f) return;
  openUrl(URL.createObjectURL(f), f.type.startsWith("image/"));
};
document.addEventListener("visibilitychange", async () => {
  if (document.visibilityState === "visible" && stream && !wakeLock) { try { wakeLock = await navigator.wakeLock.request("screen"); } catch (e) {} }
});

(async () => {
  try {
    await Promise.all([loadModel(), document.fonts ? document.fonts.load(`700 16px ${FONT}`).catch(() => {}) : null]);
    setProgress(1); document.body.classList.remove("loading");
    $("loadmsg").textContent = backend === "webgpu" ? "Sẵn sàng. Đang chạy trên chip đồ hoạ." :
      "Sẵn sàng. Trình duyệt này không có WebGPU nên chạy trên bộ xử lý, kiểm tra được ít khung hình hơn mỗi giây.";
    $("startBtn").classList.remove("hidden"); $("fileLbl").classList.remove("hidden");
    const src = new URLSearchParams(location.search).get("src");
    if (src) openUrl(src, /\.(jpe?g|png|webp)$/i.test(src));
  } catch (e) {
    document.body.classList.remove("loading");
    $("loadmsg").textContent = `Không tải được mô hình (${e.message}). Kiểm tra kết nối mạng rồi tải lại trang.`;
  }
})();
