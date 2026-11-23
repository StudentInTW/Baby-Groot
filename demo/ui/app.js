// 護理站介面原型（給影片的操作示範錄螢幕用）。不接後端、不連外；燈號全部來自 cribs.js（系統對娃娃影片的真實判定）。
// 畫面：1 安裝（框床位、標床頭、拍空床照，存 localStorage）→ 2 護理站牆面（8 床同步播放）→ 3 單床詳細
//       → 4 紅燈處理（Acknowledge 記到事件列表）→ 5 手機推播（只有文字）。
// 離線（紫燈，2026/11/23）：右上角 ⚡ 模擬一次晶片卡住（錄好的判定裡沒有真的斷線，所以是模擬，畫面上會標明）。
//   跟 live_view.py／crib_state.Liveness 同一套規則：最後一次判讀後 OFFLINE_AFTER 秒還沒有新的判讀就變紫燈，
//   主程序 12 秒逾時重開晶片、約 FAULT_S 秒後恢復。鏡頭畫面照常播（影像來自監視主機，不經過 AI 盒子），只是燈號不明。
//   離線期間錄好的燈號變化，恢復時才顯示（那段時間系統其實不知道）；離線前已經響的紅燈不會因為離線就收掉，要等恢復或護理師確認。
//   恢復後直接接錄好的燈號（真的系統會把離線那段當成「看不到」接著計時，見 crib_state.CribState.resume，可能更早變紅）。
//   離線要不要推播到手機還沒定，所以只在護理站顯示。
"use strict";

const PRODUCT = "FirstGlance";        // 產品名（暫定），改這裡就好
const STORE_KEY = "firstglance.setup"; // 安裝設定存在 localStorage
const ALERT_TEXT = "nose and mouth hidden 3 s";
const OFFLINE_AFTER = 2;               // 秒；跟 pipeline/crib_state.OFFLINE_AFTER 一樣
const FAULT_S = 13.5;                  // 模擬晶片卡住到恢復的秒數（hailo/結果.md：逾時 12 秒＋重開約 1.5 秒）
const OFFLINE_STATE = { color: "#7b1fa2", title: "OFFLINE", desc: "Not being analysed right now: check this crib in person" };

const $ = (id) => document.getElementById(id);
let D = null;     // cribs.js 的資料
let END = 0;      // 最長那一段的長度（秒）：播到這裡就停
const S = { t: 0, playing: false, last: null, view: "wall", detail: null, acks: [], notes: 0, phoneOn: false, faults: [] };

// ── 載入資料：data/（真資料）→ 沒有就用 sample/（假資料）；?sample 強制用假資料 ──
function loadScript(src) {
  return new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = src;
    s.onload = resolve;
    s.onerror = reject;
    document.head.appendChild(s);
  });
}

async function loadData() {
  const sources = new URLSearchParams(location.search).has("sample") ? ["sample/cribs.js"] : ["data/cribs.js", "sample/cribs.js"];
  for (const src of sources) {
    try {
      await loadScript(src);
      if (window.FIRSTGLANCE_DATA) return window.FIRSTGLANCE_DATA;
    } catch (e) { /* 試下一個 */ }
  }
  throw new Error("No data. Run: uv run python -m demo.export_assets");
}

// ── 小工具 ──
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const pad = (v) => String(v).padStart(2, "0");
function clockStr(t, seconds = true) {
  const c = D.clock0 + Math.floor(t + 1e-6);
  return `${pad(Math.floor(c / 3600))}:${pad(Math.floor(c / 60) % 60)}` + (seconds ? `:${pad(c % 60)}` : "");
}
// 模擬的晶片故障：S.faults = [{ at: 卡住的時刻, back: 恢復的時刻 }]，所有床一起受影響
const faultAt = (t) => S.faults.find((f) => t > f.at + 1e-6 && t < f.back);   // 這段時間沒有判讀
const offlineAt = (t) => { const f = faultAt(t); return !!f && t >= f.at + OFFLINE_AFTER - 1e-6; };
const shown = (t) => { const f = faultAt(t); return f ? f.back : t; };        // 沒判讀那段的燈號變化，恢復時才看得到
function stateAt(c, t) {
  const f = faultAt(t);
  if (f && t >= f.at + OFFLINE_AFTER - 1e-6) return { show: "offline", silent: t - f.at };
  if (f) t = f.at;                                                             // 還沒到 2 秒：維持卡住前最後一個燈號
  const i = clamp(Math.floor(t / D.step + 1e-6), 0, c.show.length - 1);
  return { show: c.show[i], hidden: c.hidden[i], count: c.count[i], adult: c.adult[i], matches: c.matches[i], vlm: c.vlm[i] };
}
const colorOf = (show) => D.states[show].color;
const byId = (id) => D.cribs.find((c) => c.id === id);
function el(tag, cls, html) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html !== undefined) e.innerHTML = html;
  return e;
}
function store(value) {
  try {
    if (value === undefined) return JSON.parse(localStorage.getItem(STORE_KEY)) || {};
    localStorage.setItem(STORE_KEY, JSON.stringify(value));
  } catch (e) { /* 私密視窗等情況：只是示意，存不了也沒關係 */ }
  return value || {};
}

// 每床預先算好：系統事件（跟 demo_render.render_grid 同一個規則）與紅燈段落
function prep(c) {
  c.events = [];
  c.reds = [];
  let prev = null;
  c.show.forEach((s, i) => {
    const t = i * D.step;
    if (s !== prev && D.event_text[s] && (prev !== null || s === "look" || s === "empty")) c.events.push({ t, crib: c.id, show: s });
    if (s === "red" && prev !== "red") c.reds.push({ start: t, end: Infinity });
    if (s !== "red" && prev === "red") c.reds[c.reds.length - 1].end = t;
    prev = s;
  });
}

// ── 時鐘與影片同步 ──
function setPlaying(on) {
  S.playing = on && S.t < END;
  $("play").textContent = S.playing ? "❚❚" : "▶";
}
function seek(t) {
  S.t = clamp(t, 0, END);
  S.acks = S.acks.filter((a) => a.at <= S.t + 1e-6);
  S.faults = S.faults.filter((f) => f.at <= S.t + 1e-6);   // 往回跳到故障之前：那次故障當作沒發生
  S.notes = Infinity; // 往回跳不要震動
}
function syncVideo(v, c) {
  if (!v || !c || v.readyState < 1 || v.seeking) return;
  const last = (isFinite(v.duration) ? v.duration : c.duration) - 0.05;
  const target = Math.min(S.t, last);
  if (S.playing && S.t < last) {
    if (v.paused) v.play().catch(() => {});
    if (Math.abs(v.currentTime - target) > 0.35) v.currentTime = target;
  } else {
    if (!v.paused) v.pause();
    if (Math.abs(v.currentTime - target) > 0.04) v.currentTime = target;
  }
}
function pauseAll(except) {
  document.querySelectorAll("video").forEach((v) => { if (!except.includes(v) && !v.paused) v.pause(); });
}

function tick(now) {
  if (S.playing && S.last !== null) {
    S.t = Math.min(S.t + (now - S.last) / 1000, END);
    if (S.t >= END) setPlaying(false);
  }
  S.last = now;
  render();
  requestAnimationFrame(tick);
}

// ── 畫面切換 ──
function show(view) {
  S.view = view;
  for (const v of ["setup", "wall", "detail"]) $(`view-${v}`).hidden = v !== view;
  document.querySelectorAll(".tabs button").forEach((b) => b.classList.toggle("active",
    b.dataset.view === view || (view === "detail" && b.dataset.view === "wall")));
  if (view === "setup") setupRender();
}
function openDetail(id) {
  const c = byId(id);
  S.detail = c;
  const v = $("detail-video");
  if (v.dataset.src !== c.video) {
    v.dataset.src = c.video;
    v.src = c.video;
  }
  $("detail-label").textContent = `Crib ${c.id} · per-crib camera`;
  $("tl-start").textContent = clockStr(0);
  $("tl-end").textContent = clockStr(c.duration);
  show("detail");
  drawTimeline(c);
}

// ── 2. 護理站牆面 ──
function buildWall() {
  const wall = $("wall");
  for (const c of D.cribs) {
    const tile = el("div", "tile");
    tile.dataset.id = c.id;
    const v = el("video");
    v.muted = true;
    v.playsInline = true;
    v.preload = "auto";
    v.src = c.video;
    v.addEventListener("error", () => { tile.querySelector(".tag").textContent = `${c.id} · video missing`; });
    tile.append(v, el("div", "tag"), el("div", "mini", "<div></div>"));
    tile.addEventListener("click", () => openDetail(c.id));
    c.tile = tile;
    c.tileVideo = v;
    wall.append(tile);
  }
  $("wall-sub").textContent = `${D.cribs.length} cribs · AI runs on the box, no video stored`;
  $("wall-note").textContent = D.sample
    ? "Sample data: synthetic clips with scripted lights (for development only)."
    : "Mock-up: all cribs are clips of the same doll videos; each light is our system's real output.";
}

function renderWall() {
  const counts = {};
  for (const c of D.cribs) {
    const st = stateAt(c, S.t);
    const meta = D.states[st.show];
    counts[meta.title] = (counts[meta.title] || 0) + 1;
    if (c.tile._show !== st.show) {
      c.tile._show = st.show;
      c.tile.style.setProperty("--c", meta.color);
      c.tile.classList.toggle("red", st.show === "red");
      c.tile.classList.toggle("offline", st.show === "offline");
      c.tile.querySelector(".tag").textContent = `${c.id}  ${meta.title}`;
    }
    const mini = c.tile.querySelector(".mini");
    mini.hidden = st.show !== "count";
    if (st.show === "count") mini.firstChild.style.width = `${Math.min(st.count / 3, 1) * 100}%`;
  }
  const off = offlineAt(S.t);
  $("offline-banner").hidden = !off;
  if (off) $("offline-text").textContent =
    `All ${D.cribs.length} cribs · no analysis for ${Math.floor(S.t - faultAt(S.t).at)} s · check the cribs in person`;
  const key = JSON.stringify(counts);
  if ($("counts")._key !== key) {
    $("counts")._key = key;
    $("counts").innerHTML = "";
    for (const title of ["GREEN", "YELLOW", "RED", "BLUE", "GRAY", "OFFLINE"]) {
      const color = Object.values(D.states).find((s) => s.title === title).color;
      const box = el("div", "", `${counts[title] || 0}<small>${title}</small>`);
      box.style.background = color;
      $("counts").append(box);
    }
  }
  renderEvents();
}

// 離線那段裡發生的變化：系統只在恢復時看到最後的結果，所以只留每床最後一個、而且恢復時還是那個燈的事件
function seenEvent(c, e) {
  const f = faultAt(e.t);
  if (!f) return true;
  const last = c.events.filter((x) => faultAt(x.t) === f).pop();
  return last === e && stateAt(c, f.back).show === e.show;
}
function renderEvents() {
  const evs = [];
  for (const c of D.cribs) for (const e of c.events) if (shown(e.t) <= S.t + 1e-6 && seenEvent(c, e)) evs.push({ ...e, t: shown(e.t) });
  for (const a of S.acks) evs.push({ t: a.at, crib: a.crib, ack: a });
  for (const f of S.faults) {
    if (f.at + OFFLINE_AFTER <= S.t + 1e-6) evs.push({ t: f.at + OFFLINE_AFTER, sys: "offline", f });
    if (f.back <= S.t + 1e-6) evs.push({ t: f.back, sys: "online", f });
  }
  evs.sort((a, b) => b.t - a.t);
  const key = evs.map((e) => `${e.crib}${e.t}${e.show || e.sys || "ack"}`).join("|");
  if ($("events")._key === key) return;
  $("events")._key = key;
  $("events").innerHTML = "";
  for (const e of evs.slice(0, 30)) {
    const li = el("li", e.ack ? "ack" : "");
    const dot = el("i", "dot");
    if (e.sys) {
      dot.style.background = e.sys === "offline" ? colorOf("offline") : colorOf("green");
      li.append(dot, el("strong", "", e.sys === "offline" ? "All cribs · OFFLINE" : "All cribs · back online"),
        el("span", "", e.sys === "offline"
          ? `${clockStr(e.t)} · AI box not answering for ${OFFLINE_AFTER} s (simulated chip fault)`
          : `${clockStr(e.t)} · analysis resumed after ${(e.f.back - e.f.at).toFixed(1)} s`));
    } else if (e.ack) {
      dot.style.background = colorOf("care");
      li.append(dot, el("strong", "", `Crib ${e.crib} · nurse at the crib`),
        el("span", "", `${clockStr(e.t)} · acknowledged ${(e.ack.at - shown(e.ack.start)).toFixed(1)} s after the alert`));
    } else {
      dot.style.background = colorOf(e.show);
      li.append(dot, el("strong", "", `Crib ${e.crib} · ${D.states[e.show].title}`),
        el("span", "", `${clockStr(e.t)} · ${D.event_text[e.show]}`));
    }
    $("events").append(li);
  }
}

// ── 3. 單床詳細 ──
function drawTimeline(c) {
  const cv = $("timeline-canvas");
  const w = (cv.width = cv.clientWidth * devicePixelRatio || 1000);
  const h = (cv.height = cv.clientHeight * devicePixelRatio || 34);
  const g = cv.getContext("2d");
  const n = c.show.length;
  c.show.forEach((s, i) => {
    g.fillStyle = colorOf(s);
    g.fillRect(Math.floor((i / n) * w), 0, Math.ceil(w / n) + 1, h);
  });
  for (const f of S.faults) {   // 模擬的離線那段
    const x0 = clamp((f.at + OFFLINE_AFTER) / c.duration, 0, 1) * w, x1 = clamp(f.back / c.duration, 0, 1) * w;
    g.fillStyle = colorOf("offline");
    g.fillRect(Math.floor(x0), 0, Math.ceil(x1 - x0), h);
  }
  cv._key = JSON.stringify(S.faults);
}

function renderDetail() {
  const c = S.detail;
  const st = stateAt(c, S.t);
  const meta = D.states[st.show];
  $("view-detail").style.setProperty("--c", meta.color);
  $("detail-title").textContent = meta.title;
  $("detail-desc").textContent = meta.desc;
  $("detail-countdown").hidden = st.show !== "count";
  if (st.show === "count") {
    $("detail-fill").style.width = `${Math.min(st.count / 3, 1) * 100}%`;
    $("detail-count").textContent = `${st.count.toFixed(1)} s of 3 s`;
  }
  const grey = colorOf("empty");
  const signals = st.show === "offline" ? [
    ["Last completed analysis", `${Math.floor(st.silent)} s ago`, colorOf("offline")],
    ["Reason", "AI box not answering (simulated chip fault)", colorOf("offline")],
    ["Camera picture", "still live: it comes from the camera recorder", grey],
  ] : [
    ["Face detector, every frame", st.hidden ? "nose and mouth NOT visible" : "nose and mouth visible",
      st.hidden ? colorOf("red") : colorOf("green")],
    ["VLM second look, every second", st.vlm === null ? "—" : `P(nose and mouth visible) = ${st.vlm.toFixed(2)}`,
      st.vlm === null ? grey : st.vlm < D.vlm_th ? colorOf("count") : colorOf("green")],
    ["Adult at the crib (edge motion)", st.adult ? "yes: timer paused" : "no", st.adult ? colorOf("care") : grey],
    ["Matches this crib's empty photo", st.matches ? "yes" : "no", grey],
  ];
  const key = JSON.stringify(signals);
  if ($("signals")._key !== key) {
    $("signals")._key = key;
    $("signals").innerHTML = "";
    for (const [label, value, color] of signals) {
      const dot = el("i", "dot");
      dot.style.background = color;
      const li = el("li");
      li.append(dot, el("span", "", label), el("strong", "", value));
      $("signals").append(li);
    }
  }
  if ($("timeline-canvas")._key !== JSON.stringify(S.faults)) drawTimeline(c);   // 按了 ⚡ 或往回跳
  $("timeline-cursor").style.left = `${clamp(S.t / c.duration, 0, 1) * 100}%`;
}

// ── 4. 紅燈處理 ──
// 紅燈段落實際被看到的時段：開始在離線那段裡的，恢復時才響；在離線那段裡結束的，恢復時才知道已經好了；
// 整段都在離線裡的，系統根本不知道（不響）
const redFrom = (ep) => shown(ep.start);
const redTo = (ep) => shown(ep.end);
const redSeen = (ep) => redFrom(ep) < redTo(ep);
function activeAlert() {
  for (const c of D.cribs) {
    const ep = c.reds.find((e) => redSeen(e) && redFrom(e) <= S.t + 1e-6 && S.t < redTo(e));
    if (ep && !S.acks.some((a) => a.crib === c.id && a.start === ep.start)) return { c, ep };
  }
  return null;
}
function renderAlert() {
  const a = S.view === "setup" ? null : activeAlert();
  $("alert").hidden = !a;
  if (!a) return;
  $("alert-title").textContent = `Crib ${a.c.id} · RED`;
  $("alert-sub").textContent = `${ALERT_TEXT} · ${(S.t - redFrom(a.ep)).toFixed(0)} s ago`;
  $("alert").dataset.crib = a.c.id;
  $("alert").dataset.start = a.ep.start;
}

// ── 5. 手機推播（只有文字） ──
function renderPhone() {
  const notes = [];
  for (const c of D.cribs) for (const ep of c.reds) if (redSeen(ep) && redFrom(ep) <= S.t + 1e-6) notes.push({ c, ep });
  notes.sort((a, b) => redFrom(b.ep) - redFrom(a.ep));
  if (notes.length > S.notes && !$("phone").hidden) {
    $("phone").classList.remove("buzz");
    void $("phone").offsetWidth;
    $("phone").classList.add("buzz");
  }
  S.notes = notes.length;
  if (S.phoneOn) {
    $("phone-time").textContent = clockStr(S.t, false);
    $("phone-clock").textContent = clockStr(S.t, false);
  }
  const key = notes.map((n) => n.c.id + n.ep.start + S.acks.some((a) => a.crib === n.c.id && a.start === n.ep.start)).join("|");
  if ($("notes")._key === key) return;
  $("notes")._key = key;
  $("notes").innerHTML = "";
  for (const { c, ep } of notes.slice(0, 5)) {
    const ack = S.acks.find((a) => a.crib === c.id && a.start === ep.start);
    const li = el("li");
    li.append(el("div", "app", `<b>${PRODUCT.toUpperCase()}</b><span>${clockStr(redFrom(ep), false)}</span>`),
      el("strong", "", `Crib ${c.id} · ${ALERT_TEXT}`));
    if (ack) li.append(el("div", "acked", `✓ You acknowledged (${(ack.at - redFrom(ep)).toFixed(1)} s)`));
    $("notes").append(li);
  }
}

// ── 1. 安裝（純前端示意） ──
const SU = { crib: null, step: 1, rect: null, head: null, drag: null, photoAt: null };
const HELP = {
  1: ["Pick a crib", "Choose a crib on the left. Each crib has its own camera; the box only reads that camera."],
  2: ["Draw the crib area", "Drag a box around the mattress. Only what's inside the box is checked."],
  3: ["Mark the head end", "Click where the baby's head goes. The box turns the picture so the head is up before looking for the face."],
  4: ["Empty-crib photo", "With the crib empty, take one photo. Later, if the crib looks like this photo, the light turns gray instead of red."],
  5: ["Done", ""],
};

function setupList() {
  const saved = store();
  $("setup-cribs").innerHTML = "";
  for (const c of D.cribs) {
    const li = el("li", c.id === SU.crib ? "active" : "", `<span>Crib ${c.id}</span>`);
    if (saved[c.id]) li.append(el("span", "ok", "✓ Ready"));
    li.addEventListener("click", () => setupPick(c.id));
    $("setup-cribs").append(li);
  }
}
function setupPick(id) {
  Object.assign(SU, { crib: id, step: 2, rect: null, head: null, drag: null, photoAt: null });
  const v = $("setup-video");
  v.src = byId(id).video;
  v.addEventListener("loadeddata", () => { v.currentTime = 0.5; }, { once: true });
  setupRender();
}
function setupRender() {
  setupList();
  document.querySelectorAll("#steps li").forEach((li) => {
    const s = Number(li.dataset.step);
    li.className = s === SU.step ? "cur" : s < SU.step ? "done" : "";
  });
  const [title, help] = HELP[SU.step];
  $("setup-title").textContent = SU.step === 5 ? `Crib ${SU.crib} is monitored` : title;
  $("setup-help").innerHTML = SU.step === 5 ? doneHtml() : help;
  $("setup-thumb-wrap").hidden = !SU.photoAt;
  const next = $("setup-next");
  const redo = $("setup-redo");
  next.hidden = redo.hidden = true;
  if (SU.step === 2 && SU.rect) { next.hidden = redo.hidden = false; next.textContent = "Next"; }
  if (SU.step === 3 && SU.head) { next.hidden = redo.hidden = false; next.textContent = "Next"; }
  if (SU.step === 4) { next.hidden = false; next.textContent = SU.photoAt ? "Finish" : "Capture empty-crib photo"; redo.hidden = !SU.photoAt; }
  if (SU.step === 5) { next.hidden = false; next.textContent = "Open nurse station"; }
  $("setup-frame").style.visibility = SU.crib ? "visible" : "hidden";
  drawSetup();
}
function doneHtml() {
  const rows = ["green", "count", "red", "care", "empty", "offline"].map((k) =>
    `<li><i style="background:${colorOf(k)}"></i><b>${D.states[k].title}</b><span>${D.states[k].desc}</span></li>`).join("");
  return `The box now watches this crib. Lights on the nurse station:<ul class="legend">${rows}</ul>`;
}
function setupCanvas() {
  const cv = $("setup-canvas");
  const r = cv.getBoundingClientRect();
  if (cv.width !== Math.round(r.width * devicePixelRatio)) {
    cv.width = Math.round(r.width * devicePixelRatio);
    cv.height = Math.round(r.height * devicePixelRatio);
  }
  return cv;
}
function drawSetup() {
  const cv = setupCanvas();
  const g = cv.getContext("2d");
  const W = cv.width, H = cv.height, k = devicePixelRatio;
  g.clearRect(0, 0, W, H);
  const r = SU.drag || SU.rect;
  if (r) {
    const [x0, y0, x1, y1] = [Math.min(r[0], r[2]) * W, Math.min(r[1], r[3]) * H, Math.max(r[0], r[2]) * W, Math.max(r[1], r[3]) * H];
    g.fillStyle = "rgba(0,0,0,.45)";
    g.fillRect(0, 0, W, y0); g.fillRect(0, y1, W, H - y1); g.fillRect(0, y0, x0, y1 - y0); g.fillRect(x1, y0, W - x1, y1 - y0);
    g.setLineDash([10 * k, 6 * k]);
    g.lineWidth = 3 * k;
    g.strokeStyle = "#fff";
    g.strokeRect(x0, y0, x1 - x0, y1 - y0);
    g.setLineDash([]);
    g.fillStyle = "#fff";
    g.font = `bold ${16 * k}px system-ui, sans-serif`;
    g.fillText("CRIB AREA", x0 + 8 * k, y0 + 22 * k);
  }
  if (SU.head) {
    const [x, y] = [SU.head[0] * W, SU.head[1] * H];
    g.fillStyle = "#1565c0";
    g.beginPath(); g.arc(x, y, 14 * k, 0, Math.PI * 2); g.fill();
    g.strokeStyle = "#fff"; g.lineWidth = 3 * k; g.stroke();
    g.fillStyle = "#fff";
    g.font = `bold ${15 * k}px system-ui, sans-serif`;
    g.fillText("HEAD", x + 20 * k, y + 5 * k);
  }
}
function setupPointer() {
  const cv = $("setup-canvas");
  const pos = (e) => {
    const r = cv.getBoundingClientRect();
    return [clamp((e.clientX - r.left) / r.width, 0, 1), clamp((e.clientY - r.top) / r.height, 0, 1)];
  };
  cv.addEventListener("pointerdown", (e) => {
    if (SU.step === 2) {
      const [x, y] = pos(e);
      SU.drag = [x, y, x, y];
      cv.setPointerCapture(e.pointerId);
    } else if (SU.step === 3) {
      SU.head = pos(e);
      setupRender();
    }
  });
  cv.addEventListener("pointermove", (e) => {
    if (!SU.drag) return;
    [SU.drag[2], SU.drag[3]] = pos(e);
    drawSetup();
  });
  cv.addEventListener("pointerup", () => {
    if (!SU.drag) return;
    const d = SU.drag;
    SU.drag = null;
    if (Math.abs(d[2] - d[0]) > 0.05 && Math.abs(d[3] - d[1]) > 0.05)
      SU.rect = [Math.min(d[0], d[2]), Math.min(d[1], d[3]), Math.max(d[0], d[2]), Math.max(d[1], d[3])];
    setupRender();
  });
  $("setup-redo").addEventListener("click", () => {
    if (SU.step === 2) SU.rect = null;
    if (SU.step === 3) SU.head = null;
    if (SU.step === 4) SU.photoAt = null;
    setupRender();
  });
  $("setup-next").addEventListener("click", () => {
    if (SU.step === 4 && !SU.photoAt) return capturePhoto();
    if (SU.step === 4) {
      const saved = store();
      saved[SU.crib] = { rect: SU.rect, head: SU.head, emptyPhotoAt: SU.photoAt };
      store(saved);
    }
    if (SU.step === 5) return show("wall");
    SU.step += 1;
    setupRender();
  });
  $("setup-reset").addEventListener("click", () => {
    store({});
    Object.assign(SU, { crib: null, step: 1, rect: null, head: null, photoAt: null });
    setupRender();
  });
}
function capturePhoto() {
  const f = $("setup-flash");
  f.classList.remove("go");
  void f.offsetWidth;
  f.classList.add("go");
  const v = $("setup-video");
  const th = $("setup-thumb");
  const g = th.getContext("2d");
  g.fillStyle = "#111";
  g.fillRect(0, 0, th.width, th.height);
  if (v.videoWidth) {
    const s = Math.min(th.width / v.videoWidth, th.height / v.videoHeight);
    const w = v.videoWidth * s, h = v.videoHeight * s;
    g.drawImage(v, (th.width - w) / 2, (th.height - h) / 2, w, h);
  }
  SU.photoAt = clockStr(S.t);
  $("setup-thumb-caption").textContent = `Empty-crib photo · saved ${SU.photoAt}`;
  setupRender();
}

// ── 每一幀 ──
function render() {
  $("clock").textContent = clockStr(S.t);
  if (S.view === "wall") {
    renderWall();
    const vids = D.cribs.map((c) => c.tileVideo);
    D.cribs.forEach((c) => syncVideo(c.tileVideo, c));
    pauseAll(vids);
  } else if (S.view === "detail") {
    renderDetail();
    syncVideo($("detail-video"), S.detail);
    pauseAll([$("detail-video")]);
  } else {
    pauseAll([]);
  }
  renderAlert();
  renderPhone();
}

function bind() {
  document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => show(b.dataset.view)));
  $("play").addEventListener("click", () => setPlaying(!S.playing));
  $("restart").addEventListener("click", () => { seek(0); S.acks = []; S.faults = []; S.notes = 0; setPlaying(true); });
  $("fault").addEventListener("click", () => {   // 模擬一次晶片卡住（離線中再按不算）
    if (faultAt(S.t) || S.t >= END) return;
    S.faults.push({ at: S.t, back: S.t + FAULT_S });
    setPlaying(true);
  });
  $("phone-toggle").addEventListener("click", () => {
    S.phoneOn = !S.phoneOn;
    $("phone").hidden = !S.phoneOn;
    $("phone-toggle").classList.toggle("on", S.phoneOn);
  });
  $("detail-back").addEventListener("click", () => show("wall"));
  $("timeline").addEventListener("click", (e) => {
    const r = $("timeline").getBoundingClientRect();
    seek(((e.clientX - r.left) / r.width) * S.detail.duration);
  });
  $("alert-ack").addEventListener("click", () => {
    const a = $("alert").dataset;
    S.acks.push({ crib: a.crib, start: Number(a.start), at: S.t });
  });
  $("alert-open").addEventListener("click", () => openDetail($("alert").dataset.crib));
  document.addEventListener("keydown", (e) => {
    if (e.code === "Space") { e.preventDefault(); setPlaying(!S.playing); }
    if (e.key === "Escape" && S.view === "detail") show("wall");
  });
  window.addEventListener("resize", () => { if (S.detail) drawTimeline(S.detail); drawSetup(); });
  setupPointer();
}

loadData().then((data) => {
  D = data;
  D.states.offline = D.states.offline || OFFLINE_STATE;   // 11/23 以前匯出的資料沒有離線燈
  D.cribs.forEach(prep);
  END = Math.max(...D.cribs.map((c) => c.duration));
  document.title = `${PRODUCT} · nurse station prototype`;
  $("product-name").textContent = PRODUCT;
  $("sample-badge").hidden = !D.sample;
  buildWall();
  bind();
  show("wall");
  $("loading").hidden = true;
  requestAnimationFrame(tick);
}).catch((err) => { $("loading").textContent = err.message; });
