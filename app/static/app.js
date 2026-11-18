/* 野生动物追踪串号校正与计数台 —— 前端逻辑 */
"use strict";

const PALETTE = [
  "#ff4d4f", "#36cfc9", "#ffa940", "#73d13d", "#9254de",
  "#40a9ff", "#f759ab", "#a0d911", "#ffc53d", "#597ef7",
  "#ff7a45", "#5cdbd3", "#b37feb", "#95de64", "#69c0ff",
];

const state = {
  sessionId: null,
  version: 0,
  nFrames: 0,
  detections: [],
  judgments: [],          // [{a, b, type}]，a/b 为框号
  params: { w_dist: 1, w_app: 10000, penalty: 15000 },
  images: [],             // HTMLImageElement
  currentFrame: 0,
  selection: [],          // 已选框号（至多 2 个）
  result: null,           // 当前展示的求解结果
  jobId: null,
  pollTimer: null,
  vectorJobId: null,      // 当前向量核查任务
  vectorToken: 0,         // 核查请求代次：改判/换会话后旧响应作废
  vectorPollTimer: null,
};

const $ = (id) => document.getElementById(id);
const canvas = $("canvas");
const ctx = canvas.getContext("2d");

const detById = (id) => state.detections.find((d) => d.id === id);
const pairKey = (a, b) => [a, b].sort().join("||");

function trajIndexOf(boxId) {
  if (!state.result || !state.result.feasible) return null;
  for (let i = 0; i < state.result.trajectories.length; i++) {
    if (state.result.trajectories[i].boxes.some((b) => b.id === boxId)) return i;
  }
  return null;
}

/* ---------------------------------------------------------------- API */

async function api(path, options = {}) {
  const resp = await fetch(path, options);
  if (!resp.ok) {
    let detail = `HTTP ${resp.status}`;
    try {
      const body = await resp.json();
      if (body.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch (_) { /* ignore */ }
    throw new Error(detail);
  }
  return resp.json();
}

function showError(targetId, err) {
  const el = $(targetId);
  el.textContent = `错误：${err.message}`;
  el.classList.remove("muted");
  el.style.color = "#ff9d97";
}

/* ------------------------------------------------------------ 会话载入 */

function loadSession(payload) {
  state.sessionId = payload.session_id;
  state.version = payload.version;
  state.nFrames = payload.n_frames;
  state.detections = payload.detections;
  state.judgments = payload.judgments;
  state.params = payload.params;
  state.currentFrame = 0;
  state.selection = [];
  state.result = null;
  state.jobId = null;
  if (state.pollTimer) { clearTimeout(state.pollTimer); state.pollTimer = null; }
  resetVectorCheck("会话已切换，需要重新核查。");

  state.images = [];
  for (let i = 0; i < state.nFrames; i++) {
    const img = new Image();
    img.src = `/api/sessions/${state.sessionId}/frames/${i}`;
    img.onload = () => { if (i === state.currentFrame) draw(); };
    state.images.push(img);
  }

  $("p-wdist").value = payload.params.w_dist;
  $("p-wapp").value = payload.params.w_app;
  $("p-penalty").value = payload.params.penalty;

  const info = $("session-info");
  info.classList.add("muted");
  info.style.color = "";
  info.innerHTML =
    `会话 <b>${payload.session_id}</b>：${payload.n_frames} 帧、` +
    `${payload.detections.length} 个检测框。` +
    (payload.description ? `<br>${payload.description}` : "");

  $("solve-btn").disabled = false;
  $("no-session").style.display = "none";
  renderAll();
}

async function uploadSession() {
  const imgs = $("img-input").files;
  const jsons = $("json-input").files;
  if (imgs.length === 0 || imgs.length > 10) {
    alert("请选择 1~10 张帧图片。");
    return;
  }
  if (jsons.length !== 1) {
    alert("请选择 1 个检测 JSON 文件。");
    return;
  }
  const fd = new FormData();
  for (const f of imgs) fd.append("images", f);
  fd.append("detections", jsons[0]);
  try {
    const payload = await api("/api/sessions", { method: "POST", body: fd });
    loadSession(payload);
  } catch (err) {
    showError("session-info", err);
  }
}

async function loadExample() {
  try {
    const payload = await api("/api/example", { method: "POST" });
    loadSession(payload);
  } catch (err) {
    showError("session-info", err);
  }
}

/* ------------------------------------------------------------ 画面渲染 */

function detsInFrame(f) {
  return state.detections.filter((d) => d.frame === f);
}

function draw() {
  const img = state.images[state.currentFrame];
  if (!img || !img.complete || img.naturalWidth === 0) return;
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  ctx.drawImage(img, 0, 0);

  for (const d of detsInFrame(state.currentFrame)) {
    const [x, y, w, h] = d.rect;
    const ti = trajIndexOf(d.id);
    const color = ti !== null ? PALETTE[ti % PALETTE.length] : "#ffe600";
    const selected = state.selection.includes(d.id);

    ctx.lineWidth = selected ? 4 : 2;
    ctx.strokeStyle = selected ? "#ffffff" : color;
    ctx.strokeRect(x, y, w, h);

    const label = `#${d.id}` + (ti !== null ? ` T${ti + 1}` : "");
    ctx.font = "bold 12px sans-serif";
    const tw = ctx.measureText(label).width;
    const ly = y >= 16 ? y - 16 : y + h + 2;
    ctx.fillStyle = "rgba(0,0,0,0.65)";
    ctx.fillRect(x, ly, tw + 6, 15);
    ctx.fillStyle = color;
    ctx.fillText(label, x + 3, ly + 12);
  }
}

function renderFrameNav() {
  const n = state.nFrames || 0;
  $("frame-indicator").textContent =
    n === 0 ? "— / —" : `第 ${state.currentFrame + 1} / ${n} 帧（本帧 ${detsInFrame(state.currentFrame).length} 框）`;
  $("prev-frame").disabled = n === 0 || state.currentFrame === 0;
  $("next-frame").disabled = n === 0 || state.currentFrame >= n - 1;
}

function switchFrame(delta) {
  const next = state.currentFrame + delta;
  if (next < 0 || next >= state.nFrames) return;
  state.currentFrame = next;
  renderFrameNav();
  draw();
}

/* ------------------------------------------------------------ 选择与标记 */

function renderSelection() {
  const info = $("selection-info");
  if (state.selection.length === 0) {
    info.textContent = "未选择框。点击画面中的检测框进行选择（可跨帧选两个）。";
  } else {
    const parts = state.selection.map((id) => {
      const d = detById(id);
      return d ? `帧 ${d.frame} 的 #${id}` : `#${id}`;
    });
    let text = `已选：${parts.join(" 与 ")}`;
    if (state.selection.length === 2) {
      const j = findJudgment(state.selection[0], state.selection[1]);
      if (j) text += `（当前判断：${j.type === "same" ? "同只" : "异只"}）`;
    }
    info.textContent = text;
  }
  const two = state.selection.length === 2;
  $("mark-same").disabled = !two;
  $("mark-diff").disabled = !two;
  $("mark-clear").disabled = !two || !findJudgment(state.selection[0], state.selection[1]);
}

function findJudgment(a, b) {
  const key = pairKey(a, b);
  return state.judgments.find((j) => pairKey(j.a, j.b) === key) || null;
}

function onCanvasClick(ev) {
  if (!state.sessionId) return;
  const rect = canvas.getBoundingClientRect();
  const sx = canvas.width / rect.width;
  const sy = canvas.height / rect.height;
  const px = (ev.clientX - rect.left) * sx;
  const py = (ev.clientY - rect.top) * sy;
  const dets = detsInFrame(state.currentFrame);
  for (let i = dets.length - 1; i >= 0; i--) {
    const [x, y, w, h] = dets[i].rect;
    if (px >= x && px <= x + w && py >= y && py <= y + h) {
      toggleSelect(dets[i].id);
      return;
    }
  }
}

function toggleSelect(id) {
  const i = state.selection.indexOf(id);
  if (i >= 0) {
    state.selection.splice(i, 1);
  } else {
    if (state.selection.length >= 2) state.selection.shift();
    state.selection.push(id);
  }
  renderSelection();
  draw();
}

async function setJudgment(type) {
  if (state.selection.length !== 2) return;
  const [a, b] = state.selection;
  const key = pairKey(a, b);
  const rest = state.judgments.filter((j) => pairKey(j.a, j.b) !== key);
  const next = type ? [...rest, { a, b, type }] : rest;
  invalidateResult(); // 修改即撤下旧结果
  resetVectorCheck("身份判断已修改，旧核查建议已撤下。");
  try {
    const resp = await api(`/api/sessions/${state.sessionId}/judgments`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ judgments: next }),
    });
    state.version = resp.version;
    state.judgments = next;
    renderJudgments();
    updateVectorAvailability();
    renderSelection();
    draw();
  } catch (err) {
    alert(`保存判断失败：${err.message}`);
  }
}

async function removeJudgment(a, b) {
  const key = pairKey(a, b);
  const next = state.judgments.filter((j) => pairKey(j.a, j.b) !== key);
  invalidateResult();
  resetVectorCheck("身份判断已修改，旧核查建议已撤下。");
  try {
    const resp = await api(`/api/sessions/${state.sessionId}/judgments`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ judgments: next }),
    });
    state.version = resp.version;
    state.judgments = next;
    renderJudgments();
    updateVectorAvailability();
    renderSelection();
    draw();
  } catch (err) {
    alert(`删除判断失败：${err.message}`);
  }
}

function renderJudgments() {
  const ul = $("judgment-list");
  ul.innerHTML = "";
  $("judgment-count").textContent = `（${state.judgments.length} 条）`;
  for (const j of state.judgments) {
    const da = detById(j.a);
    const db = detById(j.b);
    const li = document.createElement("li");
    const tag = document.createElement("span");
    tag.className = `tag ${j.type === "same" ? "same" : "diff"}`;
    tag.textContent = j.type === "same" ? "同只" : "异只";
    const text = document.createElement("span");
    text.textContent = `帧${da ? da.frame : "?"} #${j.a} ↔ 帧${db ? db.frame : "?"} #${j.b}`;
    const del = document.createElement("button");
    del.textContent = "删除";
    del.onclick = () => removeJudgment(j.a, j.b);
    li.append(tag, text, del);
    ul.appendChild(li);
  }
}

/* ------------------------------------------------------------ 参数 */

async function applyParams() {
  if (!state.sessionId) return;
  const wDist = parseFloat($("p-wdist").value);
  const wApp = parseFloat($("p-wapp").value);
  const penalty = parseFloat($("p-penalty").value);
  if (![wDist, wApp, penalty].every((v) => Number.isFinite(v) && v >= 0)) {
    alert("参数必须是有限非负数。");
    return;
  }
  invalidateResult(); // 修改即撤下旧结果
  resetVectorCheck("代价参数已修改，旧核查建议已撤下。");
  try {
    const resp = await api(`/api/sessions/${state.sessionId}/params`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ w_dist: wDist, w_app: wApp, penalty }),
    });
    state.version = resp.version;
    state.params = { w_dist: wDist, w_app: wApp, penalty };
  } catch (err) {
    alert(`保存参数失败：${err.message}`);
  }
}

/* ------------------------------------------------------------ 向量核查 */

function resetVectorCheck(message) {
  // 代次 +1：任何已在途的核查响应/轮询都会因代次不符被丢弃
  const hadOutput = state.vectorJobId !== null ||
    $("vector-result").childElementCount > 0;
  state.vectorToken += 1;
  state.vectorJobId = null;
  if (state.vectorPollTimer) {
    clearTimeout(state.vectorPollTimer);
    state.vectorPollTimer = null;
  }
  $("vector-result").innerHTML = "";
  const status = $("vector-status");
  status.classList.add("muted");
  status.style.color = "";
  status.textContent = message && hadOutput ? message : "需要先标记同只与异只判断。";
  updateVectorAvailability();
}

function updateVectorAvailability() {
  const same = state.judgments.filter((j) => j.type === "same").length;
  const diff = state.judgments.filter((j) => j.type === "different").length;
  const btn = $("vector-btn");
  btn.disabled = !state.sessionId;
  if (!state.sessionId) return;
  if (same === 0 || diff === 0) {
    const miss = same === 0 && diff === 0 ? "同只与异只" : (same === 0 ? "同只" : "异只");
    btn.disabled = true;
    $("vector-status").textContent = `核查需要同时包含同只与异只判断；当前缺少${miss}判断。`;
  }
}

async function runVectorCheck() {
  if (!state.sessionId) return;
  const status = $("vector-status");
  status.classList.add("muted");
  status.style.color = "";
  $("vector-result").innerHTML = "";
  const token = ++state.vectorToken;
  try {
    const resp = await api(`/api/sessions/${state.sessionId}/vector-check`, { method: "POST" });
    if (token !== state.vectorToken) return; // 期间已改判或换会话
    state.vectorJobId = resp.job_id;
    status.textContent = `核查中（输入版本 v${resp.version}）…`;
    $("vector-btn").disabled = true;
    pollVectorJob(resp.job_id, token);
  } catch (err) {
    if (token !== state.vectorToken) return;
    showError("vector-status", err);
    $("vector-btn").disabled = false;
  }
}

async function pollVectorJob(jobId, token) {
  try {
    const job = await api(`/api/jobs/${jobId}`);
    if (token !== state.vectorToken || jobId !== state.vectorJobId) return; // 迟到结果丢弃
    if (job.status === "running") {
      state.vectorPollTimer = setTimeout(() => pollVectorJob(jobId, token), 300);
      return;
    }
    state.vectorJobId = null;
    $("vector-btn").disabled = false;
    const status = $("vector-status");
    if (job.status === "error") {
      status.textContent = job.error || "核查失败。";
      status.style.color = "#ff9d97";
      return;
    }
    if (job.version !== state.version) {
      status.textContent =
        `核查完成，但基于旧输入（v${job.version}，当前 v${state.version}），建议已作废。`;
      status.style.color = "#ffd887";
      return;
    }
    status.textContent = `核查完成（输入版本 v${job.version}）。`;
    status.style.color = "";
    renderVectorResult(job.result);
  } catch (err) {
    if (token !== state.vectorToken) return;
    showError("vector-status", err);
    $("vector-btn").disabled = false;
  }
}

function simLine(j) {
  return `判断 ${j.a} ↔ ${j.b}（${j.type === "same" ? "同只" : "异只"}）：` +
    `余弦相似度 ${Number(j.similarity).toFixed(4)}`;
}

function renderVectorResult(r) {
  const box = $("vector-result");
  box.innerHTML = "";
  if (!r) return;

  const scope = document.createElement("div");
  scope.className = "muted small";
  if (r.scope) {
    scope.textContent =
      `本次涉及 ${r.scope.boxes} 个框（上限 ${r.scope.max_boxes}）、` +
      `${r.scope.judgments} 条判断（上限 ${r.scope.max_judgments}）。`;
  }
  box.appendChild(scope);

  if (!r.available) {
    const b = document.createElement("div");
    b.className = "banner stale";
    b.textContent = r.message || "不提供重算建议。";
    box.appendChild(b);
    if (r.judgments && r.judgments.length) {
      const ul = document.createElement("ul");
      ul.className = "sim-list";
      for (const j of r.judgments) {
        const li = document.createElement("li");
        li.textContent = simLine(j);
        ul.appendChild(li);
      }
      box.appendChild(ul);
    }
    return;
  }

  const banner = document.createElement("div");
  if (r.recommendation === "none") {
    banner.className = "banner ok";
    banner.textContent = r.message;
    box.appendChild(banner);
  } else if (r.recommendation === "ambiguous") {
    banner.className = "banner stale";
    banner.textContent = r.message;
    box.appendChild(banner);
  } else {
    banner.className = "banner stale";
    banner.textContent = r.message;
    box.appendChild(banner);
  }

  const detail = document.createElement("div");
  detail.className = "vector-detail";
  if (r.threshold !== null && r.threshold !== undefined) {
    detail.innerHTML +=
      `<div>可用阈值：<b>${Number(r.threshold).toFixed(4)}</b>` +
      `（区间 ${Number(r.threshold_range[0]).toFixed(4)} ~ ${Number(r.threshold_range[1]).toFixed(4)}，` +
      `间隔 ${Number(r.separation).toFixed(4)}，要求 ≥ ${Number(r.margin_required).toFixed(2)}）</div>`;
  }
  if (r.recommendation === "recompute") {
    detail.innerHTML +=
      `<div>待重算外观向量的框号：<b>${r.boxes_to_recompute.join("、")}</b>（${r.boxes_to_recompute.length} 个）</div>`;
    const dropped = r.candidate_sets[0].dropped_judgments || [];
    if (dropped.length) {
      detail.innerHTML +=
        `<div>与待重算框相关、被排除的判断：` +
        dropped.map((j) =>
          `${j.a} ↔ ${j.b}（${j.type === "same" ? "同只" : "异只"}，相似度 ${Number(j.similarity).toFixed(4)}）`
        ).join("；") + "</div>";
    }
  }
  box.appendChild(detail);

  if (!r.unique && r.candidate_sets.length > 1) {
    const h = document.createElement("h3");
    h.textContent = `不唯一的候选框集（共 ${r.candidate_sets.length} 组，均为最少重算数）`;
    box.appendChild(h);
    r.candidate_sets.forEach((c, i) => {
      const d = document.createElement("div");
      d.className = "cand";
      d.textContent =
        `候选 ${i + 1}：重算 {${c.boxes_to_recompute.join("、")}}；` +
        `阈值 ${Number(c.threshold).toFixed(4)}` +
        `（区间 ${Number(c.threshold_range[0]).toFixed(4)} ~ ${Number(c.threshold_range[1]).toFixed(4)}）` +
        `，保留同只 ${c.retained_same} 条 / 异只 ${c.retained_different} 条判断。`;
      box.appendChild(d);
    });
  }

  if (r.judgments && r.judgments.length) {
    const h = document.createElement("h3");
    h.textContent = "相关判断的余弦相似度";
    box.appendChild(h);
    const ul = document.createElement("ul");
    ul.className = "sim-list";
    for (const j of r.judgments) {
      const li = document.createElement("li");
      li.textContent = simLine(j);
      li.style.color = j.type === "same" ? "#7ee2a0" : "#ff9d97";
      ul.appendChild(li);
    }
    box.appendChild(ul);
  }
  if (r.recommendation !== "none") {
    const note = document.createElement("div");
    note.className = "muted small";
    note.textContent = "提示：核查只依据相似度与人工判断，不认定向量确实有误，也不会自动修改数据。";
    box.appendChild(note);
  }
}

/* ------------------------------------------------------------ 求解 */

function invalidateResult() {
  if (state.result !== null) {
    state.result = null;
    const banner = $("result-banner");
    banner.className = "banner stale";
    banner.textContent = "输入已修改，旧结果已撤下。请重新计算。";
    $("result-summary").innerHTML = "";
    $("trajectory-list").innerHTML = "";
    draw();
  }
}

async function solve() {
  if (!state.sessionId) return;
  const status = $("solve-status");
  status.classList.add("muted");
  status.style.color = "";
  try {
    const resp = await api(`/api/sessions/${state.sessionId}/solve`, { method: "POST" });
    state.jobId = resp.job_id;
    status.textContent = `计算中（输入版本 v${resp.version}）… 可继续修改判断与参数。`;
    if (state.pollTimer) clearTimeout(state.pollTimer);
    pollJob(resp.job_id);
  } catch (err) {
    showError("solve-status", err);
  }
}

async function pollJob(jobId) {
  try {
    const job = await api(`/api/jobs/${jobId}`);
    if (jobId !== state.jobId) return; // 已被更新的任务取代
    if (job.status === "running") {
      state.pollTimer = setTimeout(() => pollJob(jobId), 400);
      return;
    }
    const status = $("solve-status");
    if (job.status === "error") {
      status.textContent = job.error || "求解失败。";
      status.style.color = "#ff9d97";
      return;
    }
    if (job.version !== state.version) {
      // 迟到结果：输入在计算期间已变化，丢弃不展示
      status.textContent =
        `任务完成，但其基于旧输入（v${job.version}，当前 v${state.version}），结果已作废。请重新计算。`;
      status.style.color = "#ffd887";
      const banner = $("result-banner");
      banner.className = "banner stale";
      banner.textContent = "存在基于旧输入的迟到结果，已自动丢弃，未覆盖当前输入。";
      return;
    }
    status.textContent = `计算完成（输入版本 v${job.version}）。`;
    status.style.color = "";
    state.result = job.result;
    renderResult();
    draw();
  } catch (err) {
    showError("solve-status", err);
  }
}

function renderResult() {
  const banner = $("result-banner");
  const summary = $("result-summary");
  const list = $("trajectory-list");
  const r = state.result;
  if (!r) return;
  if (!r.feasible) {
    banner.className = "banner err";
    banner.textContent = `无解：${r.message || "约束冲突。"}`;
    summary.innerHTML = "<b>不提供动物计数。</b>";
    list.innerHTML = "";
    return;
  }
  banner.className = "banner ok";
  banner.textContent = r.message || "求解成功。";
  summary.innerHTML =
    `<div><span class="count">${r.animal_count}</span>只动物（按轨迹条数计）</div>` +
    `<div>轨迹条数：${r.n_trajectories}</div>` +
    `<div>总代价：${r.total_cost.toFixed(3)}` +
    `（关联代价 ${r.assoc_cost.toFixed(3)} + 轨迹罚分 ${r.penalty_cost.toFixed(3)}）</div>`;
  list.innerHTML = "";
  r.trajectories.forEach((t, i) => {
    const div = document.createElement("div");
    div.className = "traj";
    const sw = document.createElement("span");
    sw.className = "swatch";
    sw.style.background = PALETTE[i % PALETTE.length];
    const boxes = document.createElement("span");
    boxes.className = "boxes";
    boxes.textContent = `轨迹 T${i + 1}：` + t.boxes.map((b) => `帧${b.frame} #${b.id}`).join(" → ");
    const len = document.createElement("span");
    len.className = "len";
    len.textContent = `（${t.length} 框）`;
    div.append(sw, boxes, len);
    list.appendChild(div);
  });
}

/* ------------------------------------------------------------ 汇总渲染 */

function renderAll() {
  renderFrameNav();
  renderSelection();
  renderJudgments();
  draw();
  const banner = $("result-banner");
  banner.className = "banner muted";
  banner.textContent = "尚无结果。";
  $("result-summary").innerHTML = "";
  $("trajectory-list").innerHTML = "";
  updateVectorAvailability();
}

/* ------------------------------------------------------------ 事件绑定 */

$("upload-btn").addEventListener("click", uploadSession);
$("example-btn").addEventListener("click", loadExample);
$("prev-frame").addEventListener("click", () => switchFrame(-1));
$("next-frame").addEventListener("click", () => switchFrame(1));
$("mark-same").addEventListener("click", () => setJudgment("same"));
$("mark-diff").addEventListener("click", () => setJudgment("different"));
$("mark-clear").addEventListener("click", () => setJudgment(null));
$("solve-btn").addEventListener("click", solve);
$("vector-btn").addEventListener("click", runVectorCheck);
canvas.addEventListener("click", onCanvasClick);
for (const id of ["p-wdist", "p-wapp", "p-penalty"]) {
  $(id).addEventListener("change", applyParams);
}
document.addEventListener("keydown", (ev) => {
  if (ev.target.tagName === "INPUT") return;
  if (ev.key === "ArrowLeft") switchFrame(-1);
  if (ev.key === "ArrowRight") switchFrame(1);
});
