/* prompt2vid 调试控制台 —— 原生 JS，轮询 2s 刷新 */
"use strict";

const $ = (sel) => document.querySelector(sel);

const state = {
  runs: [],
  selectedRun: null,
  detail: null,
  steps: [],           // 步骤目录（/api/steps）
};

const artifactCache = new Map();   // key -> {ts, data}
const lastStatus = {};             // stepId -> status，用于状态变化时清缓存
const CACHE_TTL = 5000;
let lastDetailSig = "";            // 上次渲染的详情签名，无变化不重渲染

/* ---------------- 基础工具 ---------------- */

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try { detail = (await res.json()).detail || detail; } catch (e) { /* 非 JSON 响应 */ }
    throw new Error(detail);
  }
  return res.json();
}

function toast(msg, isError = true) {
  const el = $("#toast");
  el.textContent = msg;
  el.classList.toggle("error", isError);
  el.hidden = false;
  clearTimeout(el._t);
  el._t = setTimeout(() => { el.hidden = true; }, 4000);
}

function escapeHtml(s) {
  return String(s ?? "")
    .replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function timeDiff(startIso, endIso) {
  const ms = new Date(endIso) - new Date(startIso);
  if (!Number.isFinite(ms) || ms < 0) return "";
  return ms < 60_000 ? `${(ms / 1000).toFixed(0)}s` : `${(ms / 60_000).toFixed(1)}m`;
}

function labelOf(stepId) {
  const s = state.detail?.steps.find((x) => x.id === stepId);
  return s ? s.label : stepId;
}

/* ---------------- 运行列表 ---------------- */

function renderRunList() {
  const ul = $("#run-list");
  ul.innerHTML = "";
  if (!state.runs.length) {
    ul.innerHTML = '<li class="run-item-prompt">（暂无运行）</li>';
    return;
  }
  for (const run of state.runs) {
    const li = document.createElement("li");
    li.className = "run-item" + (run.id === state.selectedRun ? " selected" : "");
    li.innerHTML = `
      <div class="run-item-head">
        <span class="run-name">${escapeHtml(run.id)}</span>
        ${run.running ? '<span class="badge running pulse">运行中</span>' : ""}
        ${run.failed ? `<span class="badge failed">${run.failed} 失败</span>` : ""}
      </div>
      <div class="run-item-prompt">${escapeHtml(run.prompt || "（无 prompt 记录）")}</div>
      <div class="run-item-progress">${run.done}/${run.total} 步完成</div>`;
    li.onclick = () => selectRun(run.id);
    ul.appendChild(li);
  }
}

function selectRun(runId) {
  if (state.selectedRun === runId) return;
  state.selectedRun = runId;
  artifactCache.clear();
  lastDetailSig = "";
  renderRunList();
  pollDetail();
}

/* ---------------- 运行详情 ---------------- */

const STATUS_MAP = {
  done: ["完成", "done"], running: ["运行中", "running"],
  failed: ["失败", "failed"], interrupted: ["中断", "interrupted"],
};

function statusBadge(status) {
  const [label, cls] = STATUS_MAP[status] || ["未开始", ""];
  const pulse = cls === "running" ? " pulse" : "";
  return `<span class="badge ${cls}${pulse}">${label}</span>`;
}

function renderDetail() {
  const d = state.detail;
  if (!d) {
    $("#run-detail").hidden = true;
    $("#empty-state").hidden = false;
    return;
  }
  // 详情无变化时跳过重渲染：2s 轮询不重建 DOM，
  // 避免按钮被替换导致点击丢失、<details> 展开状态被重置
  const sig = JSON.stringify(d);
  if (sig === lastDetailSig) return;
  lastDetailSig = sig;
  $("#empty-state").hidden = true;
  $("#run-detail").hidden = false;
  $("#run-id").textContent = d.id;
  $("#run-prompt").textContent = d.meta.prompt || "（无 prompt 记录）";
  $("#run-dir").textContent = d.dir;
  renderExecBanner(d);
  renderSteps(d);
  renderArtifacts(d);
  $("#btn-cancel").disabled = !(d.execution && d.execution.run_id === d.id);
}

function renderExecBanner(d) {
  const el = $("#exec-banner");
  if (!d.execution) { el.hidden = true; return; }
  el.hidden = false;
  if (d.execution.run_id !== d.id) {
    el.textContent = `提示：另一个运行（${d.execution.run_id}）正在执行，需等待或取消后才能操作本运行。`;
    return;
  }
  const cur = d.execution.current_step ? labelOf(d.execution.current_step) : "准备中";
  const cancelNote = d.execution.cancel_requested ? "（已请求取消，当前步骤完成后停止）" : "";
  el.textContent = `执行中：${cur}${cancelNote}`;
}

function renderSteps(d) {
  const tbody = $("#steps-body");
  tbody.innerHTML = "";
  for (const s of d.steps) {
    const tr = document.createElement("tr");
    if (s.status === "running") tr.className = "row-running";
    const dur = s.status === "running"
      ? "…"
      : s.started_at && s.finished_at ? timeDiff(s.started_at, s.finished_at) : "";
    tr.innerHTML = `
      <td>${s.index}</td>
      <td>
        <div class="step-name">${escapeHtml(s.label)}</div>
        <div class="step-id">${escapeHtml(s.id)}</div>
        ${s.error ? `<div class="step-error">${escapeHtml(s.error)}</div>` : ""}
      </td>
      <td class="mono">${escapeHtml(s.agent)}</td>
      <td><span class="kind kind-${s.kind}">${s.kind_label}</span></td>
      <td>${statusBadge(s.status)}</td>
      <td>${dur}</td>
      <td class="actions">
        <button data-act="one" data-id="${s.id}" class="btn small">▶ 此步</button>
        <button data-act="rest" data-id="${s.id}" class="btn small">▶ 到最后</button>
        ${s.kind === "media"
          ? `<button data-act="force" data-id="${s.id}" class="btn small warn">↻ 强制</button>` : ""}
        ${s.artifact_exists && s.artifact
          ? `<a class="btn small link" href="/api/runs/${d.id}/files/${s.artifact}" target="_blank">产物</a>`
          : ""}
      </td>`;
    tbody.appendChild(tr);
  }
}

/* ---------------- 产物 ---------------- */

async function fetchRunFile(d, rel) {
  const key = `${d.id}/${rel}`;
  const hit = artifactCache.get(key);
  if (hit && Date.now() - hit.ts < CACHE_TTL) return hit.data;
  const res = await fetch(`/api/runs/${d.id}/files/${rel}`);
  if (!res.ok) throw new Error(`${res.status}`);
  const data = await res.text();
  artifactCache.set(key, { ts: Date.now(), data });
  return data;
}

function artifactExists(d, stepId) {
  return Boolean(d.steps.find((s) => s.id === stepId)?.artifact_exists);
}

async function renderArtifacts(d) {
  const box = $("#artifacts");
  const parts = [];

  // 成片（视频播放）
  if (d.final_video) {
    parts.push(`
      <details class="artifact" open>
        <summary>🎬 成片 final.mp4</summary>
        <div class="artifact-body">
          <video controls preload="metadata" src="${d.final_video}"></video>
          <div><a class="btn small link" href="${d.final_video}" target="_blank">新窗口打开</a></div>
        </div>
      </details>`);
  }

  // 质检报告
  if (artifactExists(d, "inspect_quality")) {
    renderQuality(box, d); // 异步填充，不阻塞其他卡片
  }

  // 分镜表（含首帧缩略图）
  if (artifactExists(d, "plan_storyboard")) {
    renderStoryboard(box, d);
  }

  // 剧本 / 需求（纯文本查看）
  for (const [stepId, title, rel] of [
    ["write_script", "📝 剧本 script.md", "script.md"],
    ["analyze_requirement", "📋 需求 requirement.json", "requirement.json"],
  ]) {
    if (!artifactExists(d, stepId)) continue;
    fetchRunFile(d, rel)
      .then((text) => {
        let body = text;
        if (rel.endsWith(".json")) {
          try { body = JSON.stringify(JSON.parse(text), null, 2); } catch (e) { /* 保留原文 */ }
        }
        box.insertAdjacentHTML("beforeend", `
          <details class="artifact">
            <summary>${title}</summary>
            <div class="artifact-body"><pre>${escapeHtml(body)}</pre></div>
          </details>`);
      })
      .catch(() => { /* 文件读取失败时静默跳过 */ });
  }

  box.innerHTML = parts.join("");
  if (!parts.length && !artifactExists(d, "inspect_quality") && !artifactExists(d, "plan_storyboard")) {
    box.innerHTML = '<div class="hint">尚无产物。执行步骤后，这里会展示需求 / 剧本 / 分镜（含首帧缩略图）/ 成片 / 质检报告。</div>';
  }
}

async function renderStoryboard(box, d) {
  try {
    const sb = JSON.parse(await fetchRunFile(d, "storyboard.json"));
    let imgName = {};
    try {
      const manifest = JSON.parse(await fetchRunFile(d, "frames/manifest.json"));
      for (const e of manifest.shots || []) {
        if (e.status === "ok" && e.path) {
          imgName[e.id] = String(e.path).split(/[\\/]/).pop();
        }
      }
    } catch (e) { /* 首帧未生成时无缩略图 */ }
    const rows = (sb.shots || []).map((s) => `
      <tr>
        <td>#${s.id}</td>
        <td><img loading="lazy" src="/api/runs/${d.id}/files/frames/${encodeURIComponent(imgName[s.id] || `shot_${s.id}.png`)}"
             onerror="this.style.display='none'"></td>
        <td>${escapeHtml(s.narration || "")}</td>
        <td>${escapeHtml(s.visual || "")}</td>
        <td>${s.duration_seconds ?? "?"}s</td>
      </tr>`).join("");
    box.insertAdjacentHTML("beforeend", `
      <details class="artifact">
        <summary>🎞 分镜 storyboard.json <span class="sub">共 ${sb.shots?.length ?? 0} 镜 / ${escapeHtml(String(sb.total_duration_seconds ?? "?"))}s</span></summary>
        <div class="artifact-body">
          <table class="storyboard-table">
            <thead><tr><th>镜</th><th>首帧</th><th>旁白</th><th>画面</th><th>时长</th></tr></thead>
            <tbody>${rows}</tbody>
          </table>
        </div>
      </details>`);
  } catch (e) { /* 解析失败时静默跳过 */ }
}

async function renderQuality(box, d) {
  try {
    const q = JSON.parse(await fetchRunFile(d, "report/quality_report.json"));
    const checks = (q.checks || []).map((c) =>
      `<li class="quality-line ${c.passed ? "ok" : "bad"}">${c.passed ? "✔" : "✘"} ${escapeHtml(c.name)}：${escapeHtml(c.detail || "")}</li>`
    ).join("");
    box.insertAdjacentHTML("beforeend", `
      <details class="artifact" open>
        <summary>🔍 质检报告 <span class="sub">${q.passed ? "通过" : "未通过"}</span></summary>
        <div class="artifact-body">
          <div class="quality-line ${q.passed ? "ok" : "bad"}">${q.passed ? "✔" : "✘"} ${escapeHtml(q.summary || "")}</div>
          <ul>${checks}</ul>
        </div>
      </details>`);
  } catch (e) { /* 解析失败时静默跳过 */ }
}

/* ---------------- 日志 ---------------- */

async function loadLog() {
  const d = state.detail;
  if (!d) return;
  try {
    const data = await api(`/api/runs/${d.id}/log?tail=300`);
    const el = $("#log-panel");
    const nearBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 40;
    el.textContent = data.lines.join("\n") || "（暂无日志）";
    if (nearBottom) el.scrollTop = el.scrollHeight;
  } catch (e) { /* run.log 尚未创建 */ }
}

/* ---------------- 轮询 ---------------- */

function invalidateCacheOnTransition(d) {
  for (const s of d.steps) {
    const prev = lastStatus[s.id];
    if (prev === "running" && s.status && s.status !== "running") {
      artifactCache.clear();  // 某步骤刚结束，产物可能已更新
      break;
    }
  }
  for (const s of d.steps) lastStatus[s.id] = s.status;
}

async function pollDetail() {
  if (!state.selectedRun) return;
  try {
    const d = await api(`/api/runs/${state.selectedRun}`);
    invalidateCacheOnTransition(d);
    state.detail = d;
    renderDetail();
    loadLog();
  } catch (e) {
    /* 运行目录可能被清理，静默等待下一轮 */
  }
}

async function poll() {
  try {
    const data = await api("/api/runs");
    state.runs = data.runs;
    if (!state.selectedRun && state.runs.length) {
      state.selectedRun = state.runs[0].id;
    }
    renderRunList();
    await pollDetail();
  } catch (e) {
    /* 网络抖动：下一轮重试 */
  }
}

/* ---------------- 动作 ---------------- */

async function runSteps(d, ids, force = false, prompt = null) {
  const body = { steps: ids, force };
  if (prompt) body.prompt = prompt;
  await api(`/api/runs/${d.id}/steps`, { method: "POST", body: JSON.stringify(body) });
  toast(`已开始执行：${ids.map(labelOf).join("、")}`, false);
  pollDetail();
}

function restFrom(d, stepId) {
  const idx = d.steps.findIndex((s) => s.id === stepId);
  return d.steps.slice(idx).map((s) => s.id);
}

async function createRun(autoRun) {
  const prompt = $("#prompt-input").value.trim();
  if (!prompt) { toast("请先输入一句话描述"); return; }
  try {
    const res = await api("/api/runs", {
      method: "POST",
      body: JSON.stringify({ prompt, auto_run: autoRun }),
    });
    state.selectedRun = res.run_id;
    artifactCache.clear();
    if (res.warning) toast(res.warning);
    else toast(`已创建 ${res.run_id}`, false);
    poll();
  } catch (e) {
    toast(e.message);
  }
}

$("#btn-create").onclick = () => createRun(null);
$("#btn-create-creative").onclick = () =>
  createRun(["analyze_requirement", "write_script", "plan_storyboard"]);
$("#btn-create-all").onclick = () => createRun(["all"]);

$("#btn-run-all").onclick = async () => {
  const d = state.detail;
  if (!d) return;
  try { await runSteps(d, d.steps.map((s) => s.id)); }
  catch (e) { toast(e.message); }
};

$("#btn-cancel").onclick = async () => {
  const d = state.detail;
  if (!d) return;
  try {
    const res = await api(`/api/runs/${d.id}/cancel`, { method: "POST" });
    toast(res.message, false);
    pollDetail();
  } catch (e) { toast(e.message); }
};

document.addEventListener("click", async (ev) => {
  const btn = ev.target.closest("button[data-act]");
  if (!btn) return;
  const d = state.detail;
  if (!d) return;
  const id = btn.dataset.id;
  try {
    if (btn.dataset.act === "one") {
      // 需求分析需要 prompt；历史运行没有记录时现场补录
      let prompt = null;
      if (id === "analyze_requirement" && !d.meta.prompt) {
        const p = window.prompt("该运行没有记录一句话描述，请输入（将保存到 meta.json）：");
        if (!p) return;
        prompt = p;
      }
      await runSteps(d, [id], false, prompt);
    } else if (btn.dataset.act === "rest") {
      await runSteps(d, restFrom(d, id));
    } else if (btn.dataset.act === "force") {
      if (confirm(`强制重跑「${labelOf(id)}」会先删除该阶段的产物目录，确定？`)) {
        await runSteps(d, [id], true);
      }
    }
  } catch (e) {
    toast(e.message);
  }
});

/* ---------------- 启动 ---------------- */

(async function init() {
  try {
    const data = await api("/api/steps");
    state.steps = data.steps;
  } catch (e) { /* 目录接口失败不阻塞页面 */ }
  await poll();
  setInterval(poll, 2000);
})();
