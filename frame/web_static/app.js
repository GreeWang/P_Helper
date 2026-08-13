const state = {
  token: "", papers: [], jobs: [], sessions: [], selectedPaper: null,
  sessionId: null, importMode: "upload", poller: null, scopeSelection: null, scopeAll: true,
  summaryRequest: 0, askRequest: 0, sessionRequest: 0, asking: false,
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

document.addEventListener("DOMContentLoaded", async () => {
  bindNavigation();
  bindImport();
  bindAsk();
  bindLibrary();
  await bootstrap();
  if (window.lucide) window.lucide.createIcons();
});

async function bootstrap() {
  try {
    const data = await request("/api/bootstrap");
    state.token = data.csrf_token;
    const configured = Object.values(data.configured).every(Boolean);
    const status = $("#model-status");
    status.classList.toggle("configured", configured);
    status.querySelector("span:last-child").textContent = configured ? "模型已配置" : "模型未配置";
    await Promise.all([refreshLibrary(), refreshSessions()]);
  } catch (error) {
    toast(error.message, true);
  }
}

function bindNavigation() {
  const titles = {
    library: ["PAPER LIBRARY", "论文库"],
    ask: ["EVIDENCE Q&A", "论文问答"],
    sessions: ["CONVERSATIONS", "历史会话"],
  };
  $$(".nav-item").forEach((button) => button.addEventListener("click", () => {
    const view = button.dataset.view;
    $$(".nav-item").forEach((item) => {
      const active = item === button;
      item.classList.toggle("active", active);
      if (active) item.setAttribute("aria-current", "page");
      else item.removeAttribute("aria-current");
    });
    $$(".view").forEach((item) => item.classList.toggle("active", item.id === `${view}-view`));
    $("#view-eyebrow").textContent = titles[view][0];
    $("#view-title").textContent = titles[view][1];
    $("#paper-search-box").hidden = view !== "library";
    if (view === "library") $("#summary-pane").classList.remove("mobile-open");
    if (view !== "ask") closeScope();
    if (view === "sessions") refreshSessions().catch((error) => toast(error.message, true));
    if (window.lucide) window.lucide.createIcons();
  }));
}

function bindLibrary() {
  $("#refresh-button").addEventListener("click", () => refreshLibrary(true).catch(() => {}));
  $$("[data-paper-search]").forEach((input) => input.addEventListener("input", (event) => {
    $$("[data-paper-search]").forEach((peer) => { if (peer !== event.target) peer.value = event.target.value; });
    renderPapers();
  }));
  $("#summary-pane").addEventListener("click", (event) => {
    if (event.currentTarget.classList.contains("mobile-open") && event.target === event.currentTarget) {
      event.currentTarget.classList.remove("mobile-open");
    }
  });
}

async function refreshLibrary(notify = false) {
  try {
    const data = await request("/api/library");
    const previousJobs = new Map(state.jobs.map((job) => [job.id, job]));
    const wasRunning = state.jobs.some((job) => ["queued", "running"].includes(job.status));
    state.papers = data.papers;
    state.jobs = data.jobs;
    renderMetrics();
    renderJobs();
    renderPapers();
    renderScope();
    const running = state.jobs.some((job) => ["queued", "running"].includes(job.status));
    if (running && !state.poller) state.poller = setInterval(() => refreshLibrary().catch(() => {}), 2500);
    if (!running && state.poller) { clearInterval(state.poller); state.poller = null; }
    if (wasRunning && !running) {
      const finished = state.jobs.filter((job) => ["queued", "running"].includes(previousJobs.get(job.id)?.status) && !["queued", "running"].includes(job.status));
      const failed = finished.filter((job) => job.status === "failed");
      toast(failed.length ? `${failed.length} 个任务失败，请查看任务详情` : "处理任务已完成", Boolean(failed.length));
    }
    if (notify) toast("论文库已刷新");
  } catch (error) {
    if (notify) toast(`刷新失败：${error.message}`, true);
    throw error;
  }
}

function renderMetrics() {
  $("#paper-count").textContent = state.papers.length;
  $("#summary-count").textContent = state.papers.filter((paper) => paper.summary_status === "success").length;
  $("#index-count").textContent = state.papers.filter((paper) => paper.index_status === "success").length;
  $("#job-count").textContent = state.jobs.filter((job) => ["queued", "running"].includes(job.status)).length;
}

function renderJobs() {
  const panel = $("#job-panel");
  const visible = state.jobs.filter((job) => job.status !== "success").slice(0, 5);
  panel.replaceChildren();
  panel.hidden = !visible.length;
  visible.forEach((job) => {
    const row = el("div", `job-row ${job.status}`);
    const status = { queued: "排队中", running: "处理中", failed: "失败" }[job.status] || job.status;
    row.append(el("span", "job-status", status), el("span", "job-source", job.source), el("span", "job-error", job.error || ""));
    panel.append(row);
  });
}

function renderPapers() {
  const root = $("#paper-list");
  root.replaceChildren();
  const query = $("#paper-search").value.trim().toLowerCase();
  const papers = state.papers.filter((paper) => paper.source_path.toLowerCase().includes(query));
  $("#paper-filter-count").textContent = `${papers.length} 篇`;
  if (!papers.length) {
    root.append(emptyNode("book-open", query ? "没有匹配论文" : "还没有论文", query ? "换一个关键词试试。" : "导入 PDF 后会显示在这里。"));
    icons();
    return;
  }
  papers.forEach((paper) => {
    const button = el("button", "paper-row");
    button.classList.toggle("active", paper.fingerprint === state.selectedPaper);
    button.type = "button";
    const thumb = paper.image_url ? el("img", "paper-thumb") : el("div", "paper-thumb placeholder");
    if (paper.image_url) { thumb.src = paper.image_url; thumb.alt = ""; }
    else thumb.append(icon("file-text"));
    const copy = el("span", "paper-copy");
    const title = el("span", "paper-title", paper.source_path.replace(/\.pdf$/i, ""));
    const meta = el("span", "paper-meta");
    meta.append(statusBadge(paper.summary_status, "摘要"), statusBadge(paper.index_status, "索引"));
    copy.append(title, meta);
    button.append(thumb, copy, icon("chevron-right"));
    button.addEventListener("click", () => selectPaper(paper));
    root.append(button);
  });
  icons();
}

function statusBadge(status, label) {
  const names = { success: `${label}就绪`, failed: `${label}失败`, processing: "处理中", not_run: "未建立" };
  const badge = el("span", `badge ${status === "failed" ? "failed" : status === "success" ? "" : "pending"}`, names[status] || status);
  return badge;
}

async function selectPaper(paper) {
  if (paper.summary_status !== "success") {
    toast(paper.error || "这篇论文的摘要尚不可用", true);
    return;
  }
  const requestId = ++state.summaryRequest;
  state.selectedPaper = paper.fingerprint;
  renderPapers();
  const pane = $("#summary-pane");
  pane.replaceChildren(summaryBackButton(), emptyNode("loader-circle", "读取摘要", ""));
  pane.classList.add("mobile-open");
  icons();
  try {
    const data = await request(`/api/papers/${paper.fingerprint}/summary`);
    if (requestId !== state.summaryRequest || state.selectedPaper !== paper.fingerprint) return;
    pane.replaceChildren(summaryBackButton());
    const header = el("div", "summary-header");
    const copy = el("div");
    copy.append(el("p", "eyebrow", "PAPER SUMMARY"), el("h2", "", paper.source_path.replace(/\.pdf$/i, "")), el("p", "", paper.source_path));
    header.append(copy);
    if (paper.image_url) { const img = el("img", "summary-cover"); img.src = paper.image_url; img.alt = "代表图片"; header.append(img); }
    const body = el("article", "markdown-body");
    renderMarkdown(data.markdown, body);
    pane.append(header, body);
  } catch (error) {
    if (requestId !== state.summaryRequest) return;
    pane.replaceChildren(summaryBackButton(), emptyNode("circle-alert", "无法读取摘要", error.message));
  }
  icons();
}

function renderMarkdown(markdown, root) {
  const lines = markdown.split(/\r?\n/);
  let list = null;
  for (let index = 0; index < lines.length; index += 1) {
    const raw = lines[index];
    const line = raw.trim();
    if (!line) { list = null; continue; }
    const image = markdownImage(line);
    if (image) {
      const node = el("img", "markdown-image");
      node.src = image.url; node.alt = image.alt || "代表图片"; node.loading = "lazy";
      node.addEventListener("error", () => node.remove(), { once: true });
      root.append(node); list = null; continue;
    }
    if (isTableRow(line) && isTableDivider(lines[index + 1]?.trim())) {
      const rows = [splitTableRow(line)];
      index += 2;
      while (index < lines.length && isTableRow(lines[index].trim())) {
        rows.push(splitTableRow(lines[index].trim()));
        index += 1;
      }
      index -= 1;
      root.append(renderTable(rows));
      list = null;
      continue;
    }
    const heading = line.match(/^(#{1,3})\s+(.+)$/);
    if (heading) { list = null; root.append(el(`h${heading[1].length}`, "", stripInline(heading[2]))); continue; }
    const listMatch = line.match(/^([-*]|\d+\.)\s+(.+)$/);
    if (listMatch) {
      const tag = listMatch[1].endsWith(".") ? "ol" : "ul";
      if (!list || list.tagName.toLowerCase() !== tag) { list = el(tag); root.append(list); }
      list.append(el("li", "", stripInline(listMatch[2])));
      continue;
    }
    list = null;
    root.append(el("p", "", stripInline(line)));
  }
}

function isTableRow(line) { return Boolean(line?.startsWith("|") && line.endsWith("|") && line.split("|").length >= 4); }
function isTableDivider(line) { return Boolean(line && isTableRow(line) && splitTableRow(line).every((cell) => /^:?-{3,}:?$/.test(cell))); }
function splitTableRow(line) { return line.slice(1, -1).split("|").map((cell) => cell.trim()); }
function renderTable(rows) { const wrapper = el("div", "table-scroll"); const table = el("table"); const head = el("thead"); const headRow = el("tr"); rows[0].forEach((cell) => headRow.append(el("th", "", stripInline(cell)))); head.append(headRow); const body = el("tbody"); rows.slice(1).forEach((values) => { const row = el("tr"); values.forEach((cell) => row.append(el("td", "", stripInline(cell)))); body.append(row); }); table.append(head, body); wrapper.append(table); return wrapper; }
function markdownImage(line) { const match = line.match(/^!\[([^\]]*)\]\(([^)]+)\)$/); if (!match) return null; const path = match[2].replace(/^(\.\.\/)+/, ""); const parts = path.split("/"); if (parts[0] !== "images" || parts.some((part) => !part || part === "." || part === "..")) return null; return { alt: stripInline(match[1]), url: `/artifacts/${parts.map(encodeURIComponent).join("/")}` }; }
function stripInline(value) { return value.replace(/\*\*|__|`/g, ""); }

function bindImport() {
  const dialog = $("#import-dialog");
  $("#import-button").addEventListener("click", () => dialog.showModal());
  const modeButtons = $$("[data-import-mode]");
  modeButtons.forEach((button, index) => {
    button.addEventListener("click", () => {
      state.importMode = button.dataset.importMode;
      modeButtons.forEach((item) => { const active = item === button; item.classList.toggle("active", active); item.setAttribute("aria-selected", String(active)); item.tabIndex = active ? 0 : -1; });
      $$(".import-panel").forEach((panel) => { const active = panel.id === `${state.importMode}-panel`; panel.classList.toggle("active", active); panel.hidden = !active; });
    });
    button.addEventListener("keydown", (event) => {
      if (!["ArrowLeft", "ArrowRight"].includes(event.key)) return;
      event.preventDefault();
      const offset = event.key === "ArrowRight" ? 1 : -1;
      const target = modeButtons[(index + offset + modeButtons.length) % modeButtons.length];
      target.click(); target.focus();
    });
  });
  $("#pdf-files").addEventListener("change", showSelectedFiles);
  const drop = $("#drop-zone");
  ["dragenter", "dragover"].forEach((type) => drop.addEventListener(type, (event) => { event.preventDefault(); drop.classList.add("dragging"); }));
  ["dragleave", "drop"].forEach((type) => drop.addEventListener(type, (event) => { event.preventDefault(); drop.classList.remove("dragging"); }));
  drop.addEventListener("drop", (event) => { $("#pdf-files").files = event.dataTransfer.files; showSelectedFiles(); });
  $("#import-form").addEventListener("submit", submitImport);
  document.addEventListener("keydown", (event) => {
    if (!$("#scope-panel").classList.contains("mobile-open")) return;
    if (event.key === "Escape") { event.preventDefault(); closeScope(true); return; }
    if (event.key !== "Tab") return;
    const focusable = $$("#scope-panel button:not(:disabled), #scope-panel input:not(:disabled)");
    if (!focusable.length) return;
    const first = focusable[0]; const last = focusable[focusable.length - 1];
    if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
    else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
  });
}

function showSelectedFiles() {
  const files = [...$("#pdf-files").files];
  const problem = uploadProblem(files);
  $("#file-selection").textContent = problem || (files.length ? `已选择 ${files.length} 个 PDF，共 ${formatBytes(files.reduce((sum, file) => sum + file.size, 0))}` : "尚未选择文件");
  $("#file-selection").classList.toggle("invalid", Boolean(problem));
}

async function submitImport(event) {
  event.preventDefault();
  const submit = event.submitter;
  if (submit?.value === "cancel") { $("#import-dialog").close(); return; }
  submit.disabled = true;
  try {
    const language = $("#import-language").value;
    const workers = $("#import-workers").value;
    const force = $("#import-force").checked;
    let response;
    if (state.importMode === "upload") {
      const files = [...$("#pdf-files").files];
      if (!files.length) throw new Error("请先选择 PDF 文件");
      const problem = uploadProblem(files);
      if (problem) throw new Error(problem);
      const data = new FormData();
      files.forEach((file) => data.append("files", file));
      data.append("language", language); data.append("workers", workers); data.append("force", force);
      response = await request("/api/jobs/upload", { method: "POST", body: data, token: true });
    } else {
      const input = $("#input-path").value.trim();
      if (!input) throw new Error("请输入 PDF 文件或目录路径");
      response = await request("/api/jobs/path", { method: "POST", token: true,
        body: JSON.stringify({ input, language, workers: Number(workers), force }), json: true });
    }
    $("#import-dialog").close();
    resetImportForm();
    toast(`任务 ${response.id} 已加入队列`);
    await refreshLibrary();
  } catch (error) { toast(error.message, true); }
  finally { submit.disabled = false; }
}

function bindAsk() {
  $("#top-k").addEventListener("input", (event) => { $("#top-k-value").textContent = event.target.value; });
  $("#all-papers").addEventListener("change", (event) => {
    $$("#scope-list input").forEach((input) => { input.checked = event.target.checked; });
    syncScopeSelection(event.target.checked);
  });
  $("#clear-scope").addEventListener("click", () => {
    $("#all-papers").checked = false; $$("#scope-list input").forEach((input) => { input.checked = false; });
    syncScopeSelection(false);
  });
  $("#new-chat").addEventListener("click", newChat);
  $("#scope-toggle").addEventListener("click", openScope);
  $("#close-scope").addEventListener("click", () => closeScope(true));
  $("#scope-backdrop").addEventListener("click", () => closeScope(true));
  bindPromptButtons(document);
  $("#ask-form").addEventListener("submit", submitQuestion);
  $("#question").addEventListener("input", (event) => {
    event.target.style.height = "auto"; event.target.style.height = `${Math.min(event.target.scrollHeight, 150)}px`;
  });
  $("#question").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); $("#ask-form").requestSubmit(); }
  });
}

function renderScope() {
  const indexed = state.papers.filter((paper) => paper.index_status === "success");
  const selected = state.scopeSelection === null || (state.scopeAll && !state.sessionId)
    ? new Set(indexed.map((paper) => paper.fingerprint)) : state.scopeSelection;
  const root = $("#scope-list"); root.replaceChildren();
  indexed.forEach((paper) => {
    const label = el("label", "scope-option");
    const input = el("input"); input.type = "checkbox"; input.checked = selected.has(paper.fingerprint); input.value = paper.fingerprint;
    input.addEventListener("change", syncScopeSelection);
    label.append(input, el("span", "", paper.source_path)); root.append(label);
  });
  syncScopeSelection();
  setConversationLocked(Boolean(state.sessionId));
}

async function submitQuestion(event) {
  event.preventDefault();
  const input = $("#question"); const question = input.value.trim();
  if (!question || state.asking) return;
  const requestId = ++state.askRequest;
  const sessionAtStart = state.sessionId;
  const selected = $$("#scope-list input:checked").map((item) => item.value);
  if (!state.sessionId && !selected.length) { toast("至少选择一篇论文", true); return; }
  addMessage("user", question); input.value = ""; input.style.height = "auto";
  const loading = addMessage("assistant loading", "正在检索论文证据…");
  const button = $(".send-button"); button.disabled = true;
  state.asking = true;
  input.disabled = true;
  $("#new-chat").disabled = true;
  try {
    const data = await request("/api/ask", { method: "POST", token: true, json: true,
      body: JSON.stringify({ question, session_id: state.sessionId,
        papers: state.sessionId ? [] : selected, top_k: state.sessionId ? null : Number($("#top-k").value) }) });
    if (requestId !== state.askRequest || state.sessionId !== sessionAtStart) return;
    state.sessionId = data.session_id;
    $("#conversation-label").textContent = `会话 ${state.sessionId}`;
    setConversationLocked(true);
    loading.querySelector(".message-body").textContent = data.answer;
    loading.classList.remove("loading");
    refreshSessions().catch((error) => toast(`会话列表刷新失败：${error.message}`, true));
  } catch (error) {
    if (requestId !== state.askRequest) return;
    loading.querySelector(".message-body").textContent = `未能完成回答：${error.message}`;
    loading.classList.remove("loading");
  } finally {
    if (requestId === state.askRequest) {
      state.asking = false; button.disabled = false; input.disabled = false;
      $("#new-chat").disabled = false; input.focus();
    }
  }
}

function addMessage(role, text) {
  $(".welcome-message")?.remove();
  const node = el("div", `message ${role}`);
  node.append(el("div", "message-label", role.startsWith("user") ? "你" : "P-Helper"), el("div", "message-body", text));
  $("#messages").append(node); node.scrollIntoView({ behavior: "smooth", block: "end" });
  return node;
}

function newChat() {
  if (state.asking) return;
  state.askRequest += 1;
  state.sessionRequest += 1;
  state.sessionId = null;
  state.scopeSelection = null;
  state.scopeAll = true;
  renderScope();
  setConversationLocked(false);
  $("#conversation-label").textContent = "新会话";
  $("#messages").replaceChildren();
  $("#messages").append(welcomeMessage()); icons();
}

async function refreshSessions() {
  const data = await request("/api/sessions"); state.sessions = data.sessions; renderSessions();
}

function renderSessions() {
  const root = $("#session-list"); root.replaceChildren();
  $("#session-count").textContent = `${state.sessions.length} 个会话`;
  if (!state.sessions.length) { root.append(emptyNode("history", "暂无历史会话", "完成一次问答后，会话会保存在这里。")); icons(); return; }
  state.sessions.forEach((session) => {
    const row = el("div", "session-row"); const open = el("button"); open.type = "button";
    open.append(el("span", "session-id", session.id), el("span", "session-meta", `${session.language.toUpperCase()} · ${session.fingerprints.length} 篇论文 · Top ${session.top_k}`));
    open.addEventListener("click", () => openSession(session.id));
    const time = el("span", "session-meta", formatDate(session.last_used_at));
    const remove = el("button", "icon-button delete-session"); remove.type = "button"; remove.title = "删除会话"; remove.setAttribute("aria-label", `删除会话 ${session.id}`); remove.append(icon("trash-2"));
    remove.addEventListener("click", () => deleteSession(session.id)); row.append(open, time, remove); root.append(row);
  }); icons();
}

async function openSession(sessionId) {
  if (state.asking) { toast("请等待当前回答完成", true); return; }
  const requestId = ++state.sessionRequest;
  try {
    const data = await request(`/api/sessions/${sessionId}/turns`);
    if (requestId !== state.sessionRequest) return;
    state.sessionId = sessionId; $("#conversation-label").textContent = `会话 ${sessionId}`;
    $("#top-k").value = data.session.top_k;
    $("#top-k-value").textContent = data.session.top_k;
    state.scopeSelection = new Set(data.session.fingerprints);
    $$("#scope-list input").forEach((input) => { input.checked = state.scopeSelection.has(input.value); });
    syncScopeSelection();
    setConversationLocked(true);
    $("#messages").replaceChildren();
    data.turns.forEach(([question, answer]) => { addMessage("user", question); addMessage("assistant", answer); });
    if (!data.turns.length) $("#messages").append(welcomeMessage());
    document.querySelector('[data-view="ask"]').click();
  } catch (error) { toast(error.message, true); }
}

async function deleteSession(sessionId) {
  if (state.asking) { toast("请等待当前回答完成", true); return; }
  if (!window.confirm(`确定删除会话 ${sessionId}？此操作不可撤销。`)) return;
  try { await request(`/api/sessions/${sessionId}`, { method: "DELETE", token: true }); if (state.sessionId === sessionId) newChat(); await refreshSessions(); toast("会话已删除"); }
  catch (error) { toast(error.message, true); }
}

async function request(url, options = {}) {
  const headers = new Headers(options.headers || {});
  if (options.token) headers.set("X-P-Helper-Token", state.token);
  if (options.json) headers.set("Content-Type", "application/json");
  const response = await fetch(url, { ...options, headers });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || `请求失败 (${response.status})`);
  return data;
}

function emptyNode(iconName, title, copy) {
  const node = el("div", "empty-state"); node.append(icon(iconName), el("h2", "", title)); if (copy) node.append(el("p", "", copy)); return node;
}
function summaryBackButton() { const button = el("button", "mobile-back"); button.type = "button"; button.append(icon("arrow-left"), document.createTextNode("返回论文库")); button.addEventListener("click", () => { $("#summary-pane").classList.remove("mobile-open"); $(".paper-row.active")?.focus(); }); return button; }
function setConversationLocked(locked) { $("#top-k").disabled = locked; $("#all-papers").disabled = locked; $$("#scope-list input").forEach((input) => { input.disabled = locked; }); $("#clear-scope").disabled = locked; }
function syncScopeSelection(allMode = null) { state.scopeSelection = new Set($$("#scope-list input:checked").map((input) => input.value)); const inputs = $$("#scope-list input"); const allChecked = Boolean(inputs.length) && inputs.every((input) => input.checked); if (typeof allMode === "boolean") state.scopeAll = allMode; else if (inputs.length) state.scopeAll = allChecked; $("#all-papers").checked = allChecked; $("#all-papers").indeterminate = state.scopeSelection.size > 0 && state.scopeSelection.size < inputs.length; }
function openScope() { const panel = $("#scope-panel"); panel.classList.add("mobile-open"); panel.setAttribute("role", "dialog"); panel.setAttribute("aria-modal", "true"); $("#scope-backdrop").hidden = false; $("#scope-toggle").setAttribute("aria-expanded", "true"); document.body.classList.add("drawer-open"); $("#close-scope").focus(); }
function closeScope(returnFocus = false) { const panel = $("#scope-panel"); panel.classList.remove("mobile-open"); panel.removeAttribute("role"); panel.removeAttribute("aria-modal"); $("#scope-backdrop").hidden = true; $("#scope-toggle").setAttribute("aria-expanded", "false"); document.body.classList.remove("drawer-open"); if (returnFocus) $("#scope-toggle").focus(); }
function resetImportForm() { $("#pdf-files").value = ""; $("#input-path").value = ""; $("#file-selection").textContent = "尚未选择文件"; $("#file-selection").classList.remove("invalid"); $("#import-force").checked = false; }
function uploadProblem(files) { if (files.length > 100) return "每批最多上传 100 个 PDF"; if (files.some((file) => !file.name.toLowerCase().endsWith(".pdf"))) return "只能上传 PDF 文件"; if (files.reduce((sum, file) => sum + file.size, 0) > 1024 ** 3) return "单批上传总大小不能超过 1 GB"; return ""; }
function welcomeMessage() { const welcome = el("div", "welcome-message"); const mark = el("div", "welcome-icon"); mark.append(icon("scan-search")); const chips = el("div", "prompt-chips"); [["研究问题", "这篇论文解决了什么问题？"], ["核心方法", "核心方法有哪些关键步骤？"], ["结果与局限", "主要实验结果和局限是什么？"]].forEach(([label, prompt]) => { const button = el("button", "", label); button.type = "button"; button.dataset.prompt = prompt; chips.append(button); }); welcome.append(mark, el("h2", "", "从论文证据出发"), el("p", "", "选择论文范围，提出研究方法、实验结果或跨论文比较问题。"), chips); bindPromptButtons(welcome); return welcome; }
function bindPromptButtons(root) { root.querySelectorAll("[data-prompt]").forEach((button) => button.addEventListener("click", () => { $("#question").value = button.dataset.prompt; $("#question").dispatchEvent(new Event("input")); $("#question").focus(); })); }
function el(tag, className = "", text = null) { const node = document.createElement(tag); if (className) node.className = className; if (text !== null) node.textContent = text; return node; }
function icon(name) { const node = document.createElement("i"); node.dataset.lucide = name; return node; }
function icons() { if (window.lucide) window.lucide.createIcons(); }
function formatBytes(value) { return value < 1024 ** 2 ? `${Math.ceil(value / 1024)} KB` : `${(value / 1024 ** 2).toFixed(1)} MB`; }
function formatDate(value) { try { return new Intl.DateTimeFormat("zh-CN", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }).format(new Date(value)); } catch { return value; } }
function toast(message, error = false) { const node = el("div", `toast${error ? " error" : ""}`, message); $("#toast-region").append(node); setTimeout(() => node.remove(), 4200); }
