/* ─────────────────────────────────────────────────────────────────────────────
   Benchmark Item + Context Authoring – ES Module Entry Point
───────────────────────────────────────────────────────────────────────────── */
import { S } from "./modules/state.js";
import { apiGet, loadContexts, loadScenarios } from "./modules/api.js";
import { updateSaveButtonLabel } from "./modules/dom-utils.js";
import { renderSidebar } from "./modules/sidebar.js";
import { renderChecklist } from "./modules/checklist.js";
import { updateEmptyState } from "./modules/dom-utils.js";
import { setMode, openItem, navigateRelative } from "./modules/navigation.js";
import { saveActiveRecord } from "./modules/editor.js";

// ── Keyboard shortcuts ───────────────────────────────────────────────────────

document.addEventListener("keydown", e => {
  if ((e.ctrlKey || e.metaKey) && e.key === "s") { e.preventDefault(); saveActiveRecord(); }
  if ((e.ctrlKey || e.metaKey) && e.key === "ArrowRight") { e.preventDefault(); navigateRelative(1); }
  if ((e.ctrlKey || e.metaKey) && e.key === "ArrowLeft") { e.preventDefault(); navigateRelative(-1); }
});

// ── Button wiring ────────────────────────────────────────────────────────────

function wire(id, handler) {
  const node = document.getElementById(id);
  if (node) node.onclick = handler;
}

wire("btn-save", saveActiveRecord);
wire("btn-prev", () => navigateRelative(-1));
wire("btn-next", () => navigateRelative(1));
wire("mode-items", () => setMode("items"));
wire("mode-eval", () => setMode("eval"));
wire("mode-howto", () => setMode("howto"));

// ── Resizable sidebar ────────────────────────────────────────────────────────

(function initSidebarResize() {
  const sidebar = document.getElementById("sidebar");
  const handle = document.getElementById("sidebar-resize-handle");
  if (!handle) return;

  // Restore saved width
  const savedW = localStorage.getItem("benchStudioSidebarW");
  if (savedW) {
    sidebar.style.width = savedW;
    document.documentElement.style.setProperty("--sidebar-w", savedW);
  }

  let startX, startW;

  handle.addEventListener("mousedown", e => {
    e.preventDefault();
    startX = e.clientX;
    startW = sidebar.offsetWidth;
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";

    function onMove(ev) {
      const newW = Math.max(220, Math.min(700, startW + ev.clientX - startX));
      sidebar.style.width = newW + "px";
      document.documentElement.style.setProperty("--sidebar-w", newW + "px");
    }
    function onUp() {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
      localStorage.setItem("benchStudioSidebarW", sidebar.style.width);
    }
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  });
})();

// ── Init ──────────────────────────────────────────────────────────────────────

async function init() {
  try {
    if (!["items", "eval", "howto"].includes(S.appMode)) S.appMode = "items";
    const [items, , , questionnaires] = await Promise.all([
      apiGet("/api/items"),
      loadContexts(),
      loadScenarios(),
      apiGet("/api/questionnaires").catch(() => []),
    ]);
    S.allItems = items;
    S.questionnaireNames = questionnaires;

    // Restore saved mode (setMode will render sidebar)
    await setMode(S.appMode, { force: true, skipConfirm: true });

    // Restore previously-open record
    if (S.appMode === "items" && S.currentId) {
      await openItem(S.currentId).catch(() => {});
    }

    renderChecklist();
    updateEmptyState();
    updateSaveButtonLabel();
  } catch (e) {
    document.getElementById("progress-label").textContent = "Failed to load authoring data";
    console.error("Init error:", e);
  }
}

init();
