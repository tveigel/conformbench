// Editor orchestration and save logic for the public Studio.
import { S } from "./state.js";
import { apiGet, apiPut, apiPost } from "./api.js";
import { makeSection, setEditorHeader, setNavButtons, markClean, showToast } from "./dom-utils.js";
import { renderChecklist, updateChecklist } from "./checklist.js";
import { getItemSections } from "./item-sections.js";
import { renderSidebar } from "./sidebar.js";

export async function renderItemEditor() {
  const data = S.currentData;
  if (!data) return;

  document.getElementById("no-selection").style.display = "none";
  const editor = document.getElementById("editor");
  editor.style.display = "flex";
  setEditorHeader({ idText: data.item_id, titleText: data.title || "(untitled)", saveLabel: "Save item" });
  setNavButtons();

  const form = document.getElementById("editor-form");
  form.innerHTML = "";
  S._ctxContainer = null;

  for (const sectionSpec of getItemSections()) {
    const section = makeSection(
      sectionSpec.id,
      sectionSpec.n,
      sectionSpec.title,
      sectionSpec.sub,
      () => {},
      sectionSpec.startCollapsed || false,
    );
    form.appendChild(section);
    const body = section.querySelector(".section-body");
    try {
      await sectionSpec.render(body);
    } catch (err) {
      console.error(`[Section ${sectionSpec.n}] ${sectionSpec.title} render error:`, err);
      body.innerHTML = `<p style="color:#ef4444;padding:12px;font-size:12px;">Error rendering section: ${err.message}</p>`;
    }
  }

  renderChecklist();
  markClean();
}

export async function renderActiveEditor() {
  const saveBtn = document.getElementById("btn-save");
  if (saveBtn) {
    saveBtn.style.display = S.appMode === "items" ? "inline-flex" : "none";
    saveBtn.onclick = saveActiveRecord;
  }
  document.getElementById("editor-layout")?.classList.toggle("howto-layout", S.appMode === "howto");

  if (S.appMode === "eval") {
    const { renderEvalView } = await import("./eval.js?v=" + Date.now());
    await renderEvalView();
    return;
  }

  if (S.appMode === "howto") {
    const { renderHowToView } = await import("./how-to.js?v=" + Date.now());
    await renderHowToView();
    return;
  }

  if (!S.currentData) {
    document.getElementById("no-selection").style.display = "flex";
    document.getElementById("editor").style.display = "none";
    renderChecklist();
    return;
  }

  await renderItemEditor();
}

export async function saveItem() {
  if (!S.currentId || !S.currentData) return;
  try {
    const pendingId = S.currentData._pendingItemId;
    if (pendingId && pendingId !== S.currentId) {
      await apiPost(`/api/items/${S.currentId}/rename`, { new_item_id: pendingId });
      S.currentId = pendingId;
      S.currentData.item_id = pendingId;
      setEditorHeader({ idText: pendingId, titleText: S.currentData.title || "(untitled)", saveLabel: "Save item" });
    }
    delete S.currentData._pendingItemId;

    await apiPut(`/api/items/${S.currentId}`, S.currentData);
    markClean();
    showToast("Item saved", "ok");
    S.allItems = await apiGet("/api/items");
    renderSidebar();
    updateChecklist();
  } catch (err) {
    showToast("Save failed: " + err.message, "err");
  }
}

export async function saveActiveRecord() {
  if (S.appMode === "items") return saveItem();
}
