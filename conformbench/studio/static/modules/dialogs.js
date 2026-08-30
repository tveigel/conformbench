// ── Create / Copy / Delete dialogs ────────────────────────────────────────────
import { S } from "./state.js";
import { apiGet, apiPost, apiDelete, loadContexts } from "./api.js";
import { el, showToast } from "./dom-utils.js";
import { FAMILIES, CONTRAST_ROLES, PRIMARY_DELTA_TYPES } from "./constants.js";
// Lazy imports – used only inside async function bodies
import { renderSidebar } from "./sidebar.js";
import { openItem } from "./navigation.js";
import { renderActiveEditor } from "./editor.js";

export async function createNewItem() {
  if (S.isDirty && !confirm("Unsaved changes — discard and create a new item?")) return;

  // ── Build a small inline dialog ──
  const overlay = el("div", { class: "modal-overlay" });
  const dialog = el("div", { class: "modal-dialog" });

  const title = el("h3", { style: "margin:0 0 14px;" }, "New benchmark item");

  // Item ID
  const idGroup = el("div", { class: "field-group" });
  const idInput = el("input", { type: "text", placeholder: "e.g. F1-003" });
  idGroup.append(el("label", {}, "Item ID"), idInput, el("p", { class: "help-text" }, "A short unique identifier. Convention: F1-001, F2-003, etc."));

  // Delta type picker
  const famGroup = el("div", { class: "field-group" });
  const famSel = el("select", { class: "code-select" });
  for (const f of PRIMARY_DELTA_TYPES) famSel.append(el("option", { value: f.value }, f.label));
  const famDesc = el("p", { class: "help-text" });
  const updateDesc = () => { const sel = PRIMARY_DELTA_TYPES.find(f => f.value === famSel.value); famDesc.textContent = sel?.desc || ""; };
  updateDesc();
  famSel.addEventListener("change", updateDesc);
  famGroup.append(el("label", {}, "Primary delta type"), famSel, famDesc);

  // Contrast role
  const roleGroup = el("div", { class: "field-group" });
  const roleSel = el("select", { class: "code-select" });
  roleSel.append(el("option", { value: "" }, "— choose later —"));
  for (const r of CONTRAST_ROLES) roleSel.append(el("option", { value: r.value }, r.label));
  roleGroup.append(el("label", {}, "Contrast role"), roleSel);

  // Buttons
  const errMsg = el("div", { style: "color:#dc2626;font-size:12px;min-height:18px;margin-top:8px;" });
  const btnRow = el("div", { style: "display:flex;gap:8px;justify-content:flex-end;margin-top:16px;" });
  const cancelBtn = el("button", { class: "btn sm", type: "button" }, "Cancel");
  const createBtn = el("button", { class: "btn sm primary", type: "button" }, "Create");

  cancelBtn.addEventListener("click", () => overlay.remove());
  overlay.addEventListener("click", (e) => { if (e.target === overlay) overlay.remove(); });

  createBtn.addEventListener("click", async () => {
    const newId = idInput.value.trim();
    if (!newId) { errMsg.textContent = "Item ID is required."; idInput.focus(); return; }
    if (!/^[A-Za-z0-9_.-]+$/.test(newId)) { errMsg.textContent = "ID may only contain letters, numbers, _, ., and -"; idInput.focus(); return; }

    createBtn.disabled = true;
    createBtn.textContent = "Creating…";
    try {
      await apiPost("/api/items", {
        item_id: newId,
        primary_delta_type: famSel.value,
        family: famSel.value,
        contrast_role: roleSel.value || null,
      });
      overlay.remove();
      S.allItems = await apiGet("/api/items");
      renderSidebar();
      await openItem(newId);
      showToast(`✓ Item "${newId}" created`, "ok");
    } catch (e) {
      errMsg.textContent = e.message;
      createBtn.disabled = false;
      createBtn.textContent = "Create";
    }
  });

  // Enter key submits
  idInput.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); createBtn.click(); } });

  btnRow.append(cancelBtn, createBtn);
  dialog.append(title, idGroup, el("div", { class: "field-row" }, famGroup, roleGroup), errMsg, btnRow);
  overlay.appendChild(dialog);
  document.body.appendChild(overlay);
  idInput.focus();
}

export async function copyItemPrompt(itemId) {
  const newId = prompt(`Duplicate "${itemId}"\n\nEnter new item ID:`, itemId + "_copy");
  if (!newId || !newId.trim()) return;
  try {
    await apiPost(`/api/items/${itemId}/copy`, { new_item_id: newId.trim() });
    S.allItems = await apiGet("/api/items");
    showToast(`✓ Item duplicated as ${newId.trim()}`, "ok");
    renderSidebar();
  } catch (e) {
    showToast("✗ Copy failed: " + e.message, "err");
  }
}

export async function deleteItemPrompt(itemId) {
  if (!confirm(`Delete item "${itemId}"?\n\nThis removes the JSON file permanently.`)) return;
  try {
    await apiDelete(`/api/items/${itemId}`);
    S.allItems = await apiGet("/api/items");
    if (S.currentId === itemId) {
      S.currentId = null;
      S.currentData = null;
    }
    showToast(`✓ Item "${itemId}" deleted`, "ok");
    renderSidebar();
    await renderActiveEditor();
  } catch (e) {
    showToast("✗ Delete failed: " + e.message, "err");
  }
}

export async function copyContextPrompt(kind, ref) {
  const newRef = prompt(`Duplicate "${kind}/${ref}"\n\nEnter new ref (questionnaire/filename):`, ref + "_copy");
  if (!newRef || !newRef.trim()) return;
  try {
    await apiPost(`/api/contexts/${kind}/${ref}/copy`, { new_ref: newRef.trim() });
    await loadContexts();
    showToast(`✓ Context duplicated as ${newRef.trim()}`, "ok");
    renderSidebar();
  } catch (e) {
    showToast("✗ Copy failed: " + e.message, "err");
  }
}

export async function deleteContextPrompt(kind, ref) {
  if (!confirm(`Delete context "${kind}/${ref}"?\n\nThis removes the JSON file permanently.`)) return;
  try {
    await apiDelete(`/api/contexts/${kind}/${ref}`);
    await loadContexts();
    if (S.currentContextKind === kind && S.currentContextRef === ref) {
      S.currentContextKind = null;
      S.currentContextRef = null;
      S.currentContextData = null;
    }
    showToast(`✓ Context deleted`, "ok");
    renderSidebar();
    await renderActiveEditor();
  } catch (e) {
    showToast("✗ Delete failed: " + e.message, "err");
  }
}

// ── Continue as Next Turn ────────────────────────────────────────────────────

export async function continueAsNextTurn() {
  if (!S.currentId || !S.currentData) return;

  const d = S.currentData;
  const gold = d.gold_resulting_state;
  if (!gold || typeof gold !== "object" || !Object.values(gold).some(v => v != null && v !== "")) {
    showToast("Fill in the Gold Resulting State before continuing to the next turn.", "err");
    return;
  }

  if (S.isDirty && !confirm("You have unsaved changes. Save first, then continue?\n\nClick OK to save now, or Cancel to go back.")) return;

  // Auto-save if dirty
  if (S.isDirty) {
    try {
      const { saveItem } = await import("./editor.js");
      await saveItem();
    } catch (e) {
      showToast("Save failed — fix errors before continuing: " + e.message, "err");
      return;
    }
    // saveItem() swallows its own errors; verify the save actually succeeded
    if (S.isDirty) {
      showToast("Item must be saved before continuing to next turn.", "err");
      return;
    }
  }

  // ── Build the dialog ──
  const overlay = el("div", { class: "modal-overlay" });
  const dialog = el("div", { class: "modal-dialog" });

  const title = el("h3", { style: "margin:0 0 6px;" }, "Continue as next turn");
  const desc = el("p", { style: "font-size:12px;color:var(--text-3);margin:0 0 16px;line-height:1.5;" },
    "Create a new item whose ", el("strong", {}, "prior state"),
    " is the gold state of ", el("strong", {}, d.item_id),
    ". A new state context and history context will be created automatically.");

  // Summary of what will happen
  const summaryWrap = el("div", { style: "background:var(--bg-2);border:1px solid var(--border);border-radius:8px;padding:10px 12px;margin-bottom:14px;font-size:12px;line-height:1.6;" });
  const goldFieldCount = Object.values(gold).filter(v => v != null && v !== "").length;
  const utterancePreview = (d.current_utterance?.text || "").slice(0, 60) + ((d.current_utterance?.text || "").length > 60 ? "…" : "");
  summaryWrap.innerHTML = `
    <div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;"><span style="color:#10b981;font-weight:600;">●</span> <strong>New state</strong> ← gold state (${goldFieldCount} field${goldFieldCount !== 1 ? "s" : ""} filled)</div>
    <div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;"><span style="color:#6366f1;font-weight:600;">●</span> <strong>New history</strong> ← current history + "${utterancePreview || "(no utterance)"}"</div>
    <div style="display:flex;align-items:center;gap:6px;"><span style="color:#f59e0b;font-weight:600;">●</span> <strong>New item</strong> ← linked to both new contexts</div>`;

  // New item ID
  const sourcePrefix = d.item_id.replace(/[-_]?\d+$/, ""); // e.g. "F2-001" → "F2-"
  const sourceNum = d.item_id.match(/(\d+)$/)?.[1];
  const suggestedId = sourceNum
    ? sourcePrefix + String(Number(sourceNum) + 1).padStart(sourceNum.length, "0")
    : d.item_id + "_next";

  const idGroup = el("div", { class: "field-group" });
  const idInput = el("input", { type: "text", value: suggestedId, placeholder: "e.g. F2-002" });
  idGroup.append(el("label", {}, "New item ID"), idInput);

  // Delta type picker
  const famGroup = el("div", { class: "field-group" });
  const famSel = el("select", { class: "code-select" });
  for (const f of PRIMARY_DELTA_TYPES) famSel.append(el("option", { value: f.value, ...(f.value === (d.primary_delta_type || d.family) ? { selected: "" } : {}) }, f.label));
  famGroup.append(el("label", {}, "Primary delta type"), famSel);

  // Contrast role
  const roleGroup = el("div", { class: "field-group" });
  const roleSel = el("select", { class: "code-select" });
  roleSel.append(el("option", { value: "" }, "— same as source —"));
  for (const r of CONTRAST_ROLES) roleSel.append(el("option", { value: r.value, ...(r.value === d.contrast_role ? { selected: "" } : {}) }, r.label));
  roleGroup.append(el("label", {}, "Contrast role"), roleSel);

  // Buttons
  const errMsg = el("div", { style: "color:#dc2626;font-size:12px;min-height:18px;margin-top:8px;" });
  const btnRow = el("div", { style: "display:flex;gap:8px;justify-content:flex-end;margin-top:16px;" });
  const cancelBtn = el("button", { class: "btn sm", type: "button" }, "Cancel");
  const createBtn = el("button", { class: "btn sm primary", type: "button" }, "→ Create next turn");

  cancelBtn.addEventListener("click", () => overlay.remove());
  overlay.addEventListener("click", (e) => { if (e.target === overlay) overlay.remove(); });

  createBtn.addEventListener("click", async () => {
    const newId = idInput.value.trim();
    if (!newId) { errMsg.textContent = "Item ID is required."; idInput.focus(); return; }
    if (!/^[A-Za-z0-9_.-]+$/.test(newId)) { errMsg.textContent = "ID may only contain letters, numbers, _, ., and -"; idInput.focus(); return; }

    createBtn.disabled = true;
    createBtn.textContent = "Creating…";
    try {
      const goldToSend = d.gold_resulting_state;
      console.log("[continue-next-turn] source_item=%s, new_item=%s, gold_keys=%d, gold_desc=%s",
        d.item_id, newId,
        goldToSend ? Object.keys(goldToSend).length : 0,
        String(goldToSend?.description_of_accident || "").slice(0, 60));
      await apiPost(`/api/items/${d.item_id}/continue-next-turn`, {
        new_item_id: newId,
        primary_delta_type: famSel.value,
        family: famSel.value,
        contrast_role: roleSel.value || null,
        gold_resulting_state: goldToSend,
        current_utterance: d.current_utterance,
      });
      overlay.remove();
      S.allItems = await apiGet("/api/items");
      await loadContexts();
      renderSidebar();
      await openItem(newId);
      showToast(`✓ Next turn "${newId}" created — state & history auto-linked`, "ok");
    } catch (e) {
      errMsg.textContent = e.message;
      createBtn.disabled = false;
      createBtn.textContent = "→ Create next turn";
    }
  });

  // Enter key submits
  idInput.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); createBtn.click(); } });

  btnRow.append(cancelBtn, createBtn);
  dialog.append(title, desc, summaryWrap, idGroup, el("div", { class: "field-row" }, famGroup, roleGroup), errMsg, btnRow);
  overlay.appendChild(dialog);
  document.body.appendChild(overlay);
  idInput.focus();
  idInput.select();
}
