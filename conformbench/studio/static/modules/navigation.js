// Navigation for the public Studio.
import { S, persistState } from "./state.js";
import { apiGet } from "./api.js";
import { markClean } from "./dom-utils.js";
import { renderSidebar } from "./sidebar.js";
import { renderActiveEditor } from "./editor.js";

const PUBLIC_MODES = new Set(["items", "eval", "howto"]);

export async function setMode(nextMode, options = {}) {
  if (!PUBLIC_MODES.has(nextMode)) nextMode = "items";
  if (nextMode === S.appMode && !options.force) return;
  if (!options.skipConfirm && S.isDirty && nextMode !== S.appMode && !confirm("Unsaved changes - discard and switch?")) return;

  S.isDirty = false;
  S.appMode = nextMode;
  persistState();
  renderSidebar();
  await renderActiveEditor();
}

export async function openItem(id) {
  if (S.isDirty && !confirm("Unsaved changes - discard and switch?")) return;
  S.currentId = id;
  S.appMode = "items";
  S.currentData = await apiGet(`/api/items/${id}`);
  markClean();
  persistState();
  renderSidebar();
  await renderActiveEditor();
}

export async function navigateRelative(delta) {
  if (S.appMode !== "items") return;
  const idx = S.allItems.findIndex(item => item.item_id === S.currentId);
  const next = S.allItems[idx + delta];
  if (next) await openItem(next.item_id);
}
